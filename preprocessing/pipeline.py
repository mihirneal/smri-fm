import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import ants
import nibabel as nib

log = logging.getLogger("preproc")

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DEFAULT_SYNTHSEG_CMD = (
    "uvx --python 3.11 --from 'git+https://github.com/MedARC-AI/SynthSeg.git' SynthSeg"
)
TEMPLATE_SPACE = "MNI152NLin2009cAsym"


def setup_logging(log_file: Path) -> None:
    logger = logging.getLogger("preproc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(fh)
    logger.addHandler(sh)


def _bids_entities(stem: str) -> dict[str, str]:
    entities = {}
    for part in stem.split("_"):
        if "-" in part:
            key, val = part.split("-", 1)
            entities[key] = val
    return entities


def preproc_output_path(input_path: Path, input_dir: Path) -> Path:
    stem = input_path.name.replace(".nii.gz", "")
    d = input_dir / "preprocessed"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_space-{TEMPLATE_SPACE}_desc-preproc.nii.gz"


def matrix_output_path(input_path: Path, input_dir: Path) -> Path:
    stem = input_path.name.replace(".nii.gz", "")
    d = input_dir / "derivatives" / "matrices"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}_from-native_to-{TEMPLATE_SPACE}_mode-image_xfm.mat"


def synthseg_output_paths(input_path: Path, input_dir: Path) -> tuple[Path, Path, Path]:
    stem = input_path.name.replace(".nii.gz", "")
    entities = _bids_entities(stem)
    sub = f"sub-{entities['sub']}" if "sub" in entities else "unknown"
    ses = f"ses-{entities['ses']}" if "ses" in entities else None
    d = input_dir / "derivatives" / "synthseg" / sub
    if ses:
        d = d / ses
    d.mkdir(parents=True, exist_ok=True)
    seg = d / f"{stem}_desc-synthseg_dseg.nii.gz"
    vol = d / f"{stem}_volumes.csv"
    qc = d / f"{stem}_qc.csv"
    return seg, vol, qc


def nib_to_ants(img: nib.Nifti1Image) -> ants.ANTsImage:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        ants_img = ants.image_read(tmp.name)
    Path(tmp.name).unlink(missing_ok=True)
    return ants_img


def ants_to_nib(ants_img: ants.ANTsImage) -> nib.Nifti1Image:
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        ants.image_write(ants_img, tmp.name)
        img = nib.load(tmp.name)
        img = nib.Nifti1Image(img.get_fdata(), img.affine, img.header)
    Path(tmp.name).unlink(missing_ok=True)
    return img


def rigid_register_to_template(
    img: nib.Nifti1Image,
    template_brain_path: Path,
    transform_path: Path,
) -> nib.Nifti1Image:
    fixed = ants.image_read(str(template_brain_path))
    moving = nib_to_ants(img)

    with tempfile.TemporaryDirectory() as tmpdir:
        outprefix = str(Path(tmpdir) / "rigid_")
        tx = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="Rigid",
            outprefix=outprefix,
        )
        fwdtransforms = tx["fwdtransforms"]
        rigid_transform = next(
            (Path(path) for path in fwdtransforms if str(path).endswith(".mat")), None
        )
        if rigid_transform is None:
            raise RuntimeError("ANTs rigid registration did not produce a matrix transform")

        registered = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=fwdtransforms,
            interpolator="bSpline",
        )
        shutil.copy2(rigid_transform, transform_path)

    return ants_to_nib(registered)


def _default_template_brain() -> Path:
    import templateflow.api as tflow
    return Path(str(tflow.get(
        TEMPLATE_SPACE, resolution=1, desc="brain", suffix="T1w", extension=".nii.gz"
    )))


def _parse_synthseg_cmd(command: str) -> list[str]:
    return shlex.split(command)


def run_synthseg(
    input_paths: list[Path],
    seg_paths: list[Path],
    vol_paths: list[Path],
    qc_paths: list[Path],
    synthseg_cmd: str,
    threads: int,
    cpu_only: bool,
    timeout: float = 600,
) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_txt = tmp_path / "inputs.txt"
        output_txt = tmp_path / "outputs.txt"
        vol_txt = tmp_path / "volumes.txt"
        qc_txt = tmp_path / "qc.txt"
        input_txt.write_text("\n".join(str(p) for p in input_paths) + "\n")
        output_txt.write_text("\n".join(str(p) for p in seg_paths) + "\n")
        vol_txt.write_text("\n".join(str(p) for p in vol_paths) + "\n")
        qc_txt.write_text("\n".join(str(p) for p in qc_paths) + "\n")

        cmd = _parse_synthseg_cmd(synthseg_cmd) + [
            "--i", str(input_txt),
            "--o", str(output_txt),
            "--vol", str(vol_txt),
            "--qc", str(qc_txt),
            "--parc",
            "--robust",
            "--threads", str(threads),
        ]
        if cpu_only:
            cmd.append("--cpu")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=os.environ.copy()
        )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr,
        )


def process_file(args: tuple) -> dict:
    input_path, input_dir, template_brain_path = args
    name = str(input_path.relative_to(input_dir))
    preproc_path = preproc_output_path(input_path, input_dir)
    xfm_path = matrix_output_path(input_path, input_dir)

    if preproc_path.exists() and xfm_path.exists():
        log.info("%s — already registered, skipping", name)
        return {"file": name, "status": "skipped"}

    log.info("%s — rigid registration to %s", name, TEMPLATE_SPACE)
    try:
        img = nib.load(input_path)
        registered = rigid_register_to_template(img, template_brain_path, xfm_path)
        nib.save(registered, preproc_path)
        log.info("%s — done → %s", name, preproc_path.name)
        return {"file": name, "status": "success"}
    except Exception as e:
        log.error("%s — failed: %s", name, e, exc_info=True)
        return {"file": name, "status": "failed", "error": str(e)}


def process_synthseg_batch(tasks: list[tuple]) -> list[dict]:
    results: list[dict] = []
    pending: list[tuple] = []

    for task in tasks:
        orig_path, preproc_path, input_dir, synthseg_cmd, synthseg_threads, cpu_only = task
        name = str(orig_path.relative_to(input_dir))
        seg_path, vol_path, qc_path = synthseg_output_paths(orig_path, input_dir)
        if seg_path.exists() and vol_path.exists() and qc_path.exists():
            log.info("%s — SynthSeg already done, skipping", name)
            results.append({"file": name, "status": "skipped"})
        else:
            pending.append(
                (
                    name,
                    orig_path,
                    preproc_path,
                    seg_path,
                    vol_path,
                    qc_path,
                    synthseg_cmd,
                    synthseg_threads,
                    cpu_only,
                )
            )

    if not pending:
        return results

    _, _, _, _, _, _, synthseg_cmd, synthseg_threads, cpu_only = pending[0]

    try:
        preproc_paths = [preproc_path for _, _, preproc_path, *_ in pending]
        seg_paths = [seg_path for _, _, _, seg_path, *_ in pending]
        vol_paths = [vol_path for _, _, _, _, vol_path, *_ in pending]
        qc_paths = [qc_path for _, _, _, _, _, qc_path, *_ in pending]

        log.info("SynthSeg: running on %d scan(s)", len(pending))
        run_synthseg(
            preproc_paths,
            seg_paths,
            vol_paths,
            qc_paths,
            synthseg_cmd,
            synthseg_threads,
            cpu_only,
            timeout=600 * len(pending),
        )

        for name, _, _, seg_path, vol_path, qc_path, *_ in pending:
            log.info("%s — SynthSeg done → %s", name, seg_path.name)
            log.info("%s — SynthSeg sidecars → %s, %s", name, vol_path.name, qc_path.name)
            results.append({"file": name, "status": "success"})

    except subprocess.CalledProcessError as e:
        log.error("SynthSeg batch failed: %s", e, exc_info=True)
        for label, text in (("stderr", e.stderr), ("stdout", e.output)):
            if text := (text or "").strip():
                log.error("SynthSeg %s:\n%s", label, text)
        for name, *_ in pending:
            results.append({"file": name, "status": "failed", "error": str(e)})
    except Exception as e:
        log.error("SynthSeg batch failed: %s", e, exc_info=True)
        for name, *_ in pending:
            results.append({"file": name, "status": "failed", "error": str(e)})

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Anat preprocessing pipeline")
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input directory containing .nii.gz files.",
    )
    parser.add_argument(
        "--log-dir", default=None, type=Path, dest="log_dir",
        help="Log directory (defaults to <input>/derivatives/logs).",
    )
    parser.add_argument(
        "--log-file", default=None, type=Path, dest="log_file",
        help="Explicit log file path (defaults to <log-dir>/run.log).",
    )
    parser.add_argument(
        "--template-brain", default=None, type=Path, dest="template_brain",
        help="Template brain image for rigid registration (defaults to MNI152NLin2009cAsym res-1 via templateflow).",
    )
    parser.add_argument("--itk-threads", default=2, type=int, dest="itk_threads",
                        help="CPU threads for ITK/ANTs.")
    parser.add_argument("--cpu", action="store_true", default=False,
                        help="Force SynthSeg CPU-only mode.")
    parser.add_argument("--synthseg-threads", default=8, type=int, dest="synthseg_threads",
                        help="CPU threads per SynthSeg call.")
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--preproc-only",
        action="store_true",
        default=False,
        dest="preproc_only",
        help="Run only rigid registration / preprocessing and skip SynthSeg.",
    )
    stage.add_argument(
        "--synthseg-only",
        action="store_true",
        default=False,
        dest="synthseg_only",
        help="Run only SynthSeg using existing preprocessed files.",
    )
    parser.add_argument("--batch-id", default=0, type=int, dest="batch_id",
                        help="Batch index (0-based) for SLURM/GNU parallel partitioning.")
    parser.add_argument("--batch-size", default=None, type=int, dest="batch_size",
                        help="Number of files per batch.")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    log_dir = (args.log_dir or input_dir / "logs").resolve()
    template_brain = (args.template_brain or _default_template_brain()).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not template_brain.exists():
        raise FileNotFoundError(f"Template brain not found: {template_brain}")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = (args.log_file.resolve() if args.log_file else log_dir / "run.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.itk_threads)

    log.info("Input: %s", input_dir)
    log.info("Preproc: %s", input_dir / "preprocessed")
    log.info("Matrices: %s", input_dir / "derivatives" / "matrices")
    log.info("SynthSeg: %s", input_dir / "derivatives" / "synthseg")
    log.info("Template brain: %s", template_brain)
    log.info("SynthSeg command: %s", DEFAULT_SYNTHSEG_CMD)

    files = sorted((input_dir / "images").rglob("*.nii.gz"))
    if not files:
        log.info("No .nii.gz files found in %s", input_dir)
        return

    if args.batch_size is not None:
        start = args.batch_id * args.batch_size
        stop = start + args.batch_size
        files = files[start:stop]

    log.info(
        "Processing %d file(s) (batch_id=%d, batch_size=%s)",
        len(files), args.batch_id, args.batch_size,
    )

    if not files:
        log.info("No files in this batch.")
        return

    reg_results: list[dict] = []
    reg_succeeded: set[str] = set()

    if not args.synthseg_only:
        reg_results = [process_file((f, input_dir, template_brain)) for f in files]
        reg_failed = [r for r in reg_results if r["status"] == "failed"]
        if reg_failed:
            log.error("Registration failed for: %s", [r["file"] for r in reg_failed])
        reg_succeeded = {r["file"] for r in reg_results if r["status"] in ("success", "skipped")}
        if args.preproc_only:
            if reg_failed:
                sys.exit(1)
            return
    else:
        reg_succeeded = {
            str(f.relative_to(input_dir))
            for f in files
            if preproc_output_path(f, input_dir).exists()
        }
        missing_preproc = [
            str(f.relative_to(input_dir))
            for f in files
            if str(f.relative_to(input_dir)) not in reg_succeeded
        ]
        if missing_preproc:
            log.error("Missing preprocessed inputs for SynthSeg-only mode: %s", missing_preproc)
            sys.exit(1)

    synthseg_tasks = [
        (f, preproc_output_path(f, input_dir), input_dir,
         DEFAULT_SYNTHSEG_CMD, args.synthseg_threads, args.cpu)
        for f in files
        if str(f.relative_to(input_dir)) in reg_succeeded
    ]
    synthseg_results = process_synthseg_batch(synthseg_tasks)

    ss_failed = [r for r in synthseg_results if r["status"] == "failed"]
    ss_succeeded = [r for r in synthseg_results if r["status"] == "success"]
    ss_skipped = [r for r in synthseg_results if r["status"] == "skipped"]
    log.info(
        "SynthSeg — %d succeeded, %d failed, %d skipped",
        len(ss_succeeded), len(ss_failed), len(ss_skipped),
    )
    if ss_failed:
        log.error("SynthSeg failed for: %s", [r["file"] for r in ss_failed])
        sys.exit(1)


if __name__ == "__main__":
    main()
