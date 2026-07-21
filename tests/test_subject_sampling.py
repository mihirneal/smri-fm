import io
import json
import random
import tarfile

import numpy as np
import pytest

from data.mri_data import (
    SUBJECT_INDEX_VERSION,
    SubjectBalancedSparseDataset,
    build_subject_index,
    subject_index_is_current,
)


def _write_index(path, subjects):
    path.write_text(
        json.dumps(
            {
                "version": SUBJECT_INDEX_VERSION,
                "include_anat": False,
                "signature": [],
                "shards": ["unused.tar"],
                "subjects": subjects,
            }
        )
    )


def _reference(key):
    return [0, 0, 1, 1, 1, key]


def test_subject_sampling_balances_subjects_instead_of_scans(tmp_path):
    index_path = tmp_path / "index.json"
    _write_index(
        index_path,
        {
            "source_sub-01": [_reference("source_sub-01_ses-01_T1w")],
            "source_sub-02": [
                _reference("source_sub-02_ses-01_T1w"),
                _reference("source_sub-02_ses-02_T1w"),
                _reference("source_sub-02_ses-03_T1w"),
            ],
        },
    )
    subject_dataset = SubjectBalancedSparseDataset(
        index_path,
        seed=11,
        sampling="subject",
    )
    scan_dataset = SubjectBalancedSparseDataset(
        index_path,
        seed=11,
        sampling="scan",
    )

    subject_rng = random.Random(17)
    scan_rng = random.Random(17)
    subject_draws = [subject_dataset._draw_reference(subject_rng)[-1] for _ in range(4_000)]
    scan_draws = [scan_dataset._draw_reference(scan_rng)[-1] for _ in range(4_000)]
    subject_fraction = sum("sub-01" in key for key in subject_draws) / len(subject_draws)
    scan_fraction = sum("sub-01" in key for key in scan_draws) / len(scan_draws)

    assert 0.45 < subject_fraction < 0.55
    assert 0.20 < scan_fraction < 0.30


def test_sampling_policy_is_validated(tmp_path):
    index_path = tmp_path / "index.json"
    _write_index(index_path, {"source_sub-01": [_reference("sample")]})

    with pytest.raises(ValueError, match="sampling"):
        SubjectBalancedSparseDataset(index_path, seed=11, sampling="session")


def _add_numpy_member(archive, name, array):
    stream = io.BytesIO()
    np.save(stream, array)
    payload = stream.getvalue()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def _add_anatomy_member(archive, name, counts):
    stream = io.BytesIO()
    np.savez(stream, counts=counts)
    payload = stream.getvalue()
    member = tarfile.TarInfo(name)
    member.size = len(payload)
    archive.addfile(member, io.BytesIO(payload))


def test_subject_index_reads_anatomy_targets(tmp_path):
    shard_path = tmp_path / "shard.000000.tar"
    key = "source_sub-01_ses-01_T1w"
    counts = np.arange(24, dtype=np.uint16).reshape(8, 3)
    with tarfile.open(shard_path, "w") as archive:
        _add_numpy_member(
            archive,
            f"{key}.image_values.npy",
            np.asarray([1.25, 2.5], dtype=np.float16),
        )
        _add_numpy_member(
            archive,
            f"{key}.img_mask.npy",
            np.asarray([192], dtype=np.uint8),
        )
        _add_anatomy_member(archive, f"{key}.anatomy.npz", counts)

    index_path = tmp_path / "index.json"
    build_subject_index(index_path, [shard_path], include_anat=True)
    assert subject_index_is_current(index_path, [shard_path], include_anat=True)

    dataset = SubjectBalancedSparseDataset(
        index_path,
        seed=3,
        read_ahead=1,
        include_anat=True,
    )
    sample = next(iter(dataset))

    np.testing.assert_array_equal(sample["image_values"], [1.25, 2.5])
    np.testing.assert_array_equal(sample["img_mask"], [192])
    np.testing.assert_array_equal(sample["anatomy_counts"], counts)
