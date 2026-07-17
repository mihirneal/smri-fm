import json
import math
import tarfile
from io import BytesIO

import nibabel as nib
import numpy as np
import pytest
import torch

from data.anatomy_targets import (
    AnatomyVocabulary,
    DEFAULT_VOCABULARY_PATH,
    augment_wds_shard_with_anatomy,
    center_crop_or_pad,
    load_anatomy_vocabulary,
    patch_label_counts,
)
from data.mri_data import collate, extract_sparse_wds_sample
from smri_mae.model_mae import MaskedAnatomyViT
from smri_mae.visualization import patch_labels_to_volume


def _tiny_masked_anatomy(num_classes: int = 3) -> MaskedAnatomyViT:
    return MaskedAnatomyViT(
        num_anatomy_classes=num_classes,
        img_size=16,
        patch_size=4,
        depth=1,
        embed_dim=24,
        num_heads=4,
        decoder_depth=1,
        decoder_embed_dim=24,
        decoder_num_heads=4,
    )


def test_masked_anatomy_forward_uses_class_head_and_fp32_loss():
    model = _tiny_masked_anatomy()
    images = torch.randn(2, 1, 16, 16, 16, dtype=torch.bfloat16)
    img_mask = torch.ones_like(images, dtype=torch.bool)
    counts = torch.zeros(2, 64, 3, dtype=torch.int16)
    counts[:, :, 0] = 32
    counts[:, ::2, 1] = 16
    counts[:, 1::2, 2] = 16

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss, state = model(
            images,
            img_mask=img_mask,
            anatomy_counts=counts,
            mask_ratio=0.5,
            masking_policy="per_sample_pad",
        )

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert state["preds"].shape[-1] == 3
    assert state["target_anatomy_labels"].shape == (2, 64)
    assert state["pred_anatomy_labels"].shape == (2, 64)
    assert "pred_images" not in state


def test_masked_anatomy_loss_ignores_padded_prediction_tokens():
    model = _tiny_masked_anatomy()
    preds = torch.tensor(
        [
            [[3.0, 0.0, 0.0], [0.0, 100.0, 0.0]],
            [[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        ]
    )
    counts = torch.zeros(2, 64, 3, dtype=torch.int16)
    counts[0, 4, 0] = 8
    counts[1, 5, 1] = 8
    counts[1, 6, 2] = 8
    pred_ids = torch.tensor([[4, 0], [5, 6]])
    pred_token_mask = torch.tensor([[True, False], [True, True]])

    loss = model.forward_anatomy_loss(
        preds,
        counts,
        pred_ids,
        pred_token_mask=pred_token_mask,
    )
    expected = (
        -torch.log_softmax(preds[0, 0], dim=-1)[0]
        + (-torch.log_softmax(preds[1, 0], dim=-1)[1] - torch.log_softmax(preds[1, 1], dim=-1)[2])
        / 2
    ) / 2
    assert torch.allclose(loss, expected)


def test_official_parcellated_synthseg_vocabulary_has_98_brain_classes():
    vocabulary = load_anatomy_vocabulary(DEFAULT_VOCABULARY_PATH)

    assert vocabulary.num_classes == 98
    assert 0 not in vocabulary.values
    assert 3 not in vocabulary.values
    assert 42 not in vocabulary.values
    assert 1001 in vocabulary.values
    assert 2035 in vocabulary.values


def test_patch_label_counts_preserves_patch_distributions():
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[:2, :2, :2] = 2
    labels[2:, 2:, 2:] = 4
    labels[:2, 2:, :2] = np.array(
        [
            [[2, 2], [4, 4]],
            [[2, 2], [4, 4]],
        ]
    )

    counts = patch_label_counts(labels, [2, 4], patch_size=2)

    assert counts.shape == (8, 2)
    assert counts.dtype == np.uint16
    assert counts[0].tolist() == [8, 0]
    assert counts[2].tolist() == [4, 4]
    assert counts[7].tolist() == [0, 8]


def test_patch_label_counts_rejects_wrong_synthseg_vocabulary():
    labels = np.full((4, 4, 4), 999, dtype=np.int32)
    with pytest.raises(ValueError, match="absent from the vocabulary"):
        patch_label_counts(labels, [2, 4], patch_size=2)


def test_center_crop_or_pad_is_center_aligned():
    source = np.arange(3 * 5 * 3).reshape(3, 5, 3)
    fitted = center_crop_or_pad(source, (5, 3, 5), fill_value=-1)

    assert fitted.shape == (5, 3, 5)
    assert np.array_equal(fitted[1:4, :, 1:4], source[:, 1:4, :])
    assert (fitted[0] == -1).all()
    assert (fitted[-1] == -1).all()


def test_wds_extraction_and_collation_include_anatomy_counts():
    sample = {
        "image_values.npy": np.array([1.0, 2.0], dtype=np.float16),
        "img_mask.npy": np.array([128], dtype=np.uint8),
        "meta.json": {"key": "sample"},
        "anatomy.npz": {
            "counts": np.ones((8, 3), dtype=np.uint16),
            "label_values": np.array([2, 4, 5]),
        },
    }
    extracted = extract_sparse_wds_sample(
        sample,
        anatomy_key="anatomy.npz",
        expected_anatomy_label_values=[2, 4, 5],
    )
    batch = collate([extracted, extracted])

    assert batch["anatomy_counts"].shape == (2, 8, 3)
    assert batch["anatomy_counts"].dtype == torch.int16


def test_wds_extraction_rejects_reordered_anatomy_classes():
    sample = {
        "image_values.npy": np.array([1.0], dtype=np.float16),
        "img_mask.npy": np.array([128], dtype=np.uint8),
        "meta.json": {},
        "anatomy.npz": {
            "counts": np.ones((8, 3), dtype=np.uint16),
            "label_values": np.array([4, 2, 5]),
        },
    }
    with pytest.raises(ValueError, match="configured vocabulary"):
        extract_sparse_wds_sample(
            sample,
            anatomy_key="anatomy.npz",
            expected_anatomy_label_values=[2, 4, 5],
        )


def test_patch_labels_expand_to_model_volume():
    labels = torch.arange(8).reshape(1, 8)
    volume = patch_labels_to_volume(labels, img_size=(4, 4, 4), patch_size=2)

    assert volume.shape == (1, 1, 4, 4, 4)
    assert volume[0, 0, :2, :2, :2].unique().item() == 0
    assert volume[0, 0, 2:, 2:, 2:].unique().item() == 7
    assert math.prod(volume.shape[2:]) == 64


def test_wds_shard_augmentation_adds_compressed_patch_counts(tmp_path):
    source_root = tmp_path / "source"
    dseg_dir = source_root / "cohort" / "derivatives" / "synthseg"
    dseg_dir.mkdir(parents=True)
    dseg_path = dseg_dir / "subject_desc-synthseg_dseg.nii.gz"
    labels = np.zeros((4, 4, 4), dtype=np.int16)
    labels[:2] = 2
    labels[2:] = 4
    nib.save(nib.Nifti1Image(labels, np.eye(4)), dseg_path)

    meta = {
        "subset": "cohort",
        "native_stem": "subject",
        "sparse_image": {"dense_shape": [1, 4, 4, 4]},
    }
    meta_bytes = json.dumps(meta).encode()
    input_shard = tmp_path / "input.tar"
    with tarfile.open(input_shard, "w") as archive:
        info = tarfile.TarInfo("sample.meta.json")
        info.size = len(meta_bytes)
        archive.addfile(info, BytesIO(meta_bytes))

    output_shard = tmp_path / "output.tar"
    vocabulary = AnatomyVocabulary("tiny", (2, 4), ("left", "right"))
    stats = augment_wds_shard_with_anatomy(
        input_shard,
        output_shard,
        source_root=source_root,
        vocabulary=vocabulary,
        patch_size=2,
        expected_img_size=4,
    )

    assert stats == {"samples": 1, "targets_added": 1, "targets_kept": 0}
    with tarfile.open(output_shard) as archive:
        target_bytes = archive.extractfile("sample.anatomy.npz").read()
    with np.load(BytesIO(target_bytes)) as target:
        counts = target["counts"]
        label_values = target["label_values"]
    assert counts.shape == (8, 2)
    assert counts.sum(axis=0).tolist() == [32, 32]
    assert label_values.tolist() == [2, 4]
