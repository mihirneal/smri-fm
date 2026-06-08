import numpy as np
import pytest
import torch

from data.mri_data import collate, densify_sparse_image_batch, extract_sparse_wds_sample


def _packed_mask(bits: list[int]) -> np.ndarray:
    return np.packbits(np.asarray(bits, dtype=np.uint8))


def test_quantized_values_are_dequantized_during_densify() -> None:
    samples = [
        {
            "image_values": np.asarray([-128, 127], dtype=np.int8),
            "image_quantization": {"scale": 0.5, "value_min": -1.0, "qmin": -128},
            "img_mask": _packed_mask([1, 0, 1, 0]),
            "meta": {},
        },
        {
            "image_values": np.asarray([-128, -126], dtype=np.int8),
            "image_quantization": {"scale": 2.0, "value_min": 10.0, "qmin": -128},
            "img_mask": _packed_mask([0, 1, 1, 0]),
            "meta": {},
        },
    ]

    batch = collate(samples, include_meta=False)
    assert batch["image_values"].dtype == torch.int8

    images, masks = densify_sparse_image_batch(
        batch["image_values"],
        batch["img_mask"],
        (1, 2, 2),
        image_quantization=batch["image_quantization"],
        dtype=torch.float16,
    )

    expected = torch.tensor(
        [[[-1.0, 0.0], [126.5, 0.0]], [[0.0, 10.0], [14.0, 0.0]]],
        dtype=torch.float16,
    ).unsqueeze(1)
    torch.testing.assert_close(images, expected)
    assert masks.dtype == torch.bool


def test_legacy_float16_values_still_densify() -> None:
    samples = [
        {
            "image_values": np.asarray([1.5, -2.0], dtype=np.float16),
            "image_quantization": None,
            "img_mask": _packed_mask([1, 0, 1, 0]),
            "meta": {},
        }
    ]

    batch = collate(samples, include_meta=False)
    assert "image_quantization" not in batch
    images, _ = densify_sparse_image_batch(
        batch["image_values"], batch["img_mask"], (1, 2, 2), dtype=torch.float16
    )
    expected = torch.tensor([[[[1.5, 0.0], [-2.0, 0.0]]]], dtype=torch.float16)
    torch.testing.assert_close(images, expected)


def test_extract_quantized_sample_keeps_int8_values() -> None:
    sample = extract_sparse_wds_sample(
        {
            "image_values.npy": np.asarray([-128, 0, 127], dtype=np.int8),
            "img_mask.npy": _packed_mask([1, 1, 1, 0]),
            "meta.json": {
                "int8_quantization": {
                    "storage_dtype": "int8",
                    "scheme": "affine_per_sample_minmax",
                    "domain": "normalized",
                    "scale": 0.1,
                    "value_min": -2.0,
                    "qmin": -128,
                }
            },
        }
    )

    assert sample["image_values"].dtype == np.int8
    assert sample["image_quantization"] == {
        "scale": 0.1,
        "value_min": -2.0,
        "qmin": -128,
    }


def test_collate_rejects_mixed_quantization() -> None:
    samples = [
        {
            "image_values": np.asarray([0], dtype=np.int8),
            "image_quantization": {"scale": 1.0, "value_min": 0.0, "qmin": -128},
            "img_mask": _packed_mask([1]),
            "meta": {},
        },
        {
            "image_values": np.asarray([0], dtype=np.int8),
            "image_quantization": None,
            "img_mask": _packed_mask([1]),
            "meta": {},
        },
    ]

    with pytest.raises(ValueError, match="cannot mix quantized and unquantized"):
        collate(samples, include_meta=False)
