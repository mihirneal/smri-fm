import numpy as np
import torch
from omegaconf import OmegaConf

from data.mri_data import collate, extract_sparse_wds_sample
from smri_mae.main_pretrain import make_plots
from smri_mae.model_mae import JointMAEAnatomyViT, MaskedAutoencoderViT
from smri_mae.visualization import patch_labels_to_volume


def test_anatomy_data_loading():
    raw = {
        "image_values.npy": np.ones(4, dtype=np.float16),
        "img_mask.npy": np.array([128], dtype=np.uint8),
        "meta.json": {},
        "anatomy.npz": {"counts": np.ones((8, 3), dtype=np.uint16)},
    }
    sample = extract_sparse_wds_sample(raw, include_anat=True)
    batch = collate([sample, sample], include_meta=False, include_anat=True)

    assert batch["anatomy_counts"].shape == (2, 8, 3)
    assert batch["anatomy_counts"].dtype == torch.int16


def test_standard_mae_accepts_the_shared_model_call():
    model = MaskedAutoencoderViT(
        img_size=16,
        patch_size=4,
        depth=0,
        embed_dim=32,
        num_heads=4,
        decoder_depth=0,
        decoder_embed_dim=32,
        decoder_num_heads=4,
    )
    images = torch.randn(1, 1, 16, 16, 16)

    loss = model(
        images,
        img_mask=torch.ones_like(images, dtype=torch.bool),
        anatomy_counts=None,
        mask_ratio=0.5,
        with_state=False,
    )

    assert torch.isfinite(loss)


def test_joint_mae_anatomy_forward_and_plots():
    model = JointMAEAnatomyViT(
        num_anatomy_classes=3,
        anatomy_loss_weight=0.25,
        img_size=16,
        patch_size=4,
        depth=0,
        embed_dim=32,
        num_heads=4,
        decoder_depth=0,
        decoder_embed_dim=32,
        decoder_num_heads=4,
    )
    images = torch.randn(2, 1, 16, 16, 16)
    img_mask = torch.ones_like(images, dtype=torch.bool)
    counts = torch.ones(2, 64, 3, dtype=torch.int16)

    loss, state = model(
        images,
        img_mask=img_mask,
        anatomy_counts=counts,
        mask_ratio=0.5,
    )

    assert torch.allclose(loss, state["image_loss"] + 0.25 * state["anatomy_loss"])
    assert state["pred_images"].shape == images.shape
    assert state["pred_anatomy_labels"].shape == (2, 64)
    visible_patches = model.pred_patchify(state["visible_mask"]).any(dim=-1)
    assert torch.equal(state["pred_anatomy_labels"] >= 0, visible_patches)

    state["anatomy_loss"].backward()
    assert model.anatomy_head.weight.grad is not None
    assert model.encoder.patch_embed.weight.grad is not None
    assert model.decoder.head.weight.grad is None

    model.zero_grad(set_to_none=True)
    train_loss = model(
        images,
        img_mask=img_mask,
        anatomy_counts=counts,
        mask_ratio=0.5,
        with_state=False,
    )
    train_loss.backward()
    assert model.decoder.head.weight.grad is not None
    assert model.anatomy_head.weight.grad is not None

    plots = make_plots(
        OmegaConf.create({"img_size": [16, 16, 16], "patch_size": 4}),
        batch={"image": images[:1], "img_mask": img_mask[:1]},
        state={
            "visible_mask": state["visible_mask"][:1],
            "pred_images": state["pred_images"][:1],
            "pred_mask": state["pred_mask"][:1],
            "target_anatomy_labels": state["target_anatomy_labels"][:1],
            "pred_anatomy_labels": state["pred_anatomy_labels"][:1],
        },
    )
    assert set(plots) == {"mask_pred", "masked_anatomy"}


def test_patch_labels_to_volume():
    labels = torch.arange(8).reshape(1, 8)
    volume = patch_labels_to_volume(labels, img_size=4, patch_size=2)

    assert volume.shape == (1, 1, 4, 4, 4)
    assert volume[0, 0, :2, :2, :2].unique().item() == 0
    assert volume[0, 0, 2:, 2:, 2:].unique().item() == 7
