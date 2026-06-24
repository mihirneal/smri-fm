import argparse
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from evaluation.backbones import build_backbone
from evaluation.heads import build_head
from evaluation.tasks import build_task
from evaluation.trainers import build_trainer


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg


def cli() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    main(args.config, args.overrides)


def main(config_path: str | Path, overrides: list[str] | None = None):
    cfg = load_config(config_path, overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task = build_task(cfg_dict["task"])
    target_spec = task.target_spec()

    backbone = build_backbone(cfg_dict["model"])
    head = build_head(cfg_dict["head"], target_spec=target_spec, input_dim=backbone.embed_dim)
    trainer = build_trainer(
        cfg_dict["mode"],
        cfg=cfg_dict,
        backbone=backbone,
        head=head,
        task=task,
    )
    return trainer.run()


if __name__ == "__main__":
    cli()
