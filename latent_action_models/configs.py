from __future__ import annotations
import  yaml
import  pathlib
import  dataclasses
from    dataclasses import dataclass, field, asdict
from    typing      import Callable, Optional, Any


@dataclass
class DataConfig:
    batch_size:          int      = 8
    data_root:           str      = "./data"
    env_source:          str      = "game"
    padding:             str      = "repeat"
    randomize:           bool     = True
    resolution:          int      = 256
    num_frames:          int      = 2
    output_format:       str      = "t h w c"
    samples_per_epoch:   int      = 1_000_000
    sampling_strategy:   str      = "pi"


@dataclass
class BaseTrainerConfig:
    # -- checkpointing
    ckpt_dir:         str                 = "checkpoints"
    resume_checkpoint:Optional[str]       = None

    # -- optimisation
    lr:              float                = 1e-4
    weight_decay:    float                = 0.0
    betas:           tuple[float, float]  = (0.9, 0.999)
    amp:             bool                 = False
    max_grad_norm:   Optional[float]      = None

    # -- loop control
    batch_size:      int                  = 8
    max_steps:       int                  = 100_000
    log_every:       int                  = 100
    ckpt_every:      int                  = 5_000

    # -- LR schedule (LambdaLR) TODO Fix this to be more flexible
    lr_lambda: Callable[[int], float]     = field(
        default_factory=lambda: (lambda step: 1.0)
    )


@dataclass
class LatentActionModelTrainingConfig(BaseTrainerConfig):
    video_dims:      tuple[int, int] = (64, 64)
    in_dim:          int             = 3
    model_dim:       int             = 64
    vae_dim:         int             = 16
    patch_size:      int             = 8
    num_enc_blocks:  int             = 4
    num_dec_blocks:  int             = 4
    num_heads:       int             = 4
    dropout:         float           = 0.0

    beta:            float           = 0.0    # KL weight
    @classmethod
    def from_yaml(cls, yaml_path: str | pathlib.Path) -> "LatentActionModelTrainingConfig":
        with open(yaml_path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        # -- build nested DataConfig
        data_dict = raw.pop("data", {})
        data_cfg  = DataConfig(**{
            k: v for k, v in data_dict.items()
            if k in DataConfig.__annotations__
        })

        field_names = {f.name for f in dataclasses.fields(cls)}
        init_kwargs = {"data": data_cfg}

        for k, v in raw.items():
            if k not in field_names:
                print(f"[from_yaml] ‼  unknown key '{k}' ignored")
                continue

            # tuple-ify list fields (e.g. video_dims, betas)
            if k in ("video_dims", "betas") and isinstance(v, list):
                v = tuple(v)
            init_kwargs[k] = v

        cfg = cls(**init_kwargs)

        # keep data.batch_size in sync with trainer batch_size
        cfg.data.batch_size = cfg.batch_size

        return cfg


    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in asdict(self).items() if k != "data"]
        return f"{self.__class__.__name__}({', '.join(parts)}, data={self.data})"
