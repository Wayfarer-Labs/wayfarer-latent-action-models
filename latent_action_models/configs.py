from __future__ import annotations
import  yaml
import  pathlib
import  dataclasses
from    dataclasses         import dataclass, field, asdict
from    typing              import Optional, Any, Literal
from    torch.utils.data    import DataLoader, Dataset


DatasetType = Literal["gta_4", "call_of_duty", "random"]

@dataclass
class DataConfig:
    batch_size:          int      = 8
    dataset_name:  DatasetType    = "gta_4"
    resolution:          int      = 256 # TODO use this.
    num_frames:          int      = 2
    samples_per_epoch:   int      = 1_000_000
    num_threads:         int      = 8

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataConfig:
        recognised     = {k: v for k, v in d.items() if k     in cls.__annotations__}
        not_recognised = {k: v for k, v in d.items() if k not in cls.__annotations__}
        
        if not_recognised:
            print(f"[DataConfig.from_dict] ‼ unknown keys ignored:")
            for k, v in not_recognised.items(): print(f"  - {k}: {v}")

        if not_recognised == {}: print(f"[DataConfig.from_dict] ✅ all keys recognised")
        if not_recognised != {}: print(f"[DataConfig.from_dict] ❌ some keys not recognised")
        
        return cls(**recognised)


@dataclass
class BaseTrainerConfig:
    data_config:     DataConfig             = field(default_factory=DataConfig)

    # -- checkpointing
    ckpt_dir:         str                   = "checkpoints"
    resume_checkpoint:Optional[str]         = None
    run_name:         Optional[str]         = None
    wandb_project:    str                   = "latent-action-models"

    # -- optimisation
    lr:              float                  = 1e-4
    weight_decay:    float                  = 0.0
    betas:           tuple[float, float]    = (0.9, 0.999)
    amp:             bool                   = False
    max_grad_norm:   Optional[float]        = None

    # -- loop control
    max_steps:       int                    = 100_000
    log_every:       int                    = 100
    ckpt_every:      int                    = 5_000
    val_every:       int                    = 2_500

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
    val_num_samples_umap:   int      = 1000
    val_num_samples_recon:  int      = 5

    @classmethod
    def from_yaml(cls, yaml_path: str | pathlib.Path) -> LatentActionModelTrainingConfig:
        with open(yaml_path, "r") as f: raw: dict[str, Any] = yaml.safe_load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> LatentActionModelTrainingConfig:
        data_cfg = DataConfig.from_dict(raw.pop("data", {}))

        # filter unknown keys & tuple-ify list fields where necessary
        field_names             = {f.name for f in dataclasses.fields(cls)}
        init_kw: dict[str, Any] = {"data_config": data_cfg}
        
        for k, v in raw.items():
            if k not in field_names:
                print(f"[LatentActionModelTrainingConfig.from_dict] ‼ unknown key '{k}' ignored")
                continue
            if isinstance(v, list): v = tuple(v)
            init_kw[k] = v

        return cls(**init_kw)


    def __repr__(self) -> str:
        parts = [f"{k}={v!r}" for k, v in asdict(self).items() if k != "data"]
        return f"{self.__class__.__name__}({', '.join(parts)}, data={self.data})"
