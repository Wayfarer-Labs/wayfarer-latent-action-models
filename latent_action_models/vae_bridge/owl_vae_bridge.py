import os, sys, yaml, torch
import omegaconf

sys.path.append("./owl_vaes")

from owl_vaes import from_pretrained


def load_cfg(path):
    with open(path, 'r') as f:
        cfg = omegaconf.OmegaConf.create(yaml.safe_load(f))
        if hasattr(cfg, 'model'): return cfg.model
        return cfg


class OWLDecodingPipeline:
    def __init__(self, cfg_path, ckpt_path):
        model = from_pretrained(cfg_path, ckpt_path)
        self.model = model.decoder.eval().bfloat16().cuda()
        del model.encoder
        # self.model = torch.compile(self.model, dynamic=False, fullgraph=True)

    @torch.no_grad()
    def __call__(self, x):
        return self.model(x)

def R3DCDecodingPipeline():
    return OWLDecodingPipeline(
        "/mnt/data/shahbuland/owl-vaes/configs/1x/no_depth.yml",
        "/mnt/data/shahbuland/owl-vaes/checkpoints/1x_rgb_no_depth/step_245000.pt"
    )

if __name__ == "__main__":
    x = R3DCDecodingPipeline()
    pass
