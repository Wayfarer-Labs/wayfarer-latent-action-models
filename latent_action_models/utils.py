import  torch
import  einops as eo
from    torch import Tensor

# reused from adaworld repo 

def patchify(videos_bnchw: Tensor, size: int) -> Tensor:
    *_, H, W     = videos_bnchw.shape
    videos_bnchw    = videos_bnchw[:,:, :, 
                                    :H - (H % size), # -- round down to nearest patch size
                                    :W - (W % size)]
    patches_bnpd    = eo.rearrange(videos_bnchw,
                                "b t c (hn hp) (wn wp) -> b t (hn wn) (hp wp c)",
                                hp=size, wp=size)
    return patches_bnpd


def unpatchify(patches_bnpd: Tensor, size: int, h_out: int, w_out: int) -> Tensor:
    h_pad       = -h_out % size
    hn          = (h_out + h_pad) // size
    video_bnhwc = eo.rearrange(patches_bnpd,
                            "b t (hn wn) (hp wp c) -> b t (hn hp) (wn wp) c",
                            hp=size, wp=size, hn=hn)
    return video_bnhwc[:,:,:h_out, :w_out]
 