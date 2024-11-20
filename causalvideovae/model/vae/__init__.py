from .modeling_causalvae import CausalVAEModel
from .modeling_wfvae import WFVAEModel
from .modeling_vqvae import VQVAEModel
from .modeling_wfvae_exp1 import WFVAEModel_Backbone
from .modeling_wfvae_exp2 import WFVAEModel_NOL3
from einops import rearrange
from torch import nn

class CausalVAEModelWrapper(nn.Module):
    def __init__(self, model_path, subfolder=None, cache_dir=None, use_ema=False, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)
        if use_ema:
            self.vae.init_from_ema(model_path)
            self.vae = self.vae.ema
    def encode(self, x):
        x = self.vae.encode(x).sample().mul_(0.18215)
        return x
    def decode(self, x):
        x = self.vae.decode(x / 0.18215)
        x = rearrange(x, 'b c t h w -> b t c h w').contiguous()
        return x

    def dtype(self):
        return self.vae.dtype