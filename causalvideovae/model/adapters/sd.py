from diffusers import AutoencoderKL
from ..registry import ModelRegistry
from einops import rearrange

@ModelRegistry.register("SD")
class SDAdapter(AutoencoderKL):
    
    def encode(self, x):
        x = rearrange(x, "b c t h w -> (b t) c h w")
        return super().encode(x).latent_dist

    def decode(self, z):
        dec = super().decode(z).sample
        dec = rearrange(dec, "(b t) c h w -> b c t h w", b=1)
        return dec
    
    
    def forward(self, sample):
        x = sample
        posterior = super().encode(x).latent_dist
        z = posterior.sample()
        dec = super().decode(z)
        return dec