from diffusers import AutoencoderKL
from ..registry import ModelRegistry
import sys, os
from .config import adapter_path
sys.path.append(os.path.join(adapter_path, "EasyAnimate"))
from easyanimate.models.autoencoder_magvit import AutoencoderKLMagvit

@ModelRegistry.register("EasyAnimate")
class EasyAnimateAdapter(AutoencoderKLMagvit):
    
    def encode(self, x):
        return super().encode(x).latent_dist

    def decode(self, z):
        return super().decode(z).sample
    
    def forward(self, sample):
        x = sample
        posterior = super().encode(x).latent_dist
        z = posterior.sample()
        dec = super().decode(z)
        return dec