import sys
import os
from .config import adapter_path
sys.path.append(os.path.join(adapter_path, "Allegro"))
from allegro.models.vae.vae_allegro import AllegroAutoencoderKL3D

from ..registry import ModelRegistry

@ModelRegistry.register("Allegro")
class AllegroAdapter(AllegroAutoencoderKL3D):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def encode(self, x):
        return super().encode(x).latent_dist

    def decode(self, z):
        return super().decode(z).sample
    
    def forward(self, x):
        posterior = super().encode(x, local_batch_size=1).latent_dist
        if True:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = super().decode(z, local_batch_size=1).sample
        return dec