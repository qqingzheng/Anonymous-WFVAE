import sys
import os

from .config import adapter_path
sys.path.append(os.path.join(adapter_path, "CV-VAE"))
from vae_models.modeling_vae import CVVAEModel

from ..registry import ModelRegistry

@ModelRegistry.register("CVVAE")
class CVVAEAdapter(CVVAEModel):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(tile_spatial_size=None, *args, **kwargs)
        self.tile_spatial_size = None
        
    def encode(self, x):
        return super().encode(x).latent_dist

    def decode(self, z):
        return super().decode(z).sample

    def forward(self, x):
        return super().forward(x)