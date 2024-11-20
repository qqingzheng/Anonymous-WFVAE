from diffusers import AutoencoderKLCogVideoX, StableVideoDiffusionPipeline
from ..registry import ModelRegistry

@ModelRegistry.register("CogVideoX")
class CogVideoXAdapter(AutoencoderKLCogVideoX):
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