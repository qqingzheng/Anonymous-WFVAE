import sys
import os
from .config import adapter_path
sys.path.append(os.path.join(adapter_path, "Open-Sora"))
from opensora.models.vae.vae import VideoAutoencoderPipelineConfig, VideoAutoencoderPipeline
from ..registry import ModelRegistry

@ModelRegistry.register("OpenSora")
class OpenSoraAdapter(VideoAutoencoderPipeline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.micro_frame_size = None
        self.micro_batch_size = None
    # def to(self, *args):
    #     self.model.to(*args)
    #     return self
        
    def encode(self, x):
        print(x.shape)
        self.num_frame = x.size(2)
        z = super().encode(x)
        return z

    def decode(self, z):
        x_rec = super().decode(z, num_frames=self.num_frame)
        return x_rec

    def forward(self, x):
        return super().forward(x)