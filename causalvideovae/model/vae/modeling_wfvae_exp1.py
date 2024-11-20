try:
    import torch_npu
    from causalvideovae.model.utils.npu_config import npu_config
except:
    torch_npu = None
    npu_config = None
    
from ..modeling_videobase import VideoBaseAE
from diffusers.configuration_utils import register_to_config
import torch
import torch.nn as nn
from ..modules import (
    ResnetBlock2D,
    ResnetBlock3D,
    Conv2d,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform3D,
    CausalConv3d,
    Normalize,
    AttnBlock3DFix,
    nonlinearity,
)
from .modeling_causalvae import Decoder as CausalDecoder
import torch.nn as nn
from ..utils.distrib_utils import DiagonalGaussianDistribution
import torch
from copy import deepcopy
import os
from ..registry import ModelRegistry
from einops import rearrange
from collections import deque
from ..utils.module_utils import resolve_str_to_obj, Module
from typing import List

class Encoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        dropout: float = 0.0,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        l1_dowmsample_block: str = "Downsample",
        l2_dowmsample_block: str = "Spatial2xTime2x3DDownsample",
    ) -> None:
        super().__init__()
        self.down1 = nn.Sequential(
            Conv2d(24, base_channels, kernel_size=3, stride=1, padding=1),
            *[
                ResnetBlock2D(
                    in_channels=base_channels,
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_dowmsample_block)(in_channels=base_channels, out_channels=base_channels),
        )
        self.down2 = nn.Sequential(
            Conv2d(
                base_channels,
                base_channels * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 2,
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_dowmsample_block)(base_channels * 2, base_channels * 2),
        )
        # Mid
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type)
            )
        self.mid = nn.Sequential(*mid_layers)

        self.norm_out = Normalize(base_channels * 4, norm_type=norm_type)
        self.conv_out = CausalConv3d(
            base_channels * 4, latent_dim * 2, kernel_size=3, stride=1, padding=1
        )
        
    def forward(self, x):
        h = self.down1(x)
        h = self.down2(h)
        h = self.mid(h)
        
        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)
            
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h

class Decoder(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        num_resblocks: int = 2,
        dropout: float = 0.0,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        norm_type: str = "groupnorm",
        t_interpolation: str = "nearest",
        l1_upsample_block: str = "Upsample",
        l2_upsample_block: str = "Spatial2xTime2x3DUpsample",
    ) -> None:
        super().__init__()
        self.conv_in = CausalConv3d(
            latent_dim, base_channels * 4, kernel_size=3, stride=1, padding=1
        )
        mid_layers = [
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
        ]
        if use_attention:
            mid_layers.insert(
                1, resolve_str_to_obj(attention_type)(in_channels=base_channels * 4, norm_type=norm_type)
            )
            
        self.mid = nn.Sequential(*mid_layers)

        self.up2 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * 4,
                    out_channels=base_channels * 4,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for _ in range(num_resblocks)
            ],
            resolve_str_to_obj(l2_upsample_block)(
                base_channels * 4, base_channels * 4, t_interpolation=t_interpolation
            ),
            ResnetBlock3D(
                in_channels=base_channels * 4,
                out_channels=base_channels * 4,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        self.up1 = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (4 if i == 0 else 2),
                    out_channels=base_channels * 2,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(num_resblocks)
            ],
            resolve_str_to_obj(l1_upsample_block)(in_channels=base_channels * 2, out_channels=base_channels * 2),
            ResnetBlock3D(
                in_channels=base_channels * 2,
                out_channels=base_channels * 2,
                dropout=dropout,
                norm_type=norm_type,
            ),
        )
        self.layer = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=base_channels * (2 if i == 0 else 1),
                    out_channels=base_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                )
                for i in range(2)
            ],
        )
    
        # Out
        self.norm_out = Normalize(base_channels, norm_type=norm_type)
        self.conv_out = Conv2d(base_channels, 24, kernel_size=3, stride=1, padding=1)

        
    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid(h)
        h = self.up2(h)
        h = self.up1(h)
        h = self.layer(h)
        if npu_config is None:
            h = self.norm_out(h)
        else:
            h = npu_config.run_group_norm(self.norm_out, h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

@ModelRegistry.register("WFVAE_Backbone")
class WFVAEModel_Backbone(VideoBaseAE):

    @register_to_config
    def __init__(
        self,
        latent_dim: int = 8,
        base_channels: int = 128,
        encoder_num_resblocks: int = 2,
        decoder_num_resblocks: int = 2,
        attention_type: str = "AttnBlock3DFix",
        use_attention: bool = True,
        dropout: float = 0.0,
        norm_type: str = "layernorm",
        t_interpolation: str = "trilinear",
        connect_res_layer_num: int = 1,
        scale: List[float] = [0.18215, 0.18215, 0.18215, 0.18215],
        shift: List[float] = [0, 0, 0, 0],
    ) -> None:
        super().__init__()
        self.use_tiling = False
        # Hardcode for now
        self.use_quant_layer = False

        self.encoder = Encoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=encoder_num_resblocks,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            attention_type=attention_type
        )
        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            num_resblocks=decoder_num_resblocks,
            dropout=dropout,
            use_attention=use_attention,
            norm_type=norm_type,
            t_interpolation=t_interpolation,
            attention_type=attention_type
        )

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def encode(self, x):
        wt = HaarWaveletTransform3D().to(x.device, dtype=x.dtype)
        x = wt(x)
        h = self.encoder(x)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)
        return posterior,
    

    def decode(self, z):
        if self.use_quant_layer:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)
        wt = InverseHaarWaveletTransform3D().to(dec.device, dtype=dec.dtype)
        dec = wt(dec)
        return dec,

    def forward(self, input, sample_posterior=True):
        posterior, = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec, = self.decode(z)
        return dec, posterior

    def get_last_layer(self):
        if hasattr(self.decoder.conv_out, "conv"):
            return self.decoder.conv_out.conv.weight
        else:
            return self.decoder.conv_out.weight

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling
        self._set_causal_cached(use_tiling)
        
    def disable_tiling(self):
        self.enable_tiling(False)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")
        print("init from " + path)

        if (
            "ema_state_dict" in sd
            and len(sd["ema_state_dict"]) > 0
            and os.environ.get("NOT_USE_EMA_MODEL", 0) == 0
        ):
            print("Load from ema model!")
            sd = sd["ema_state_dict"]
            sd = {key.replace("module.", ""): value for key, value in sd.items()}
        elif "state_dict" in sd:
            print("Load from normal model!")
            if "gen_model" in sd["state_dict"]:
                sd = sd["state_dict"]["gen_model"]
            else:
                sd = sd["state_dict"]

        keys = list(sd.keys())

        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(missing_keys, unexpected_keys)