# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""EnCodec model implementation."""

import sys
import os
import math
from pathlib import Path
import typing as tp

# Get the directory of the current script (encodec_model.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add 'encodec' to sys.path
sys.path.append(current_dir)

import numpy as np
import torch
from torch import nn

import quantization as qt
import modules as m
from utils import _check_checksum, _linear_overlap_add, _get_checkpoint_url
from dataclasses import dataclass, field 

ROOT_URL = 'https://dl.fbaipublicfiles.com/encodec/v0/'

EncodedFrame = tp.Tuple[torch.Tensor, tp.Optional[torch.Tensor]]

@dataclass
class QuantizedResult:
    quantized: torch.Tensor
    codes: torch.Tensor
    bandwidth: torch.Tensor  # bandwidth in kb/s used, per batch item.
    commit_loss: tp.Optional[torch.Tensor] = None
    codebook_loss: tp.Optional[torch.Tensor] = None
    latents: tp.Optional[torch.Tensor] = None
    metrics: dict = field(default_factory=dict)

class EncodecModel(nn.Module):
    """EnCodec model operating on the raw waveform.
    Args:
        target_bandwidths (list of float): Target bandwidths.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        sample_rate (int): sample rate.
        channels (int): Number of channels.
        segment (float or None): None
    """
    def __init__(self,
                 encoder: m.SEANetEncoder,
                 decoder: m.SEANetDecoder,
                 quantizer: qt.ResidualVectorQuantizer,
                 target_bandwidths: tp.List[float], 
                 sample_rate: int, #10 fs
                 channels: int, #1
                 segment: tp.Optional[float] = None):
        super().__init__()
        self.bandwidth: tp.Optional[float] = None
        self.target_bandwidths = target_bandwidths
        self.encoder = encoder 
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.segment = segment        
        self.frame_rate = math.ceil(self.sample_rate / np.prod(self.encoder.ratios))
        self.bits_per_codebook = int(math.log2(self.quantizer.bins))
        self.n_q = quantizer.n_q
        assert 2 ** self.bits_per_codebook == self.quantizer.bins, "quantizer bins must be a power of 2."
        assert self.segment == None, f"expected segment to be none, got {self.segment}"
        assert self.sample_rate == 10
        assert self.channels == 1
        
    @property
    def segment_length(self) -> tp.Optional[int]:
        if self.segment is None:
            return None
        return int(self.segment * self.sample_rate)
    
    @property
    def codebooks(self):
        return self.quantizer.codebooks

    def encode(self, x: torch.Tensor) -> tp.List[EncodedFrame]:
        """Given a tensor `x`, returns a list of frames containing
        the discrete encoded codes for `x`

        Each frames is a tuple `(codebook, scale)`, with `codebook` of
        shape `[B, K, T]`, with `K` the number of codebooks.
        """
        _, channels, length = x.shape
        assert channels == 1

        segment_length = self.segment_length
        if segment_length is None:
            segment_length = length
            stride = length
        else:
            stride = self.segment_stride  # type: ignore
            assert stride is not None

        encoded_frames: tp.List[EncodedFrame] = []
        for offset in range(0, length, stride):
            frame = x[:, :, offset: offset + segment_length]
            encoded_frames.append(self._encode_frame(frame))
        return encoded_frames

    def _encode_frame(self, x: torch.Tensor) -> EncodedFrame:
        length = x.shape[-1]
        duration = length / self.sample_rate
        assert self.segment is None or duration <= 1e-5 + self.segment

        emb = self.encoder(x)
        quantized_result : QuantizedResult = self.quantizer(emb, self.frame_rate, self.bandwidth)
        codes = quantized_result.codes.transpose(0, 1)
        # codes is [B, K, T], with T frames, K nb of codebooks.

        encoded_frame = {
            'quantized': quantized_result.quantized, #[B, D, T]
            'codes': codes, #[B, N_q, T]
            'commit_loss': quantized_result.commit_loss, #[N_q,1]
            'codebook_loss': quantized_result.codebook_loss,
        }

        return encoded_frame

    def decode(self, encoded_frames: tp.List[EncodedFrame]) -> torch.Tensor:
        """Decode the given frames into a waveform.
        Note that the output might be a bit bigger than the input. In that case,
        any extra steps at the end can be trimmed.
        """
        segment_length = self.segment_length
        if segment_length is None:
            assert len(encoded_frames) == 1
            return self._decode_frame(encoded_frames[0])
        else:
            frames = [self._decode_frame(frame) for frame in encoded_frames]
            new_frames = _linear_overlap_add(frames, self.segment_stride or 1)
            return new_frames

    def _decode_frame(self, encoded_frame: EncodedFrame) -> torch.Tensor:
        out = self.decoder(encoded_frame['quantized'])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        frames = self.encode(x)
        # flatten the frames
        codes = torch.cat([frame['codes'] for frame in frames], dim=-1)
        commit_loss = torch.cat([frame['commit_loss'] for frame in frames], dim=-1)
        codebook_loss = torch.cat([frame['codebook_loss'] for frame in frames], dim=-1)

        return self.decode(frames)[:, :, :x.shape[-1]], codes, commit_loss, codebook_loss

    def set_target_bandwidth(self, bandwidth: float):
        if bandwidth not in self.target_bandwidths:
            raise ValueError(f"This model doesn't support the bandwidth {bandwidth}. "
                             f"Select one of {self.target_bandwidths}.")
        self.bandwidth = bandwidth

    @staticmethod
    def _get_model(target_bandwidths: tp.List[float],
                   sample_rate: int = 10,
                   channels: int = 1,
                   causal: bool = True,
                   model_norm: str = 'weight_norm',
                   segment: tp.Optional[float] = None,
                   ratios=[8, 5, 4, 2],
                   bins=256,
                   dimension=128,
                   ):

        encoder = m.SEANetEncoder(channels=channels, norm=model_norm, causal=causal, ratios=ratios, dimension=dimension)
        decoder = m.SEANetDecoder(channels=channels, norm=model_norm, causal=causal, ratios=ratios, dimension=dimension)
        n_q = int(1000 * target_bandwidths[-1] // (math.ceil(sample_rate / encoder.hop_length) * 10))

        quantizer = qt.ResidualVectorQuantizer(
            dimension=encoder.dimension,
            n_q=n_q,
            bins=bins,
            codebook_dim=encoder.dimension,
        )

        model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            target_bandwidths,
            sample_rate,
            channels,
            segment=segment,
        )

        return model
