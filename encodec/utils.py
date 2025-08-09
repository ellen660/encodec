# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Various utilities."""

from hashlib import sha256
from pathlib import Path
import typing as tp

import torch
import torchaudio
import random
import numpy as np


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _get_checkpoint_url(root_url: str, checkpoint: str):
    if not root_url.endswith('/'):
        root_url += '/'
    return root_url + checkpoint


def _check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, 'rb') as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum:
        raise RuntimeError(f'Invalid checksum for file {path}, '
                           f'expected {checksum} but got {actual_checksum}')


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


def print_model_details(model, log_path: str="model.txt") -> None:
        """Save the model details of the initialized model"""
        path = log_path
        with open(path, "w") as f:
            def write(s:str=""):
                f.write(s + "\n")

            write("=" * 60)
            write("📐 MODEL ARCHITECTURE")
            write("=" * 60)
            write(str(model))

            write("\n🔢 TOTAL TRAINABLE PARAMETERS")
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            write(f"Total parameters: {total_params:,}")
            write(f"Trainable parameters: {num_params:,}")

            write("\n📦 PARAMETER BREAKDOWN BY MODULE")
            write("=" * 60)

            module_params = []
            for name, module in model.named_modules():
                if len(list(module.parameters())) == 0:
                    continue  # skip modules with no parameters
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                total = sum(p.numel() for p in module.parameters())
                module_params.append((name, total, trainable))

            # Sort by trainable parameters (descending)
            module_params.sort(key=lambda x: x[2], reverse=True)

            for name, total, trainable in module_params:
                write(f"{name or '[root module]'}: total={total:,}, trainable={trainable:,}")

            write("\n🎯 ACTIVATION FUNCTIONS USED")
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.ReLU, torch.nn.GELU, torch.nn.Sigmoid, torch.nn.Tanh, torch.nn.Softmax, torch.nn.LeakyReLU, torch.nn.SELU, torch.nn.Mish)):
                    write(f"{name}: {module.__class__.__name__}")

            write("\n🔄 NORMALIZATION LAYERS")
            for name, module in model.named_modules():
                if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
                    write(f"{name}: {module.__class__.__name__}")

            write("\n🎲 INITIALIZATION STATS")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    mean = param.data.mean().item()
                    std = param.data.std().item()
                    if 'weight' in name:
                        # print(f'weight {name} shape {type(param)}')
                        write(f"[W] {name}: mean={mean:.4f}, std={std:.4f}")
                    elif 'bias' in name:
                        # breakpoint()
                        write(f"[B] {name}: mean={mean:.4f}, std={std:.4f}")
                    else:
                        write(f"[?] {name}: mean={mean:.4f}, std={std:.4f}")
                        

def set_random_seed(random_seed: int = 42) -> None:
    """Manually set the random state to make PyPOTS output reproducible results.

    Parameters
    ----------
    random_seed :
        The seed to be set for generating random numbers in PyPOTS.

    """
    globals()["RANDOM_SEED"] = random_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    # torch.backends.cudnn.deterministic = True  # This will slow down the training process.
    print(f"Have set the random seed as {random_seed} for numpy and pytorch.")
    
    
# speed up the dataloader pin memory 
# label noise on bwh and mgh ...
# use mesa as an external validation if it is significantly better 
# all of the dataset filtering is from the same way 
# 


# 4 models (eeg, ecg, resp, etc).
# downstream don't save the model 
# representation learning save model
# first train on all of them 

# downstreams at the same time, run them all 
# give you a table on AUC performance on each one 

# ppg, 
# validation set : just on gender, '

#external data set: mesa 
    # faced with dataset distribution

# just raun on ppg + all datasets
# 

# gender, age, antidepressants 
    