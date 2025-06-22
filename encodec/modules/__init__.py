# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Torch modules."""

# flake8: noqa
from .conv import (
    pad1d,
    unpad1d,
    NormConv1d,
    NormConvTranspose1d,
    NormConv2d,
    NormConvTranspose2d,
    SConv1d,
    SConvTranspose1d,
)
from .lstm import SLSTM
from .seanet import SEANetEncoder, SEANetDecoder
from .transformer import StreamingTransformerEncoder
import torch.nn as nn


def log_model_details(model, log_path):
    with open(log_path, "w") as f:
        def write(s=""):
            f.write(s + "\n")
            print(s)

        write("=" * 60)
        write("📐 MODEL ARCHITECTURE")
        write("=" * 60)
        write(str(model))

        write("\n🔢 TOTAL TRAINABLE PARAMETERS")
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        write(f"Total parameters: {total_params:,}")
        write(f"Trainable parameters: {num_params:,}")

        write("\n🎯 ACTIVATION FUNCTIONS USED")
        for name, module in model.named_modules():
            if isinstance(module, (nn.ELU, nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.Softmax, nn.LeakyReLU, nn.SELU, nn.Mish)):
                write(f"{name}: {module.__class__.__name__}")

        write("\n🔄 NORMALIZATION LAYERS")
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
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