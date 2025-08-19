# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# This implementation is inspired from
# https://github.com/lucidrains/vector-quantize-pytorch
# which is released under MIT License. Hereafter, the original license:
# MIT License
#
# Copyright (c) 2020 Phil Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core vector quantization implementation."""

import typing as tp
import warnings

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F

# from .. import distrib
import encodec.distrib as distrib
import sys
import numpy as np


def default(val: tp.Any, d: tp.Any) -> tp.Any:
    return val if val is not None else d


def ema_inplace(moving_avg, new, decay: float):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))

def laplace_smoothing(x, n_categories: int, epsilon: float = 1e-5):
    return (x + epsilon) / (x.sum() + n_categories * epsilon)


def uniform_init(*shape: int):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t


def sample_vectors(samples, num: int):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]


def kmeans(samples, num_clusters: int, num_iters: int = 10):
    #samples (BxN, D) = first encoder batch, num_clusters
    dim, dtype = samples.shape[-1], samples.dtype

    means = sample_vectors(samples, num_clusters)

    for _ in range(num_iters):
        diffs = rearrange(samples, "n d -> n () d") - rearrange(
            means, "c d -> () c d"
        )
        dists = -(diffs ** 2).sum(dim=-1)

        buckets = dists.max(dim=-1).indices
        bins = torch.bincount(buckets, minlength=num_clusters)
        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        new_means.scatter_add_(0, repeat(buckets, "n -> n d", d=dim), samples)
        new_means = new_means / bins_min_clamped[..., None]

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins


class EuclideanCodebook(nn.Module):
    """Codebook with Euclidean distance.
    Args:
        dim (int): Dimension.
        codebook_size (int): Codebook size.
        kmeans_init (bool): Whether to use k-means to initialize the codebooks.
            If set to true, run the k-means algorithm on the first training batch and use
            the learned centroids as initialization.
        kmeans_iters (int): Number of iterations used for k-means algorithm at initialization.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        kmeans_init: int = False,
        kmeans_iters: int = 10,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        threshold_ema_dead_code: int = 2,
    ):
        super().__init__()
        self.decay = decay
        init_fn: tp.Union[tp.Callable[..., torch.Tensor], tp.Any] = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(codebook_size, dim)

        self.codebook_size = codebook_size
        
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.epsilon = epsilon
        self.threshold_ema_dead_code = threshold_ema_dead_code

        self.register_buffer("inited", torch.Tensor([False]))
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed", embed)
        self.register_buffer("embed_avg", embed.clone())
        
    # @torch.jit.ignore
#     def init_embed_(self, data):
#         if self.inited:
#             return

#         embed, cluster_size = kmeans(data, self.codebook_size, self.kmeans_iters)
#         self.embed.data.copy_(embed)
#         self.embed_avg.data.copy_(embed.clone())
#         self.cluster_size.data.copy_(cluster_size)
#         self.inited.data.copy_(torch.Tensor([True]))
#         # Make sure all buffers across workers are in sync after initialization
#         distrib.broadcast_tensors(self.buffers()) #broadcasts buffers from rank 0 to all other processes

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.inited: return 
        print(f'init rank {distrib.rank()} with kmeans {self.kmeans_init}')

        if distrib.rank() == 0:
            if self.kmeans_init:
                embed, _ = kmeans(data, self.codebook_size, self.kmeans_iters)
                self.embed.data.copy_(embed)
            else:
                pass # embed was uniforminited in construction
            self.inited.data.copy_(torch.tensor([True], device=self.inited.device))
            
        # Broadcast tensor from rank 0 to all other processess
        distrib.broadcast_tensors([self.inited])
        distrib.broadcast_tensors([self.embed])   # type: ignore

        if distrib.is_distributed():
            torch.distributed.barrier()

        # Sync initial cluster sizes and embed avg
        dtype = data.dtype
        embed_ind = self.quantize(data)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        self.cluster_size.data.copy_(embed_onehot.sum(0))
        self.embed_avg.data.copy_((data.t() @ embed_onehot).t())

        distrib.sync_buffer(buffers=[self.cluster_size], type='sum')
        distrib.sync_buffer(buffers=[self.embed_avg], type='sum')

    def replace_(self, samples, mask):
        modified_codebook = torch.where(
            mask[..., None], sample_vectors(samples, self.codebook_size), self.embed
        )
        self.embed.data.copy_(modified_codebook)
        
        
#     def expire_codes_(self, batch_samples):
#         if self.threshold_ema_dead_code == 0:
#             return

#         expired_codes = self.cluster_size < self.threshold_ema_dead_code
#         if not torch.any(expired_codes):
#             return

#         batch_samples = rearrange(batch_samples, "... d -> (...) d")
#         self.replace_(batch_samples, mask=expired_codes)
#         distrib.broadcast_tensors(self.buffers()) #only need to broadcast the new embed 

        
    def expire_codes_(self, batch_samples):
        """
        Requires: synced cluster size, embed
        Returns:
            expired_codes: bool mask of codes that were replaced
        """
        if self.threshold_ema_dead_code == 0:
            return torch.zeros(self.codebook_size, dtype=torch.bool, device=self.embed.device)

        if distrib.rank() == 0:
            expired_codes = self.cluster_size < self.threshold_ema_dead_code
            if torch.any(expired_codes):
                print(f'expiring codes')
                batch_samples = rearrange(batch_samples, "... d -> (...) d")
                self.replace_(batch_samples, mask=expired_codes)
        else:
            expired_codes = torch.zeros(self.codebook_size, dtype=torch.bool, device=self.embed.device)

        # Wait for rank 0 to finish replacement
        if distrib.is_distributed():
            torch.distributed.barrier()
            # Broadcast the mask so all ranks know which codes were replaced
            torch.distributed.broadcast(expired_codes, src=0)

        # Broadcast the updated embed
        distrib.broadcast_tensors([self.embed])

        return expired_codes
    
    def check_code_usage(self):
        """
        For debugging purpoess, checkes which codes are unused 
        Needs cluster size to be synced
        """
        if distrib.rank() == 0:
            expired_codes = self.cluster_size < self.threshold_ema_dead_code
            if torch.any(expired_codes):
                indices = torch.nonzero(expired_codes, as_tuple=True)[0]
                print(f'for threshhold {self.threshold_ema_dead_code} over {self.cluster_size.sum()} samples found {len(indices)} out of {self.codebook_size} dead codes')

    def preprocess(self, x):
        x = rearrange(x, "... d -> (...) d")
        return x

    def quantize(self, x):
        #Euclidean distance 
        embed = self.embed.t()

        dist = -( # alot of issues with this in that even if two vectors have the same director, doesn't matter if their magnitudes are different
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ embed
            + embed.pow(2).sum(0, keepdim=True)
        )

        embed_ind = dist.max(dim=-1).indices
        return embed_ind 

    def postprocess_emb(self, embed_ind, shape):
        return embed_ind.view(*shape[:-1])

    def dequantize(self, embed_ind):
        quantize = F.embedding(embed_ind, self.embed)
        return quantize

    def encode(self, x):
        shape = x.shape
        x = self.preprocess(x)
        embed_ind = self.quantize(x)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        return embed_ind

    def decode(self, embed_ind):
        quantize = self.dequantize(embed_ind)
        return quantize

    def forward(self, x):
        shape, dtype = x.shape, x.dtype
        x = self.preprocess(x)

        self.init_embed_(x) #sync here

        embed_ind = self.quantize(x) 
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = self.postprocess_emb(embed_ind, shape)
        quantize = self.dequantize(embed_ind) #no gradient for quantize

        if self.training:
            # We do the expiry of code at that point as buffers are in sync
            # and all the workers will take the same decision.
            cluster_new = embed_onehot.sum(0)        # [num_codes]
            embed_sum   = x.t() @ embed_onehot       # [d, num_codes]

            if distrib.is_distributed():
                handle1 = torch.distributed.all_reduce(cluster_new, op=torch.distributed.ReduceOp.SUM, async_op=True)
                handle2 = torch.distributed.all_reduce(embed_sum, op=torch.distributed.ReduceOp.SUM, async_op=True)
                handle1.wait()
                handle2.wait()

            ema_inplace(moving_avg = self.cluster_size, new = cluster_new, decay = self.decay)
            ema_inplace(moving_avg = self.embed_avg, new = embed_sum.t(), decay = self.decay)

            # Update expired codes 
            # replaced_codes = self.expire_codes_(x)  # return mask of replaced codes
            # self.embed_avg.data[replaced_codes] = self.embed.data[replaced_codes]

            # Normalize
            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)
            # self.check_code_usage()
            # mask = ~replaced_codes  # only normalize non-replaced codes
            # embed_normalized = self.embed_avg.clone()
            # embed_normalized[:, mask] = embed_normalized[:, mask] / cluster_size_corrected[mask].view(1, -1)

            # Copy back
            # self.embed.data[:, mask] = embed_normalized[:, mask]

            # embed_normalized[:, mask] /= cluster_size[mask].unsqueeze(1)
            # embed_normalized[mask] = embed_normalized[mask] / cluster_size[mask].unsqueeze(1)
            # self.embed.data[mask] = embed_normalized[mask]
            # self.cluster_size.data[replaced_codes] = self.threshold_ema_dead_code

        return quantize, embed_ind
    
    
#         if self.training:
#             # We do the expiry of code at that point as buffers are in sync
#             # and all the workers will take the same decision.
#             self.expire_codes_(x)
#             ema_inplace(moving_avg= self.cluster_size, new=embed_onehot.sum(0), decay=self.decay)
#             embed_sum = x.t() @ embed_onehot #sum of all vectors assigned to code j
#             ema_inplace(moving_avg=self.embed_avg, new=embed_sum.t(), decay=self.decay)
#             cluster_size = (
#                 laplace_smoothing(self.cluster_size, self.codebook_size, self.epsilon)
#                 * self.cluster_size.sum()
#             )
#             embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
#             self.embed.data.copy_(embed_normalized)

#         return quantize, embed_ind


class VectorQuantization(nn.Module):
    """Vector quantization implementation.
    Currently supports only euclidean distance.
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
        codebook_dim (int): Codebook dimension. If not defined, uses the specified dimension in dim.
        decay (float): Decay for exponential moving average over the codebooks.
        epsilon (float): Epsilon value for numerical stability.
        kmeans_init (bool): Whether to use kmeans to initialize the codebooks.
        kmeans_iters (int): Number of iterations used for kmeans initialization.
        threshold_ema_dead_code (int): Threshold for dead code expiration. Replace any codes
            that have an exponential moving average cluster size less than the specified threshold with
            randomly selected vector from the current batch.
        commitment_weight (float): Weight for commitment loss.
    """
    def __init__(
        self,
        dim: int,
        codebook_size: int,
        codebook_dim: tp.Optional[int] = None,
        decay: float = 0.99,
        epsilon: float = 1e-5,
        kmeans_init: bool = True,
        kmeans_iters: int = 50,
        threshold_ema_dead_code: int = 2,
        commitment_weight: float = 1.,
    ):
        super().__init__()
        _codebook_dim: int = default(codebook_dim, dim)

        requires_projection = _codebook_dim != dim
        self.project_in = (nn.Linear(dim, _codebook_dim) if requires_projection else nn.Identity())
        self.project_out = (nn.Linear(_codebook_dim, dim) if requires_projection else nn.Identity())

        self.epsilon = epsilon
        self.commitment_weight = 1.

        self._codebook = EuclideanCodebook(dim=_codebook_dim, codebook_size=codebook_size, kmeans_init=kmeans_init, kmeans_iters=kmeans_iters, decay=decay, epsilon=epsilon, threshold_ema_dead_code=threshold_ema_dead_code)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x):
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)
        embed_in = self._codebook.encode(x)
        return embed_in

    def decode(self, embed_ind):
        quantize = self._codebook.decode(embed_ind)
        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize

    def forward(self, x):
        device = x.device
        x = rearrange(x, "b d n -> b n d")
        x = self.project_in(x)

        quantize, embed_ind = self._codebook(x)
        
        # import torch.distributed as dist
        
        # buf_names = ["embed", "cluster_size", "embed_avg"]
        # for buf_name in buf_names:
        #     buf = getattr(self._codebook, buf_name)

        #     if dist.is_available() and dist.is_initialized():
        #         # gather tensors from all ranks to rank 0
        #         gathered = [torch.zeros_like(buf) for _ in range(dist.get_world_size())]
        #         dist.all_gather(gathered, buf)
                
        #         if dist.get_rank() == 0:
        #             # check if all tensors are equal
        #             for i, t in enumerate(gathered):
        #                 if not torch.allclose(t, gathered[0]):
        #                     print(f"Rank {i} differs from rank 0 in {buf_name}")
        #                     sys.exit()
        #             print("Buffer check done")
        #     else:
        #         print("Single-rank run; buffer shape:", buf.shape)

        if self.training:
            quantize = x + (quantize - x).detach()

        # loss = torch.tensor([0.0], device=device, requires_grad=self.training)

        # if self.training:
        #     warnings.warn('When using RVQ in training model, first check '
        #                   'https://github.com/facebookresearch/encodec/issues/25 . '
        #                   'The bug wasn\'t fixed here for reproducibility.')
        #     if self.commitment_weight > 0:
        #         commit_loss = F.mse_loss(quantize.detach(), x)
        #         # codebook_loss = F.mse_loss(quantize, x.detach())
        #         loss = loss + commit_loss * self.commitment_weight 

        quantize = self.project_out(quantize)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize, embed_ind,# loss


class ResidualVectorQuantization(nn.Module):
    """Residual vector quantization implementation.
    Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
    """
    def __init__(self, *, num_quantizers, **kwargs):
        super().__init__()
        codebook = VectorQuantization(**kwargs)
        # self.layers = nn.ModuleList(
        #     [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
        # )
        self.layers = nn.ModuleList([codebook for _ in range(num_quantizers)])
    
    @property
    def codebooks(self):
        count = 0
        codebooks = {}
        for layer in self.layers:
            codebooks[count] = layer.codebook
            count += 1
        print(f'codebooks {codebooks}')
        return codebooks

    def forward(self, x, n_q: tp.Optional[int] = None, return_quantized=False):
        quantized_out = 0.0
        residual = x
        # x   [-0.4486,  0.6813, -0.1261,  ...,  0.7333, -0.2821,  1.3599]]],
        all_losses = []
        all_indices = []
        quantized_stack = [] 
        soft = []

        n_q = n_q or len(self.layers)

        for layer in self.layers[:n_q]:
            quantized, indices = layer(residual)

            #fix issue at https://github.com/facebookresearch/encodec/issues/25
            residual = residual - quantized #NOTE: only first codebook commitment weights propagates back to x. 
            # residual = residual - quantized.detach() #NOTE: quantized.detach() will hurt the STE estimator gradient making it not 1.

            quantized_out = quantized_out + quantized
            
            # Moved commitment loss to be betweewn quantizedout and x instead of quantized and residual,
            # solves both problems above: propagtes to all codebooks, and preserve the STE estimator.
            loss = torch.tensor([0.0], device=x.device, requires_grad=self.training)

            if self.training:
                commit_loss = F.mse_loss(quantized_out.detach(), x)
                loss = loss + commit_loss  
                loss = loss / n_q
                
            all_indices.append(indices)
            all_losses.append(loss)
            quantized_stack.append(quantized)

        out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
        if return_quantized:
            return quantized_out, out_indices, out_losses, torch.stack(quantized_stack), 
        return quantized_out, out_indices, out_losses, 

    def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
        residual = x
        all_indices = []
        n_q = n_q or len(self.layers)
        for layer in self.layers[:n_q]:
            indices = layer.encode(residual)
            quantized = layer.decode(indices)
            residual = residual - quantized
            all_indices.append(indices)
        out_indices = torch.stack(all_indices)

        return out_indices

    def decode(self, q_indices: torch.Tensor, n_q=None) -> torch.Tensor:
        #N, B, T
        if n_q is None:
            n_q = len(self.layers)
        quantized_out = torch.tensor(0.0, device=q_indices.device)
        for i, indices in enumerate(q_indices):
            if i >= n_q:
                break
            layer = self.layers[i]
            quantized = layer.decode(indices)
            quantized_out = quantized_out + quantized
        return quantized_out


# import random
# class ResidualVectorQuantization(nn.Module):
#     """Residual vector quantization implementation.
#     Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf
#     """
#     def __init__(self, *, num_quantizers, **kwargs):
#         super().__init__()
#         #two codebooks
#         codebook = VectorQuantization(**kwargs)
#         codebook2 = VectorQuantization(**kwargs)
#         # codebook3 = VectorQuantization(**kwargs)
#         self.layers = nn.ModuleList([codebook])
#         #extend codebook2 for n_q > 2
#         for _ in range(num_quantizers-1):
#             self.layers.append(codebook2)
#         # self.layers = nn.ModuleList(
#         #     [VectorQuantization(**kwargs) for _ in range(num_quantizers)]
#         # )
#         #empty module liest
#         # self.layers = nn.ModuleList()
#         # for _ in range(num_quantizers//4):
#         #     codebook = VectorQuantization(**kwargs)
#         #     for _ in range(4):
#         #         self.layers.append(codebook)

#     def forward(self, x, n_q: tp.Optional[int] = None):
#         quantized_out = 0.0
#         residual = x

#         all_losses = []
#         all_indices = []

#         n_q = n_q or len(self.layers)
        
#         for i in range(1):
#             layer = self.layers[i]
#             quantized, indices, loss = layer(residual)
#             residual = residual - quantized
#             quantized_out = quantized_out + quantized

#             all_indices.append(indices)
#             all_losses.append(loss)
        
#         shuffled_layers = list(range(len(self.layers[1:n_q])))
#         random.shuffle(shuffled_layers)
#         for i in shuffled_layers:
#             layer = self.layers[1+i]
#             quantized, indices, loss = layer(residual)
#             residual = residual - quantized
#             quantized_out = quantized_out + quantized

#             all_indices.append(indices)
#             all_losses.append(loss)

#         out_losses, out_indices = map(torch.stack, (all_losses, all_indices))
#         return quantized_out, out_indices, out_losses

#     def encode(self, x: torch.Tensor, n_q: tp.Optional[int] = None) -> torch.Tensor:
#         residual = x
#         all_indices = []
#         n_q = n_q or len(self.layers)
#         for layer in self.layers[:n_q]:
#             indices = layer.encode(residual)
#             quantized = layer.decode(indices)
#             # print(f'indices, {indices}')
#             # print(f'residual {residual}')
#             # print(f'quantized {quantized}')
#             residual = residual - quantized
#             all_indices.append(indices)
#         out_indices = torch.stack(all_indices)

#         # sys.exit()
#         return out_indices

#     def decode(self, q_indices: torch.Tensor, n_q=None) -> torch.Tensor:
#         #N, B, T
#         if n_q is None:
#             n_q = len(self.layers)
#         quantized_out = torch.tensor(0.0, device=q_indices.device)
#         for i, indices in enumerate(q_indices):
#             if i >= n_q:
#                 break
#             layer = self.layers[i]
#             quantized = layer.decode(indices)
#             quantized_out = quantized_out + quantized
#             # print(f'indices {indices}')
#             # print(f'quantized {quantized}')
#             # breakpoint()
#         return quantized_out


if __name__ == "__main__":
    quantizer = ResidualVectorQuantization(
        num_quantizers=4, dim=256, codebook_size=256,
        kmeans_init=True, kmeans_iters=10, threshold_ema_dead_code=2,
    )

    for i in range(4):
        input = torch.rand((2, 256, 30), requires_grad=True)
        quantized, indices, losses = quantizer(input)
        print(quantized.shape, indices.shape, losses.shape)

        losses[i, 0].backward()
        print(input.grad)