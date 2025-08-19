import random
import time

import torch
from losses import disc_loss, total_loss
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.distributed as dist

def reduce_mean(tensor, world_size):
    # tensor: torch scalar on GPU
    if world_size < 2:
        return tensor
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor

def gather_codes(codes, world_size):
    # Ensure tensor
    if not torch.is_tensor(codes):
        codes = torch.tensor(codes, device=codes.device if hasattr(codes, "device") else "cuda")
    
    # Prepare list to gather into
    gathered = [torch.zeros_like(codes) for _ in range(world_size)]
    dist.all_gather(gathered, codes)
    
    # Concatenate along batch dimension
    return torch.cat(gathered, dim=0)

def train_one_step(
    metrics,
    epoch,
    optimizer,
    optimizer_disc,
    scheduler,
    disc_scheduler,
    model,
    disc,
    train_loader,
    config,
    writer,
    freq_loss,
    device,
    rank,
    log_dir
):
    """
    metrics: your metrics object
    epoch: current epoch number
    optimizer: generator optimizer
    optimizer_disc: discriminator optimizer
    scheduler: generator LR scheduler
    disc_scheduler: discriminator LR scheduler
    model: DDP wrapped generator model
    disc: discriminator model (also should be DDP wrapped if distributed)
    train_loader: distributed dataloader with DistributedSampler or custom IterableDataset with sharding
    config: config object
    writer: tensorboard writer
    freq_loss: frequency loss function
    device: torch device for current rank
    rank: current process rank (int), e.g. dist.get_rank()
    """

    model.train()
    if config.discrim.train_discriminator and epoch >= config.discrim.train_discriminator_start_epoch:
        disc.train()

    epoch_loss = 0
    start_data_time = time.time()
    data_loading = 0
    to_device = 0
    forward_time = 0
    all_codes = []

    for i, (item, ds_id) in enumerate(tqdm(train_loader,desc=f"Training Epoch {epoch}",unit="batch",disable=(rank != 0),)):
        x = item["x"]
        data_loading += time.time() - start_data_time

        start_to_device = time.time()
        x = x.to(device, non_blocking=True)
        to_device += time.time() - start_to_device

        start_forward_time = time.time()
        x_hat, codes, commit_loss, codebook_loss = model(x)

        train_generator = config.discrim.train_discriminator and epoch >= config.discrim.train_discriminator_start_epoch

        train_discriminator = (
            config.discrim.train_discriminator
            and epoch >= config.discrim.train_discriminator_start_epoch
            and random.random() < float(config.discrim.train_discriminator_prob)
        )

        if train_generator and not train_discriminator:
            logits_real, fmap_real = disc(x)
            logits_fake, fmap_fake = disc(x_hat)
        else:
            logits_real, logits_fake, fmap_real, fmap_fake = None, None, None, None

        commit_loss = torch.mean(commit_loss)
        codebook_loss = torch.mean(codebook_loss)
        freq_loss_dict = freq_loss(x, x_hat)
        losses_g = total_loss(
            fmap_real,
            logits_fake,
            fmap_fake,
            x,
            x_hat,
        )
        loss = (
            losses_g["l_1"] * config.loss.weight_l1
            + freq_loss_dict["total_loss"] * config.loss.weight_freq
            + losses_g["l_2"] * config.loss.weight_l2
        )
        if epoch >= config.loss.commit_start_epoch:
            loss += commit_loss * config.loss.weight_commit
        if train_generator and not train_discriminator:
            loss += losses_g["l_g"] * config.loss.weight_g + losses_g["l_feat"] * config.loss.weight_feat

        optimizer.zero_grad()
        loss.backward()
        if config.common.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.common.gradient_clipping_value)
        optimizer.step()

        forward_time += time.time() - start_forward_time

        if train_discriminator:
            logits_real, _ = disc(x)
            logits_fake, _ = disc(x_hat.detach())
            loss_disc = disc_loss(logits_real, logits_fake)

            optimizer_disc.zero_grad()
            loss_disc.backward()
            if config.common.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(disc.parameters(), config.common.gradient_clipping_value)
            optimizer_disc.step()

            if epoch % config.common.log_every == 1:
                epoch_loss += loss_disc_tensor.item()
                
                # Reduce discriminator loss
                loss_disc_tensor = torch.tensor(loss_disc.item(), device=device)
                loss_disc_tensor = reduce_mean(loss_disc_tensor, dist.get_world_size())

                # Reduce logits
                logits_real_mean = (logits_real[0].mean() + logits_real[1].mean()) / 2
                logits_fake_mean = (logits_fake[0].mean() + logits_fake[1].mean()) / 2
                logits_real_tensor = reduce_mean(logits_real_mean.detach(), dist.get_world_size())
                logits_fake_tensor = reduce_mean(logits_fake_mean.detach(), dist.get_world_size())

                # Reduce max gradient
                max_disc_gradient = torch.tensor(0.0, device=device)
                for param in disc.parameters():
                    if param.grad is not None:
                        local_max = param.grad.abs().max()
                        max_disc_gradient = torch.max(max_disc_gradient, local_max)
                max_disc_gradient = reduce_mean(max_disc_gradient, dist.get_world_size())

                # Only rank 0 logs
                if rank == 0:
                    step = epoch * len(train_loader) + i
                    metrics.fill_metrics({"Loss Discriminator": loss_disc_tensor.item()}, step)
                    metrics.fill_metrics({"Logits Real": logits_real_tensor.item()}, step)
                    metrics.fill_metrics({"Logits Fake": logits_fake_tensor.item()}, step)
                    metrics.fill_metrics({"Max Discriminator Gradient": max_disc_gradient.item()}, step)

        if epoch % config.common.log_every == 1:
            world_size = dist.get_world_size()
            epoch_loss += loss.item()
            global_codes = gather_codes(codes, world_size)

            metrics_dict = []

            # global losses
            metrics_dict.append(losses_g["l_1"].item())               # 0
            metrics_dict.append(commit_loss.item())                   # 1
            metrics_dict.append(freq_loss_dict["l1_loss"].item())     # 2
            metrics_dict.append(freq_loss_dict["acc"].item())         # 3

            # dataset-specific losses
            ds_ids = list(ds_id)   # keep for mapping back
            for j, d_id in enumerate(ds_ids):
                metrics_dict.append(losses_g["l_t"][j].item())        # 4..N
            
            # generator-specific
            if train_generator and not train_discriminator:
                metrics_dict.append(losses_g["l_g"].item())           # after dataset losses
                metrics_dict.append(losses_g["l_feat"].item())

            # gradient norm
            max_gradient = torch.tensor(0.0, device=device)
            for param in model.parameters():
                if param.grad is not None:
                    local_max = param.grad.abs().max()
                    max_gradient = torch.max(max_gradient, local_max)
            metrics_dict.append(max_gradient.item())                  # last slot

            # --- reduce all at once ---
            metrics_tensor = torch.tensor(metrics_dict, device=device)
            metrics_tensor = reduce_mean(metrics_tensor, world_size)
            
            # --- unpack back ---
            loss_L1      = metrics_tensor[0]
            commit_loss_ = metrics_tensor[1]
            freq_L1      = metrics_tensor[2]
            freq_acc     = metrics_tensor[3]
            
            loss_L1_datasets = []
            offset = 4
            for j, d_id in enumerate(ds_ids):
                loss_L1_datasets.append((d_id, metrics_tensor[offset + j]))
            offset += len(ds_ids)
            
            if train_generator and not train_discriminator:
                loss_g    = metrics_tensor[offset]
                loss_feat = metrics_tensor[offset + 1]
                offset += 2
            
            max_gradient = metrics_tensor[offset]

            # --- only rank 0 logs / plots ---
            if rank == 0:
                step = epoch * len(train_loader) + i
                metrics.fill_metrics({
                    "Loss L1": loss_L1.item(),
                    "Loss commit_loss": commit_loss_.item(),
                    "Loss Frequency L1": freq_L1.item(),
                    "Frequency Accuracy": freq_acc.item(),
                    "Max Gradient": max_gradient.item(),
                }, step)

                for d_id, val in loss_L1_datasets:
                    metrics.fill_metrics({f"Loss L1 {d_id}": val.item()}, step)

                if train_generator and not train_discriminator:
                    metrics.fill_metrics({
                        "Loss Generator": loss_g.item(),
                        "Loss Feature": loss_feat.item(),
                    }, step)

                all_codes.append(global_codes.cpu())
                plot_codebook(all_codes=all_codes, epoch=epoch, config=config, writer=writer)
                if i == 0:
                    plot_reconstruction(x=x, x_hat=x_hat, freq_loss_dict=freq_loss_dict,
                                        log_dir=log_dir, epoch=epoch, config=config)
                
        start_data_time = time.time()

    scheduler.step()
    if config.discrim.train_discriminator and epoch >= config.discrim.train_discriminator_start_epoch:
        disc_scheduler.step()

    if epoch % config.common.log_every == 1:
        epoch_loss_global = reduce_mean(torch.tensor(epoch_loss, device=device), dist.get_world_size())

        if rank == 0:
            print(
                f"Epoch {epoch}: Data loading time: {data_loading/i:.4f}s, To device time: {to_device/i:.4f}s, Forward pass time: {forward_time/i:.4f}s"
            )
            metrics_dict = metrics.compute_and_log_metrics()
            metrics_dict["Learning Rate"] = optimizer.param_groups[0]["lr"]
            loss_per_epoch = epoch_loss_global / len(train_loader)
            print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

            logger(writer, metrics_dict, "train", epoch)
            metrics.clear_metrics()


# Logger for tensorboard
def logger(writer, metrics, phase, epoch_index):
    for key, value in metrics.items():
        if type(value) != float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value) != float and len(value.shape) > 0 and value.shape[0] > 2:
            raise Exception("Need to handle multiclass")
            # bp()
        writer.add_scalar("%s/%s" % (phase, key), value, epoch_index)
    writer.flush()

def plot_reconstruction(x, x_hat, freq_loss_dict, log_dir, epoch, config):
    with torch.no_grad():
        S_x = freq_loss_dict["S_x"][:, :freq_loss_dict["S_x"].size(1)//2, :]
        S_x_hat = freq_loss_dict["S_x_hat"][:, :freq_loss_dict["S_x_hat"].size(1)//2, :]
        min_spec_val = min(S_x.min(), S_x_hat.min())
        max_spec_val = max(S_x.max(), S_x_hat.max())

        fs = config.model.sample_rate
        start_idx = 10000
        five_seconds = fs * 5
        thirty_seconds = fs * 30

        x0 = x[0].detach().cpu().numpy().squeeze()
        xhat0 = x_hat[0].detach().cpu().numpy().squeeze()

        x_time = np.arange(x0.shape[0])

        fig, axs = plt.subplots(2, 3, figsize=(20, 10))

        # Five seconds
        axs[0, 0].plot(x_time[start_idx : start_idx + five_seconds], x0[start_idx : start_idx + five_seconds])
        axs[0, 0].set_title('Original Five seconds')
        axs[0, 0].set_ylim(-6, 6)

        axs[1, 0].plot(x_time[start_idx : start_idx + five_seconds], xhat0[start_idx : start_idx + five_seconds])
        axs[1, 0].set_title('Reconstructed Five seconds')
        axs[1, 0].set_ylim(-6, 6)

        # Thirty seconds
        axs[0, 1].plot(x_time[start_idx : start_idx + thirty_seconds], x0[start_idx : start_idx + thirty_seconds])
        axs[0, 1].set_title('Original Thirty seconds')
        axs[0, 1].set_ylim(-6, 6)

        axs[1, 1].plot(x_time[start_idx : start_idx + thirty_seconds], xhat0[start_idx : start_idx + thirty_seconds])
        axs[1, 1].set_title('Reconstructed Thirty seconds')
        axs[1, 1].set_ylim(-6, 6)

        # Spectrograms
        extent = [0, x0.shape[0], 0, S_x.size(1)]
        axs[0, 2].imshow(S_x[0].detach().cpu().numpy(), cmap='jet', aspect='auto',
                         extent=extent, vmin=min_spec_val, vmax=max_spec_val)
        axs[0, 2].invert_yaxis()
        axs[0, 2].set_title('Original Spectrogram')

        axs[1, 2].imshow(S_x_hat[0].detach().cpu().numpy(), cmap='jet', aspect='auto',
                         extent=extent, vmin=min_spec_val, vmax=max_spec_val)
        axs[1, 2].invert_yaxis()
        axs[1, 2].set_title('Reconstructed Spectrogram')

        fig.tight_layout()
        fig.savefig(f"{log_dir}/{epoch}.png")
        plt.close(fig)

    
def plot_codebook(all_codes, writer, epoch, config):
    all_codes = torch.cat(all_codes, dim=0) # B, num_codebooks, T
    all_codes = torch.permute(all_codes, (1, 0, 2))

    # flatten the last two dimensions
    all_codes = all_codes.reshape(all_codes.shape[0], -1)

    # log the distribution of codes. one distribution for each codebook
    entropies = []
    for i in range(all_codes.shape[0]):
        writer.add_histogram(f'Codes/Codebook {i}', all_codes[i], epoch)
        #calculate entropy
        _, counts = torch.unique(all_codes[i], return_counts=True)
        probabilities = counts.float() / counts.sum()
        entropy = -(probabilities * probabilities.log2()).sum()
        entropies.append(entropy.item())
    #create a graph of entropy
    fig, ax = plt.subplots()
    x_axis = np.arange(0, len(entropies))
    ax.plot(x_axis, entropies)
    ax.set_title("Entropy of Codebooks")
    ax.set_xlabel("Codebook index")
    ax.set_ylabel("Entropy")
    ax.set_ylim(0, math.log2(config.model.bins))
    fig.tight_layout()
    writer.add_figure(f"Entropy/{epoch}", fig)
    plt.close(fig)

