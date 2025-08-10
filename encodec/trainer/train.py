from losses import total_loss, disc_loss
import time
from tqdm import tqdm
import random
import torch


def train_one_step(metrics, epoch, optimizer, optimizer_disc, scheduler, disc_scheduler, model, disc, train_loader, config, writer, freq_loss, device, rank):
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

    for i, (item, ds_id) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch", disable=(rank != 0))):
        x = item["x"]
        data_loading +=  time.time() - start_data_time
        
        start_to_device = time.time()
        x = x.to(device, non_blocking=True)
        to_device += time.time() - start_to_device
        
        start_forward_time = time.time()
        x_hat, _, commit_loss, codebook_loss = model(x)

        train_generator = (
            config.discrim.train_discriminator
            and epoch >= config.discrim.train_discriminator_start_epoch
        )

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
        loss = losses_g['l_1'] * config.loss.weight_l1 + freq_loss_dict["total_loss"] * config.loss.weight_freq + losses_g['l_2'] * config.loss.weight_l2
        if epoch >= config.loss.commit_start_epoch:
            loss += commit_loss * config.loss.weight_commit 
        if train_generator and not train_discriminator:
            loss += losses_g['l_g'] * config.loss.weight_g + losses_g['l_feat'] * config.loss.weight_feat

        optimizer.zero_grad()
        loss.backward()
        if config.common.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.common.gradient_clipping_value)
        optimizer.step()
        
        forward_time += time.time() - start_forward_time
        start_data_time = time.time()

        if train_discriminator:
            logits_real, _ = disc(x)
            logits_fake, _ = disc(x_hat.detach())
            loss_disc = disc_loss(logits_real, logits_fake)

            optimizer_disc.zero_grad()
            loss_disc.backward()
            if config.common.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(disc.parameters(), config.common.gradient_clipping_value)
            optimizer_disc.step()

            if epoch % config.common.log_every == 0 and rank == 0:
                metrics.fill_metrics({'Loss Discriminator': loss_disc.item()}, epoch*len(train_loader) + i)
                metrics.fill_metrics({'Logits Real': (torch.mean(logits_real[0]).item() + torch.mean(logits_real[1]).item())/2}, epoch*len(train_loader) + i)
                metrics.fill_metrics({'Logits Fake': (torch.mean(logits_fake[0]).item() + torch.mean(logits_fake[1]).item())/2}, epoch*len(train_loader) + i)
                epoch_loss += loss_disc.item()

                max_disc_gradient = torch.tensor(0.0).to(device)
                for param in disc.parameters():
                    if param.grad is not None:
                        max_disc_gradient = max(max_disc_gradient, param.grad.abs().max().item())
                metrics.fill_metrics({'Max Discriminator Gradient': max_disc_gradient}, epoch*len(train_loader) + i)

        if epoch % config.common.log_every == 0 and rank == 0:
            epoch_loss += loss.item()
            metrics.fill_metrics({
                'Loss L1': losses_g['l_1'].item(),
                'Loss commit_loss': commit_loss.item(),
                'Loss Frequency L1': freq_loss_dict["l1_loss"].item(),
                'Frequency Accuracy': freq_loss_dict["acc"].item(),
            }, epoch*len(train_loader) + i)

            for j, d_id in enumerate(ds_id):
                dataset_id = d_id
                metrics.fill_metrics({f'Loss L1 {dataset_id}': losses_g['l_t'][j].item()}, epoch*len(train_loader) + i)
        
            if train_generator and not train_discriminator:
                metrics.fill_metrics({
                    'Loss Generator': losses_g['l_g'].item(),
                    'Loss Feature': losses_g['l_feat'].item()
                },epoch*len(train_loader) + i)

            max_gradient = torch.tensor(0.0).to(device)
            for param in model.parameters():
                if param.grad is not None:
                    max_gradient = max(max_gradient, param.grad.abs().max().item())

            metrics.fill_metrics({
                'Max Gradient': max_gradient
            }, epoch*len(train_loader) + i)

    if rank == 0:
        print(f"Epoch {epoch}: Data loading time: {data_loading/i:.4f}s, To device time: {to_device/i:.4f}s, Forward pass time: {forward_time/i:.4f}s")
    
    scheduler.step()  
    if config.discrim.train_discriminator and epoch >= config.discrim.train_discriminator_start_epoch:
        disc_scheduler.step()

    if epoch % config.common.log_every == 0 and rank == 0:
        metrics_dict = metrics.compute_and_log_metrics()
        metrics_dict['Learning Rate'] = optimizer.param_groups[0]['lr']
        loss_per_epoch = epoch_loss / len(train_loader)
        print(f"Epoch {epoch}, training loss: {loss_per_epoch}")

        logger(writer, metrics_dict, 'train', epoch)
        metrics.clear_metrics()


#Logger for tensorboard
def logger(writer, metrics, phase, epoch_index):
    for key, value in metrics.items():
        if type(value)!= float and len(value.shape) > 0 and value.shape[0] == 2:
            value = value[1]
        elif type(value)!= float and len(value.shape) > 0 and value.shape[0] > 2:
            raise Exception("Need to handle multiclass")
            # bp()
        writer.add_scalar("%s/%s"%(phase, key), value, epoch_index)
    writer.flush()