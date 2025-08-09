import argparse
import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from losses import ReconstructionLoss, total_loss
from ppg.ppg_dataset import PpgDataset
from torch.utils.data import DataLoader
from train import init_model, load_config
from utils import set_random_seed


def init_dataset(config, mode: Literal["train", "val", "test"] = "test"):
    """
    Ppg datasets
    """
    cv = config.dataset.cv
    max_length = config.dataset.max_length

    datasets = {}

    datasets["bwh"] = {
        "ppg": (PpgDataset(dataset="bwh", mode=mode, cv=cv, max_length=max_length)),
        #  "abdominal":(BreathingDataset(dataset = "mgh_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length)),
        #  "rf":(BreathingDataset(dataset = "mgh_new", mode = mode, cv = cv, channels = rf_channels, max_length = max_length))
    }
    datasets["mesa"] = {
        "ppg": (PpgDataset(dataset="mesa", mode=mode, cv=cv, max_length=max_length)),
        # "abdominal":(BreathingDataset(dataset = "shhs2_new", mode = mode, cv = cv, channels = abdominal_channels, max_length = max_length))
    }

    return datasets


def infer(ds_name, item, model, freq_loss, device, save_dir, fs=64):
    os.makedirs(f"{save_dir}/{ds_name}", exist_ok=True)
    num_codebooks = model.quantizer.n_q
    x = item["x"].to(device)
    outputs = {}
    fig, axs = plt.subplots(1 + num_codebooks, 5, figsize=(20, 20))
    thirty_seconds = fs * 30
    five_seconds = fs * 5
    start_idx = 10000

    # plot x and the reconstructed x
    time = np.arange(0, thirty_seconds)
    axs[0, 0].plot(
        time, x[0].cpu().numpy().squeeze()[start_idx : start_idx + thirty_seconds]
    )
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_title("Original Signal 30 second")
    axs[0, 0].set_ylim(-4, 4)
    axs[0, 1].plot(x[0].cpu().numpy().squeeze()[start_idx : start_idx + five_seconds])
    axs[0, 1].set_title("Original Signal 5 second")
    axs[0, 1].set_ylim(-4, 4)

    emb = model.encoder(x)
    output = model.quantizer.intermediate_results(x=emb, n_q=num_codebooks)
    x_hat = model.decoder(output["quantized"])

    # breakpoint()

    # logits_real, fmap_real = disc(x)
    # logits_fake, fmap_fake = disc(x_hat)
    # disc_loss(logits_real, logits_fake)
    freq_loss_dict = freq_loss(x, x_hat)
    S_x = freq_loss_dict["S_x"]
    S_x_hat = freq_loss_dict["S_x_hat"]
    _, num_freq, _ = S_x.size()
    S_x = S_x[:, : num_freq // 2, :]
    S_x_hat = S_x_hat[:, : num_freq // 2, :]

    # use this to set the scale of the spectrogram
    # min_spec_val = min(S_x.min(), S_x_hat.min())
    # max_spec_val = max(S_x.max(), S_x_hat.max())

    # time_start = 0
    # time_end = x.shape[-1]

    # x_time = np.arange(time_start, time_end, 1)

    # # plot x and the reconstructed x
    # fig1, axs1 = plt.subplots(4, 1, figsize=(20, 10), sharex=True)

    # axs1[0].plot(x_time, x[0].cpu().numpy().squeeze())
    # axs1[0].set_title('Original')
    # axs1[0].set_ylim(-6, 6)
    # axs1[1].imshow(S_x.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
    # axs1[1].invert_yaxis()
    # axs1[1].set_title('Original Spectrogram')

    # axs1[2].plot(x_time, x_hat[0].detach().cpu().numpy().squeeze())
    # axs1[2].set_title('Reconstructed')
    # axs1[2].set_ylim(-6, 6)
    # axs1[3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap='jet', aspect='auto', extent=[time_start, time_end, 0, num_freq//2], vmin=min_spec_val, vmax=max_spec_val)
    # axs1[3].invert_yaxis()
    # axs1[3].set_title('Reconstructed Spectrogram')

    # fig1.tight_layout()
    # fig1.savefig(f'{save_dir}/{ds_name}_{item["filename"][0][:10]}.png')
    # plt.close(fig1)

    l1_losses = []
    freq_losses = []

    # for n_q in range(1, num_codebooks+1, num_codebooks//8):
    for n_q in range(1, num_codebooks + 1):
        output = model.quantizer.intermediate_results(x=emb, n_q=n_q)
        out = model.decoder(output["quantized"])
        l1_loss = total_loss(
            fmap_real=None,
            logits_fake=None,
            fmap_fake=None,
            input_wav=x,
            output_wav=out,
            sample_rate=fs,
        )["l_1"]

        freq_loss_dict = freq_loss(x, out)
        print(f"codebook {(num_codebooks)}, l1 loss: {l1_loss}")
        # print(f'out sie: {out.size()}')
        S_x = freq_loss_dict["S_x"]
        S_x_hat = freq_loss_dict["S_x_hat"]
        l1_losses.append(l1_loss.cpu().detach().numpy())
        freq_losses.append(freq_loss_dict["l1_loss"].cpu().detach().numpy())
        _, num_freq, _ = S_x.size()
        S_x = S_x[:, : num_freq // 2, :]
        S_x_hat = S_x_hat[:, : num_freq // 2, :]

        # axs[n_q,0].plot(out[0].detach().cpu().numpy().squeeze())
        # axs[n_q,0].set_title(f'n_q={n_q}')
        # axs[n_q,0].set_ylim(-2, 2)
        # n_q = n_q//(num_codebooks//8)
        if n_q == 1:
            axs[0, 3].imshow(S_x.detach().cpu().numpy()[0], cmap="jet", aspect="auto")
            axs[0, 3].invert_yaxis()
            axs[0, 3].set_title("Original Spectrogram")
            axs[0, 4].imshow(
                S_x.detach()
                .cpu()
                .numpy()[0, :, start_idx // 50 : (start_idx + thirty_seconds) // 50],
                cmap="jet",
                aspect="auto",
            )
            axs[0, 4].invert_yaxis()
            axs[0, 4].set_title("Spectrogram")

        time = np.arange(0, thirty_seconds)
        axs[n_q, 0].plot(
            time,
            out.detach()
            .cpu()
            .numpy()
            .squeeze()[start_idx : start_idx + thirty_seconds],
        )
        # plot original signal with transparent
        axs[n_q, 0].plot(
            time,
            x[0]
            .detach()
            .cpu()
            .numpy()
            .squeeze()[start_idx : start_idx + thirty_seconds],
            alpha=0.3,
        )
        axs[n_q, 0].set_xlabel("Time")
        axs[n_q, 0].set_title(f"Signal n_q={n_q*4}")
        axs[n_q, 0].set_ylim(-6, 6)
        axs[n_q, 1].plot(
            out.detach().cpu().numpy().squeeze()[start_idx : start_idx + five_seconds]
        )
        # plot original signal with transparent
        axs[n_q, 1].plot(
            x[0].detach().cpu().numpy().squeeze()[start_idx : start_idx + five_seconds],
            alpha=0.3,
        )
        axs[n_q, 1].set_title(f"Signal n_q={n_q*4}, , Loss = {l1_loss.item()}")
        axs[n_q, 1].set_ylim(-6, 6)

        axs[n_q, 3].imshow(S_x_hat.detach().cpu().numpy()[0], cmap="jet", aspect="auto")
        axs[n_q, 3].invert_yaxis()
        axs[n_q, 3].set_title(f"Reconstructed Spectrogram, n_q={4*n_q}")
        axs[n_q, 4].imshow(
            S_x_hat.detach()
            .cpu()
            .numpy()[0, :, start_idx // 50 : (start_idx + thirty_seconds) // 50],
            cmap="jet",
            aspect="auto",
        )
        axs[n_q, 4].invert_yaxis()
        axs[n_q, 4].set_title(f"Spectrogram n_q={n_q*4}")

        outputs[n_q] = out.detach().cpu().numpy().squeeze()

        del output, l1_loss, S_x, S_x_hat

    # plot the cumulative signal
    for j in range(1, num_codebooks):
        a = outputs[j]
        b = outputs[j + 1]
        diff = b - a
        time = np.arange(0, thirty_seconds)
        axs[j, 2].plot(time, diff[start_idx : start_idx + thirty_seconds])
        axs[j, 2].set_title(f"Signal n_q={(j+1)} - Signal n_q={j}")
        axs[j, 2].set_ylim(-2, 2)

    # save figure
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{ds_name}/{item["filename"][0]}.png')
    plt.close(fig)

    # plot the l1_losses and freq_losses
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(l1_losses)
    axs[0].set_title("L1 Losses")
    axs[1].plot(freq_losses)
    axs[1].set_title("Frequency Losses")
    fig.tight_layout()
    fig.savefig(f'{save_dir}/{ds_name}/losses_{item["filename"][0]}.png')
    plt.close(fig)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="20250804_010937")
    parser.add_argument(
        "--from_dir",
        type=str,
        default="/data/scratch/ellen660/encodec/encodec/ablations/ppg/mesa_new",
    )
    parser.add_argument("--datasets", nargs="+", type=str, help="List of strings")
    parser.add_argument("--modality", type=str, default="ppg")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()

    # Load the YAML file
    config = load_config(f"{args.from_dir}/{args.model_path}/config.yaml")
    set_random_seed(config.common.seed)
    device = torch.device("cuda:6")

    # dataset initialization
    datasets = init_dataset(config, mode="val")

    # Initialize model and discriminator
    model, disc = init_model(config)
    model = model.to(device)
    # disc = disc.to(device)
    checkpoint_path_model = f"{args.from_dir}/{args.model_path}/model.pth"
    # checkpoint_path_disc = f"{log_dir}/disc.pth"
    print("Loading model and discriminator from checkpoint...")
    checkpoint_model = torch.load(checkpoint_path_model, map_location=device)
    # checkpoint_disc = torch.load(checkpoint_path_disc, map_location=device)
    model.load_state_dict(checkpoint_model["model_state_dict"])
    model.eval()
    # disc.load_state_dict(checkpoint_disc)

    freq_loss = ReconstructionLoss(
        alpha=config.spectrogram_loss.alpha,
        bandwidth=config.spectrogram_loss.bandwidth,
        sampling_rate=config.model.sample_rate,
        n_fft=config.spectrogram_loss.n_fft,
        hop_length=config.spectrogram_loss.hop_length,
        win_length=config.spectrogram_loss.win_length,
        device=device,
    )

    for ds_name, dataset in datasets.items():
        dataloader = DataLoader(
            dataset[args.modality],
            batch_size=1,
            shuffle=False,
            num_workers=config.common.num_workers,
            drop_last=False,
            pin_memory=True,
            persistent_workers=True,
        )
        for i, item in enumerate(dataloader):
            if i >= 5:
                break
            infer(
                ds_name=ds_name,
                item=item,
                model=model,
                freq_loss=freq_loss,
                save_dir=f"{args.from_dir}/{args.model_path}/visualizations",
                fs=config.model.sample_rate,
                device=device,
            )
