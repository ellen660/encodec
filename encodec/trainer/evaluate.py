import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from baseline_data import UniversalWrapper, init_dataset
from torch.utils.data import DataLoader

from encodec.losses import ReconstructionLoss, total_loss
from encodec.trainer.init_config import cleanup, load_config, setup_device
from encodec.trainer.init_model import init_model, load_checkpoint

# Add the B directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[3] / "time_series_foundation_models/dataloaders"))
from universal_loader import Object  # type: ignore

"""
Generate visualizations

poetry run python encodec/trainer/evaluate.py --datasets bwh mgh2 mesa
"""


@torch.no_grad()
def infer(item, model, freq_loss, device, save_dir: str, fs: int):
    os.makedirs(f"{save_dir}", exist_ok=True)
    num_codebooks = model.quantizer.n_q
    x = item["x"].to(device)
    outputs = {}
    fig, axs = plt.subplots(1 + num_codebooks, 5, figsize=(20, 20))
    thirty_seconds = fs * 30
    five_seconds = fs * 5
    start_idx = 10000

    # plot x and the reconstructed x
    time = np.arange(0, thirty_seconds)
    axs[0, 0].plot(time, x[0].cpu().numpy().squeeze()[start_idx : start_idx + thirty_seconds])
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
                S_x.detach().cpu().numpy()[0, :, start_idx // 50 : (start_idx + thirty_seconds) // 50],
                cmap="jet",
                aspect="auto",
            )
            axs[0, 4].invert_yaxis()
            axs[0, 4].set_title("Spectrogram")

        time = np.arange(0, thirty_seconds)
        axs[n_q, 0].plot(
            time,
            out.detach().cpu().numpy().squeeze()[start_idx : start_idx + thirty_seconds],
        )
        # plot original signal with transparent
        axs[n_q, 0].plot(
            time,
            x[0].detach().cpu().numpy().squeeze()[start_idx : start_idx + thirty_seconds],
            alpha=0.3,
        )
        axs[n_q, 0].set_xlabel("Time")
        axs[n_q, 0].set_title(f"Signal n_q={n_q*4}")
        axs[n_q, 0].set_ylim(-6, 6)
        axs[n_q, 1].plot(out.detach().cpu().numpy().squeeze()[start_idx : start_idx + five_seconds])
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
            S_x_hat.detach().cpu().numpy()[0, :, start_idx // 50 : (start_idx + thirty_seconds) // 50],
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
    fig.savefig(f'{save_dir}/{item["filename"][0]}.png')
    plt.close(fig)

    # plot the l1_losses and freq_losses
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].plot(l1_losses)
    axs[0].set_title("L1 Losses")
    axs[1].plot(freq_losses)
    axs[1].set_title("Frequency Losses")
    fig.tight_layout()
    fig.savefig(f'{save_dir}/losses_{item["filename"][0]}.png')
    plt.close(fig)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="ecg/20250812_2349")
    parser.add_argument(
        "--from_dir",
        type=str,
        default="/data/scratch/ellen660/encodec/encodec/ablations/baseline",
    )
    # /data/scratch/ellen660/encodec/encodec/ablations/baseline/ecg/20250812_2349
    parser.add_argument("--datasets", nargs="+", type=str, help="List of strings")
    return parser.parse_args()


if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Load the YAML file
    _, config = load_config(f"{args.from_dir}/{args.model_path}/config.yaml")
    device, local_rank, rank, world_size = setup_device()

    # Load dataset
    compression_ratio = np.prod(config.model.ratios)
    dataset_args = Object(
        dataset=args.datasets,
        mode=config.dataset.mode,
        label="mit_gender",
        seq_len=config.model.sample_rate * config.dataset.max_length,
        fold=config.dataset.cv,
        z_score=True,
        exclude_dataset=None,
        debug=False,
    )
    # create train/val datasets
    dataset, train_loader, sampler = init_dataset(
        config=config,
        type="training",
        datasets=config.dataset.datasets,
        ddp=True,
        pin_memory=True,
        debug_training=args.debug,
    )
    # dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=config.common.num_workers,
    #     drop_last=False,
    #     pin_memory=True,
    #     persistent_workers=True,
    # )
    for i, (item, ds_id) in enumerate(train_loader):
        print(i)
    breakpoint()

    # Initialize model
    model, disc = init_model(config, train_discriminator=False, save_path=None)
    model = model.to(device)
    load_checkpoint(path=f"{args.from_dir}/{args.model_path}/model.pth", model=model, local_rank=local_rank)
    model.eval()

    freq_loss = ReconstructionLoss(
        alpha=config.spectrogram_loss.alpha,
        bandwidth=config.spectrogram_loss.bandwidth,
        sampling_rate=config.model.sample_rate,
        n_fft=config.spectrogram_loss.n_fft,
        hop_length=config.spectrogram_loss.hop_length,
        win_length=config.spectrogram_loss.win_length,
        device=device,
    )

    # run evaluation
    for i, (item, ds_id) in enumerate(dataloader):
        if i >= 5:
            break
        infer(
            item=item,
            model=model,
            freq_loss=freq_loss,
            save_dir=f"{args.from_dir}/{args.model_path}/visualizations",
            fs=config.model.sample_rate,
            device=device,
        )

    cleanup()
