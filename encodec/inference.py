import os
import sys
import torch
import torch.nn as nn

from clean_model import EncodecModel
from data.dataset import BreathingDataset
from data.bwh import BwhDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import yaml
# Define train one step function
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.spatial.distance import jensenshannon
# from ppg import BwhPpgDataset
from train import load_config, init_model, load_checkpoint
from baseline_data import init_dataset

import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset, DataLoader
import numpy as np
import os
from tqdm import tqdm

def build_model_from_config(config):
    return EncodecModel._get_model(
        config.model.target_bandwidths, 
        config.model.sample_rate, 
        config.model.channels,
        causal=config.model.causal,
        model_norm=config.model.norm,
        segment=eval(config.model.segment),
        ratios=config.model.ratios,
        bins=config.model.bins,
        dimension=config.model.dimension,
    )


@torch.no_grad()
def process_dataset(rank, test_ds, build_model_fn, model_ckpt_path, save_dir, compression_ratio, done, fs, device_ids, config):
    """
    Process a chunk of the dataset on a specific GPU (rank).
    """
    device = torch.device(f"cuda:{device_ids[rank]}")
    
    # Re-initialize the model and load weights
    model = build_model_fn(config).to(device)
    model.load_state_dict(torch.load(model_ckpt_path, map_location=device)['model_state_dict'])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    
    for item, ds_id in tqdm(test_loader, desc=f"[GPU {device_ids[rank]}] Saving codes...", position=rank):
        x = item["x"].to(device, non_blocking=True)
        filename = item["filename"][0]

        if filename not in done:
            _, codes, _, _ = model(x)

            save_path = os.path.join(save_dir, ds_id[0], filename)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, data=codes.squeeze().cpu().numpy(), fs=fs / compression_ratio)

    print(f"[GPU {device_ids[rank]}] Finished processing.")


# @torch.no_grad()
# def process_dataset(test_ds, model, save_dir, compression_ratio, done, fs):
#     """
#     Process a single dataset on the specified GPU.
#     """
#     test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
#     l1, count = 0, 0
#     model.eval()
#     for item, ds_id in tqdm(test_loader, desc=f"Saving codes..."):
#         x = item["x"].to(device)
#         filename = item["filename"][0]
#         if filename not in done:
#             # print(f'filename: {filename} x shape: {x.shape}')
#             _, codes, _, _, = model(x)
#             # print(f'x.shape: {x.shape}')
#             # l1 += torch.nn.L1Loss(reduction='mean')(x, x_hat).item()
#             # count += 1
#             # breakpoint()

#             # Save the prediction
#             # np.savez(os.path.join(save_dir, "shhs2_new", "thorax", filename), data=x_hat, fs=10)

#             # Save the codes
#             save_path = os.path.join(save_dir, ds_id[0], filename)
#             # os.makedirs(os.path.dirname(save_path), exist_ok=True)
#             np.savez(save_path, data=codes.squeeze().cpu().detach().numpy(), fs=fs/compression_ratio)
#     print(f"Finished processing {ds_name}")
#     return l1 / count if count != 0 else None

def get_code_distribution(channel, ds_name, train_ds, save_dir, model, bins):
    all_codes = []
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=4, drop_last=True)
    l1, count = 0, 0
    for i, item in enumerate(tqdm(train_loader, desc=f"Processing {ds_name} channel {channel}")):
        if i >= 10:
            break
        x = item["x"].to(device)
        filename = item["filename"][0]
        x_hat, codes, _, _, = model(x)
        all_codes.append(codes)
        l1 += torch.nn.L1Loss(reduction='mean')(x, x_hat).item()
        count += 1
    all_codes = torch.cat(all_codes, dim=0)
    all_codes = torch.permute(all_codes, (1, 0, 2)) # num_codebooks, B, T
    all_codes = all_codes.reshape(all_codes.shape[0], -1) # num_codebooks, B*T
    # breakpoint()
    num_codebooks = all_codes.shape[0]
    #ceil of square root
    height = int(np.ceil(np.sqrt(num_codebooks)))
    fig, axs = plt.subplots(height, height, figsize=(15, 15))
    axs = axs.flatten()
    all_codes = all_codes.cpu().numpy()
    for i in range(all_codes.shape[0]):
        axs[i].hist(all_codes[i], bins=bins, range=(0, bins), alpha=0.5, label=f'Codebook {i}', density=True)
        axs[i].set_title(f'Codebook {i}')
        axs[i].set_xlim(-10, bins + 10)
        axs[i].set_ylabel('Frequency')
    #save
    save_path = os.path.join(save_dir, ds_name, channel, f"code_distribution.png")  # Save as PNG
    plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    # print(f"Finished processing {ds_name} channel {channel}")
    print(f"L1 score for {ds_name} channel {channel}: {l1 / count if count != 0 else None}")
    return all_codes

def compare_distributions(train_dictionary, datasets, channels, save_dir, model, bins):
    # Plot distributions for each dataset
    codes_dictionary = { ds_name: {} for ds_name in datasets}
    for ds_name in datasets:
        for channel in channels:
            if ds_name in train_dictionary and channel in train_dictionary[ds_name]:
                train_ds = train_dictionary[ds_name][channel]
                all_codes = get_code_distribution(channel, ds_name, train_ds, save_dir, model, bins)
                num_codebooks = all_codes.shape[0]
                codes_dictionary[ds_name][channel] = all_codes
    
    token_counts = {}
    for i in range(num_codebooks):
        for channel_1 in channels:
            for dataset in datasets:
                histogram = np.histogram(codes_dictionary[dataset][channel_1][i], bins=bins, range=(0, bins))[0]
                # breakpoint()
                if i not in token_counts:
                    token_counts[i] = histogram
                else:
                    token_counts[i] += histogram
    # breakpoint()
    # for i in range(num_codebooks):
    #     histogram = token_counts[i]
    #     #get the least frequency one
    #     least = np.argsort(histogram)
    #     print(f"least: {least}")
    #     print(f"Frequency: {histogram[least]}")

    # across all codebooks
    # histogram = token_counts[0]
    # for i in range(1, num_codebooks):
    #     histogram += token_counts[i]
    # least = np.argsort(histogram)
    # print(f"Least frequent token in all codebooks: {least}")
    # print(f"Frequency: {histogram[least]}")
    # #normalize the histogram
    # histogram = histogram / histogram.sum()
    # print(f"Normalized frequency: {histogram[least]}")

    # dataset_histogram = {}
    # for dataset in datasets:
    #     histogram = np.zeros(bins)
    #     for channel_1 in channels:
    #         for i in range(0, num_codebooks):
    #             histogram += np.histogram(codes_dictionary[dataset][channel_1][i], bins=bins, range=(0, bins))[0]
    #     least = np.argsort(histogram)
    #     print(f"Least frequent token in {dataset}: {least}")
    #     print(f"Frequency: {histogram[least]}") #i.e. are there tokens that are not used in specific dataset?

    # with open(f"{save_dir}/js_distances_new.txt", "w") as f:
    #     for i in range(num_codebooks):
    #         for channel_1 in channels:
    #             for channel_2 in channels:
    #                 for j, dataset1 in enumerate(datasets[:-1]):
    #                     for dataset2 in datasets[j+1:]:
    #                         prob_1 = np.histogram(codes_dictionary[dataset1][channel_1][i], bins=bins, range=(0, bins), density=True)[0]
    #                         prob_2 = np.histogram(codes_dictionary[dataset2][channel_2][i], bins=bins, range=(0, bins), density=True)[0]
    #                         dist = jensenshannon(prob_1, prob_2)
    #                         line = f"JS distance between {dataset1}_{channel_1} and {dataset2}_{channel_2} codebook {i}: {dist}\n"
    #                         print(line.strip())  # print to console
    #                         f.write(line)        # append to file
                            # print(f"JS distance between {dataset1} and {dataset2} for channel {channel} codebook {i}: {dist}")
                            # breakpoint()

                            #     0.0	Perfectly identical distributions
# ~0.0–0.1	Very similar distributions
# ~0.1–0.3	Moderately similar
# ~0.3–0.6	Somewhat different
# ~0.6–1.0	Very different distributions
# 1.0	Completely disjoint (no overlap
#all of them are less than 0.2, so across datasets, the distribution is similar

def get_codebook(model):
    codebook = model.codebooks[0].cpu().detach().numpy()
    
    def euclidean_distance(a, b): #magnitude and direction
        return np.linalg.norm(a - b)

    def cosine_similarity(a, b): #direction only
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    min_euclidean, token1, token2 = np.inf, None, None
    max_cosine, one, two = -np.inf, None, None
    
    for i in range(codebook.shape[0]-1):
        for j in range(i+1, codebook.shape[0]):
            dist = euclidean_distance(codebook[i], codebook[j])
            cos_sim = cosine_similarity(codebook[i], codebook[j])
            if dist < min_euclidean:
                min_euclidean = dist
                token1, token2 = i, j
            if cos_sim > max_cosine:
                max_cosine = cos_sim
                one, two = i, j
    print(f"Minimum Euclidean distance: {min_euclidean} between tokens {token1} and {token2}")
    print(f"Maximum Cosine similarity: {max_cosine} between tokens {one} and {two}")
    min_mean, min_std = np.inf, np.inf
    mean_token, std_token = None, None
    max_std = -np.inf
    for i in range(codebook.shape[0]):
        mean, std = np.mean(codebook[i]), np.std(codebook[i])
        if mean < min_mean:
            min_mean = mean
            mean_token = i
        if std < min_std:
            min_std = std
            std_token = i
        if std > max_std:
            max_std = std
    print(f"Minimum mean: {min_mean} for token {mean_token}")
    print(f"Minimum std: {min_std} for token {std_token}")
    print(f"Maximum std: {max_std}")

def get_code_distribution_ppg(model, model_name, test_datasets, datasets, channels, save_dir, bins, pivot=None):
    for ds_name in datasets:
        if ds_name in test_datasets:
            test_ds = test_datasets[ds_name]
            all_codes = []
            for filename in tqdm(test_ds.file_list):
                codes = np.load(os.path.join(save_dir, ds_name, "thorax", filename))['data'] 
                all_codes.append(codes)
            num_codebooks = all_codes[0].shape[0]
            histogram_bins = bins

            # Prepare to aggregate data for each feature
            feature_counts = np.zeros((num_codebooks, histogram_bins), dtype=int)

            # Aggregate counts for each feature
            for sample in all_codes:
                for codebook_idx in range(num_codebooks):
                    feature_data = sample[codebook_idx] #T
                    assert sample[codebook_idx].min() >= 0, "min 0"
                    assert sample[codebook_idx].max() < bins, f"max {sample[codebook_idx].max()}"
                    counts, _ = np.histogram(feature_data, bins=histogram_bins, range=(0, histogram_bins - 1))
                    feature_counts[codebook_idx] += counts
            
            codebook_embeddings = model.codebooks[0].cpu().detach().numpy() 
            index_to_distance = {i: np.linalg.norm(codebook_embeddings[i]) for i in range(1024)}  
            distances = np.array([index_to_distance[i] for i in range(1024)])
            sorted_indices = np.argsort(distances)
            distances_sorted = distances[sorted_indices]
            print(f'min distance {distances.min()} max distance {distances.max()}')

            plt.figure(figsize=(10, 5))  # Create one figure outside the loop

            for codebook_idx in range(num_codebooks):
                counts_sorted = feature_counts[codebook_idx][sorted_indices]
                total = np.sum(counts_sorted)
                densities = counts_sorted / total  # Normalize to density

                # plt.plot(distances, densities, lw=2, alpha=0.5, label=f'Depth {codebook_idx}')  # Add alpha
                plt.fill_between(distances_sorted, densities, alpha=0.3, label=f'Codebook {codebook_idx}')

            plt.xlabel(f"Embedding Norms")
            plt.ylabel("Density")
            plt.title("Distribution of Counts by Distance (All Codebooks)")
            plt.suptitle(f"{model_name}")   # Subtitle above the whole figure
            plt.legend()
            plt.tight_layout()
            save_path = os.path.join(f"/data/scratch/ellen660/encodec/encodec/visualizations/token_distribution", "ppg")  # Save as PNG
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/{ds_name}.png", dpi=300, bbox_inches="tight")  # High-quality save
            plt.close()  # Close the figure to free memory
            print(f'done plotting token distribution for {ds_name}')

    
    # if pivot is None:
    #     #sort by highest to lowest frequency 
    #     pivot = {}
    #     for codebook_idx in range(num_codebooks):
    #         sorted_indices_desc = np.argsort(feature_counts[codebook_idx])[::-1] #highest to lowest
    #         pivot[codebook_idx] = sorted_indices_desc
    #         assert np.array_equal(feature_counts[codebook_idx][sorted_indices_desc],sorted(feature_counts[codebook_idx], reverse=True))
    #     most_common = {}
    #     for codebook_idx in range(num_codebooks):
    #         # feature_counts[codebook_idx] = feature_counts[codebook_idx][pivot[codebook_idx]]
    #         most_common[codebook_idx] = {i: {"idx": pivot[codebook_idx][i],
    #                                          "count": feature_counts[codebook_idx][pivot[codebook_idx][i]], 
    #                                          "frequency": feature_counts[codebook_idx][pivot[codebook_idx][i]]/feature_counts[codebook_idx].sum()}
    #                                         for i in range(5)} #5 most common
    #     return pivot, most_common
    # else:
    #     for codebook_idx in range(num_codebooks):
    #         feature_counts[codebook_idx] = feature_counts[codebook_idx][pivot[codebook_idx]]
            
    #     # Plot histograms for each feature
    #     fig, axes = plt.subplots(8, 4, figsize=(20, 15))  # 8 rows, 4 columns for 32 features
    #     axes = axes.flatten()

    #     for codebook_idx in range(num_codebooks):
    #         axes[codebook_idx].bar(range(histogram_bins), feature_counts[codebook_idx], color='blue', alpha=0.7)
    #         axes[codebook_idx].set_title(f'Codebook {codebook_idx} Distribution')
    #         axes[codebook_idx].set_xlim(-10, histogram_bins + 10)
    #         axes[codebook_idx].set_xlabel('Index')
    #         axes[codebook_idx].grid(True)
    #         axes[codebook_idx].set_ylabel('Frequency')

    #     # Remove empty subplots if any
    #     for i in range(num_codebooks, len(axes)):
    #         fig.delaxes(axes[i])

    #     plt.tight_layout()
    #     save_path = os.path.join(save_dir, ds_name, f"{ds_name}_token_distribution.png")  # Save as PNG
    #     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
    #     plt.close()  # Close the figure to free memory

    # print(f"Finished processing {ds_name}")

# def plot_most_frequent_signals(ds_name, pivot, model, save_dir, config, device):
#     num_codebooks = int(100 * config.model.target_bandwidths[0])
#     # print(f'num codebooks {num_codebooks}')
#     codes = []
#     for codebook_idx in range(num_codebooks):
#         most_common_code = pivot[codebook_idx][0]
#         codes.append(most_common_code)
#     codes = torch.tensor(codes).unsqueeze(1).unsqueeze(2) #N, B, T
#     codes = codes.to(device)

#     # Plot histograms for each feature
#     fig, axes = plt.subplots(8, 4, figsize=(20, 15))  # 8 rows, 4 columns for 32 features
#     axes = axes.flatten()

#     prev = None
#     for n_q in range(1,num_codebooks+1):
#         print(f'codes shape {codes.shape}')
#         quantized = model.quantizer.decode(codes, n_q=n_q)
#         print(f'quantized shape {quantized.shape}')
#         output = model.decoder(quantized).detach().cpu().numpy().squeeze()
#         print(f'model shape {model.shape}')
#         sys.exit()
#         if prev is not None:
#             diff = output - prev 
#         else:
#             diff = output
#         mean = np.mean(diff)
#         std = np.std(diff)
#         time = np.arange(0, 300)
#         axes[n_q-1].plot(time, diff)
#         axes[n_q-1].set_title(f"Signal n_q={n_q} - Signal n_q={n_q-1}, mean {float(f'{mean:.6f}')}, std {float(f'{std:.6f}')}")
#         axes[n_q-1].set_ylim(-0.5, 0.5)
#         prev = output
    
#     plt.tight_layout()
#     save_path = os.path.join(save_dir, ds_name, f"{ds_name}_most_common_signals.png")  # Save as PNG
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
#     plt.close()  # Close the figure to free memory

#     #plot the most common signal 
#     quantized = model.quantizer.decode(codes, n_q=num_codebooks)
#     output = model.decoder(quantized).detach().cpu().numpy().squeeze()
#     fig, ax = plt.subplots()  # Create a single axes (not multiple)
#     # Plot only the specified axes
#     time = np.arange(0, 300)
#     ax.plot(time, output)
#     ax.set_title(f"{ds_name}_generic_signal")
#     ax.set_ylim(-2, 2)
#     save_path = os.path.join(save_dir, ds_name, f"{ds_name}_generic_signal.png")  # Save as PNG
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")  # High-quality save
#     plt.close()  # Close the figure to free memory

#     print(f"Finished processing {ds_name}")

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_dir", type=str, default="/home/ellen660/encodec/encodec/ablations/baseline")
    parser.add_argument("--save_dir", type=str, default="/nobackup/users/ellen660/encodec_codes/baseline")
    parser.add_argument("--model_dir", type=str, default="resp/20250811_2355")
    parser.add_argument("--datasets",type=str, nargs='+', default=["mesa", "bwh", "mgh2"],help="List of dataset names (e.g., --datasets mesa bwh mgh2)")
    parser.add_argument("--resume", type=bool, default=True)
    parser.add_argument("--do_channel", type=List[str], default=["chest"])
    parser.add_argument("--do_code_generation", type=bool, default=True)
    parser.add_argument("--do_token_distribution", type=bool, default=False)
        
    return parser.parse_args()

if __name__ == "__main__":
    args = set_args()
    log_dir = os.path.join(args.user_dir, args.model_dir)
    datasets = args.datasets
    resume = args.resume
    do_channel = args.do_channel

    # Load the YAML file
    _, config = load_config(filepath=f'{log_dir}/config.yaml', schemapath=None)
    compression_ratio = np.prod(config.model.ratios)
    print(f'compression ratio {compression_ratio} which is {compression_ratio/config.model.sample_rate} seconds')

    
    #Initialize the model
    model, _ = init_model(config=config, train_discriminator=False, save_path=None)
    device = torch.device("cuda:6")
    model = model.to(device)
    epoch = load_checkpoint(model=model, optimizer=None, scheduler=None, path=f"{log_dir}/model.pth", device=device)
    model.eval()    
    save_dir = os.path.join(args.save_dir, args.model_dir, str(epoch))
    # Initialize directories
    if args.do_code_generation:
        os.makedirs(save_dir, exist_ok=True)
        for ds_name in datasets:
            # for channel in do_channel:
            os.makedirs(os.path.join(save_dir, ds_name), exist_ok=True)

    #Code Generation
    if args.do_code_generation:
        inference_dataset = init_dataset(config=config, type="inference", datasets=args.datasets, ddp=False, pin_memory=True)
        done = set()
        if resume:
            done = {
                f
                for ds_name in args.datasets
                for f in os.listdir(os.path.join(save_dir, ds_name))
                if f.endswith(".npz")
            }
            done = frozenset(done)
        
        def launch_multi_gpu_processing(test_ds, build_model_fn, model_ckpt_path, save_dir, compression_ratio, done, fs, config):
            device_ids = list(range(torch.cuda.device_count()))
            num_gpus = len(device_ids)

            chunk_size = len(test_ds) // num_gpus
            subsets = [Subset(test_ds, range(i * chunk_size, (i + 1) * chunk_size)) for i in range(num_gpus - 1)]
            subsets.append(Subset(test_ds, range((num_gpus - 1) * chunk_size, len(test_ds))))  # last chunk
            
            ctx = mp.get_context('spawn')
            processes = []

            for rank in range(num_gpus):
                p = ctx.Process(
                    target=process_dataset,
                    args=(rank, subsets[rank], build_model_fn, model_ckpt_path, save_dir, compression_ratio, done, fs, device_ids, config)
                )
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

            
        launch_multi_gpu_processing(
            test_ds=inference_dataset,
            build_model_fn=build_model_from_config,
            model_ckpt_path=f"{log_dir}/model.pth",
            save_dir=save_dir,
            compression_ratio=compression_ratio,
            done=done,
            fs=config.model.sample_rate,
            config=config
        )
        # train_l1 = process_dataset(test_ds=inference_dataset, model=model, save_dir=save_dir, compression_ratio=compression_ratio, done=done, fs=config.model.sample_rate)
        # print(f'train_l1 for {args.datasets}: {train_l1}')

    #Token Distribution
    if args.do_token_distribution:
        get_codebook(model)
        test_datasets = init_dataset(config, mode="test")
        # breakpoint()
        num_codebooks = 6
        histogram_bins = 1024
        get_code_distribution_ppg(model, args.model_dir, test_datasets, datasets, do_channel, save_dir, bins=histogram_bins)
        # compare_distributions(test_datasets, datasets, do_channel, save_dir, model, histogram_bins)

    # Prepare to aggregate data for each feature
    # feature_counts = np.zeros((num_codebooks, histogram_bins), dtype=int)

    # for ds_name in datasets: #on internal datasets only
    #     test_ds = test_datasets[ds_name]
    #     all_codes = []
    #     for filename in tqdm(test_ds.file_list):
    #         codes = np.load(os.path.join(save_dir, ds_name, "codes", filename))['data'] #32 by ?
    #         all_codes.append(codes)

    #     # Aggregate counts for each feature
    #     for sample in all_codes:
    #         for codebook_idx in range(num_codebooks):
    #             feature_data = sample[codebook_idx] #T
    #             assert sample[codebook_idx].min() >= 0, "min 0"
    #             assert sample[codebook_idx].max() < 512, f"max {sample[codebook_idx].max()}"
    #             counts, _ = np.histogram(feature_data, bins=histogram_bins, range=(0, histogram_bins - 1))
    #             feature_counts[codebook_idx] += counts
    # general_pivot = {}
    # for codebook_idx in range(num_codebooks):
    #     sorted_indices_desc = np.argsort(feature_counts[codebook_idx])[::-1] #highest to lowest
    #     general_pivot[codebook_idx] = sorted_indices_desc
    
    # # pivot, _ = get_code_distribution("shhs1", test_datasets["shhs1"], save_dir, config.model.bins)
    # for i, (ds_name, test_ds) in enumerate(test_datasets.items()):
    #     try:
    #         get_code_distribution(ds_name, test_ds, save_dir, config.model.bins, pivot=general_pivot)
    #     except: 
    #         print(f'failed for {ds_name}')

    #Plot most frequent bwh signals (augment of the dataset?)
    # test_datasets = init_dataset(config, mode="test")
    # dataframe = {}
    # for channel in do_channel:
    #     for ds_name in datasets:
    #         try:
    #             test_ds = test_datasets[ds_name][channel]
    #         except:
    #             print(f'channel {channel} not ofund in dataset {ds_name}')
    #             break
    #         pivot, most_common = get_code_distribution(channel, ds_name, test_ds, save_dir, config.model.bins)
    #         dataframe[ds_name] = most_common
    #         plot_most_frequent_signals(ds_name, pivot, model, save_dir, config, device)

    # import pandas as pd

    # # Create a list to hold rows of data
    # rows = []

    # # Process each dataset
    # output_file = "datasets_summary_140935.xlsx"
    # with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    #     # Iterate through each codebook index to create a separate sheet
    #     for codebook_idx in range(32):
    #         rows = []
    #         for ds_name, dataset in dataframe.items():
    #             row = [ds_name]  # Start with the dataset name
    #             for i in range(5):  # Iterate through the top 5 entries
    #                 entry = dataset[codebook_idx][i]  # Get the details for the current codebook index
    #                 row.extend([entry["idx"], entry["count"], entry["frequency"]])
    #             rows.append(row)

    #         # Define columns for the DataFrame
    #         columns = ["Dataset Name"]
    #         for i in range(5):
    #             columns.extend([f"i={i}_idx", f"i={i}_count", f"i={i}_frequency"])

    #         # Create a DataFrame for the current codebook index
    #         df = pd.DataFrame(rows, columns=columns)

    #         # Write this DataFrame to a new sheet in the Excel file
    #         sheet_name = f"Codebook_{codebook_idx}"
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)

    # print(f"Excel file with 32 sheets saved as {output_file}")




    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s"

    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s_disc"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s_disc"

    # log_dir = "/data/netmit/wifall/breathing_tokenizer/encodec_weights/model_30s_new"
    # save_dir = "/data/netmit/wifall/breathing_tokenizer/predictions/model_30s_new"

    # log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250114/175306"
    # save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/175306"

    # log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250115/140935"
    # save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/140935"

    # log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250118/135321"
    # save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/135321"

    # log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250209/142145"
    # save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/142145"
    # encodec\tensorboard\091224_l1\20250209\142145

    # log_dir = "/data/scratch/ellen660/encodec/encodec/tensorboard/091224_l1/20250304/134009"
    # save_dir = "/data/scratch/ellen660/encodec/encodec/predictions/20250304"