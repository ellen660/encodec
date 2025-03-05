import os
import numpy as np
import json
from tqdm import tqdm
import time
from scipy.ndimage import zoom
import csv
import matplotlib.pyplot as plt
import pandas as pd
import ast
from multiprocessing import Pool, Manager, cpu_count

# Constants
DATA_DIR = "/data/netmit/sleep_lab/ali_2/thorax"
WINDOW_SIZE = 200 * 5  # 5 seconds
SAVE_DIR = "/data/netmit/sleep_lab/ali_2/thorax_clipped"
fns_to_ignore = ["47df397adcd2c5d5a4e9f9e549f9a85ca664a77d0e6ed7f345c3520f1083b317.npz", "0b51fcaba3911c612d58109d4608a0ab7a981a9dd88a77336d443b08dc0f5d83.npz", "064652b513c067bc4168c95771d3c274cdfbf18985c33cd3fe4641788d920ec7.npz", "08c2b6695dcdcd48a9db1b21cc8aa3e73715f2ffad9ec5fb28f0a560dc397f5c.npz", "101d9753b542f40897a5fbdbb859bfe226f679c1df5384a6ae5e9a410d76c44c.npz", "14b02c6a09571e327118553aef7855cfecbdd451b31e164fdb6bd405d6aaacf0.npz", "18bc346f175917efa2e079d89f68d9e02cd01b1fcea99ded71aea6f28a7f5366.npz", "1dbb33d2ee924cb448eb29925e044b868a88c567a9810b94a621c91de384d31f.npz", "3bfa840693f9852bd7758fc05d2df691d295d107d79921b5dd5933e2856b4056.npz", "3c16e9d36cc8b54bd2258e69491f2949a331d836e10b58ed3fce3d8c59c6aa99.npz", "3db146b2ef8b0c1ee4071592eab122992cb56c83ef025d2ecf097b16c105cbb0.npz", "4775e61e52914defa3d51ff5ae729a2b076af1c6af67420a6470dd15427abbf6.npz", "53ac8492771977b1a794182e299eb62ce47b3925b9b07122c50c6d1d670733e7.npz", "66b6bd39f03ef48c10e653245c27e0eb14cd114ca987b9e55bf957ae3f054933.npz", "6bd917541e9982ad6d39ea1f8b33b5c7c48053e7f2047d6d0abbee0bb4729a9c.npz", "77a5bbc21ae1330ed20980f9f22594aa8975d6fb0061e573dc8a10428055003d.npz", "834a5aaac39ebaa7f84394535ea4d734d38df5a237b2407f5adb10fa732b5cd7.npz", "9a3fa6fc2d3c1c1ecb219283a4558f06d96012a7eb812d7a5f720e257bc63d5d.npz", "be9f9db3c6b335bf0b769277d653eba2eb19f6b6400878e51ba329ed3fb66a70.npz", "c4d2554d7732af747f068059d98341a3d005b4aaa7fdb4397c25402fa2ad2208.npz", "cba4216684aa1d3d5b406ab0dd5c612f17fbdba4ee54141b4a1c3dd32accd060.npz"]

# Function to process a single file
def process_file(file_info):
    f, shared_mapping, shared_ignore_list = file_info
    file_path = os.path.join(DATA_DIR, f)

    try:
        x = np.load(file_path)['data'].squeeze()

        # Create a sliding window view of the array
        strided_view = np.lib.stride_tricks.sliding_window_view(x, WINDOW_SIZE)

        # Compare all values in each sliding window to the first value of the window
        all_same = np.all(strided_view == strided_view[:, 0][:, None], axis=1)
        all_same = [i for i, val in enumerate(all_same) if val]

        if all_same:
            val = strided_view[all_same[0], 0]
            non_val_count = np.count_nonzero(x != val)
            if non_val_count < 200 * 60 * 60 * 4:  # Ignore files with fewer non-val elements
                shared_ignore_list.append(f)
                return

        zero_indices = set(all_same)
        first_zero = 0
        for i in range(0,int(x.shape[0]*0.25)): #first 20
            if i in zero_indices:
                first_zero = i + WINDOW_SIZE
        last_zero = x.shape[0]
        for i in range(x.shape[0], int(x.shape[0]*0.95), -1): #last 5
            if i in zero_indices:
                last_zero = i
        
        patches_of_0 = []
        for i in range(int(x.shape[0]*0.25), int(x.shape[0]*0.95)):
            if i in zero_indices:
                if patches_of_0 and i + WINDOW_SIZE <= patches_of_0[-1][-1] + 1:
                    patches_of_0[-1][-1]=(i + WINDOW_SIZE)
                else:
                    patches_of_0.append([i, i + WINDOW_SIZE])
        
        for patch in patches_of_0:
            #replace x with the patch
            x[patch[0]:patch[1]] = np.random.randn(patch[1]-patch[0])
        x = x[first_zero:last_zero]
        np.savez(os.path.join(SAVE_DIR, f), data=x, fs=200)

        # Save the indices to the shared mapping
        shared_mapping[f] = (first_zero, last_zero, patches_of_0)

    except Exception as e:
        print(f"Error processing file {f}: {e}")
        shared_ignore_list.append(f)

# Main function
def main(file_list, write_csv=False):
    # Get the list of files

    # Shared structures for multiprocessing
    manager = Manager()
    shared_mapping = manager.dict()
    shared_ignore_list = manager.list()

    # Prepare data for parallel processing
    files_to_process = [(f, shared_mapping, shared_ignore_list) for f in file_list]

    # Use multiprocessing pool
    num_processes = 32  # Number of CPU cores
    print(f'cpu count {num_processes}')
    with Pool(num_processes) as pool:
        list(tqdm(pool.imap_unordered(process_file, files_to_process), total=len(file_list), desc="Processing files"))

    # Save results
    if write_csv:
        output_csv = "bwh_start_end_patches.csv"
        
        # Write the dictionary to a CSV file
        with open(output_csv, mode="w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            # Write the header
            writer.writerow(["file", "start", "end", "patches"])
            # Write each key-value pair
            for file, (start, end, patches_of_0) in shared_mapping.items():
                writer.writerow([file, start, end, patches_of_0])

        with open("fns_to_ignore_less_than_4.py", "w") as f:
            f.write(f"fns_to_ignore = {json.dumps(list(shared_ignore_list))}")
    
    print("Processing complete.")
    print(f"Ignored files: {len(shared_ignore_list)}")
    print(f"Mapping entries: {len(shared_mapping)}")

# def process(file_list, start_end_patches, save_dir = "/data/netmit/sleep_lab/ali_2/thorax_clipped"):
#     os.makedirs(save_dir, exist_ok=True)
#     for f in tqdm(file_list):
#         file_path = os.path.join(DATA_DIR, f)
#         x = np.load(file_path)['data'].squeeze()
#         start, end, patches = start_end_patches[start_end_patches['file'] == f].values[0][1:]
#         patches = ast.literal_eval(patches)
#         for patch in patches:
#             #replace x with the patch
#             breakpoint()
#             x[int(patch[0]):int(patch[1])] = np.random.randn(int(patch[1]-int(patch[0])))
#             breakpoint()
#         x = x[start:end]
#         np.savez(os.path.join(save_dir, f), data=x, fs=200)
        # 1131262 to 1183531

if __name__ == "__main__":
    # file_list = [#"090c63ab1168a53c9ecd7404e25e8921cb8fa24fe1165247d74dae053116ccdb.npz", #32001	6473880
    #         # "6ac248b6a2dd08d3acc304d66bc6829d859b21fef7c0a9988265b16ab5b7ec7c.npz", #112102	6132386
    #         # "2fc5ed5fd389a0b3a9c980629d129da3264a90ab95e6dd1e77a8f75bed547125.npz",
    #         # "e4a088cd60f9e4ae29b13e3bb75fb9a8f21be8175fdb3599a1f0f968331186ba.npz",
    #         # "b9eb066b40d2b4d3aae76413572a8b70e28952a846b0fd8ef81b58614b0505a5.npz",
    #         # "ad5ca7ad138aea5b6dd2ed8f81892f391676d4aad7a8961d67ce1673d8eb12a8.npz",
    #         # "aa28437c14c65bbab5e170c751bb3b615899396dd291b6a57ae1d92ae934391c.npz",
    #         # "791eca9a4cea40f09551014da4fbd713226bdff8a9f4c2215a8cd36fb870e4d0.npz",
    #         # "f990790be81d1e75f616f2c522afcc937e1a53d8c4c2b7fd3a375ede951bc596.npz",
    #         "33565a8bea22c1c64f12728c7d747b5b75f2f2b99f3c55e16038231f13f1bfb4.npz"]
            # "c4d2554d7732af747f068059d98341a3d005b4aaa7fdb4397c25402fa2ad2208.npz" ]#not in ignore]
    # file_list = ["0c058d54074b66ddf0ae4345eca900dea2e55366ba5b9564620af684501c8a32.npz",
    #              "2a5d68e1fb9ec0233665d92158555d974b83f945d46130e29186936b86b1e4b2.npz",
    #              "1e8b987374f9610dbac805750d068f057bba6b40eacca87b61a9d3caeb464268.npz"]

    file_list = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.npz')])
    file_list = [f for f in file_list if f not in fns_to_ignore]
    
    main(file_list, write_csv=True)

    # start_end_patches = pd.read_csv(f'/data/scratch/ellen660/encodec/encodec/data/bwh_start_end_patches.csv')
    # process(file_list, start_end_patches)

    # for file in file_list:
    #     start, end, patches = start_end_patches[start_end_patches['file'] == file].values[0][1:]
    #     patches = ast.literal_eval(patches)
    #     print(f'file {file}')
    #     print(f'patches {patches}, {type(patches)}')
    #     x = np.load(os.path.join("/data/netmit/sleep_lab/ali_2", "thorax_clipped", file))['data'].squeeze()
    #     #copy x
    #     copy = x.copy()
    #     length = x.shape[0]
    #     x_time = np.arange(0, 1, 1/length)
    #     fig, axs = plt.subplots(3, 1, figsize=(8, 6))
    #     axs[0].plot(x_time, x)
    #     axs[0].set_xlabel("Time")
    #     axs[0].set_title(f"Original Signal bwh_{file[:6]} len {length}")
    #     #set 100 xticks
    #     axs[0].set_xticks(np.arange(0, 1, 0.1))
    #     axs[0].grid(True)

        # x = x[:(start_idx+600)*20]
        # try:
        #     x = x[start:end+1]
        # except:
        #     x = x[start:end]
        # time_end = x.shape[0]
        # x_time = np.arange(0, time_end/length, 1/length)
        # axs[1].plot(x_time, x)
        # axs[1].set_xlabel("Time")
        # axs[1].set_title(f"Start {start/length} end {end/length}")
        # #set 100 xticks
        # axs[1].set_xticks(np.arange(0, time_end/length, 0.1))
        # axs[1].grid(True)

        # x = copy[int(patches[0][0])-1000:int(patches[0][1])+1000]
        # time_end = x.shape[0]
        # x_time = np.arange(0, time_end/length, 1/length)
        # axs[2].plot(x_time, x)
        # axs[2].set_xlabel("Time")
        # axs[2].set_title(f"Patches {int(patches[0][0])/length} {int(patches[0][1])/length}")
        # #set 100 xticks
        # axs[2].set_xticks(np.arange(0, time_end/length, 0.01))
        # axs[2].grid(True)
        # plt.savefig(f'/data/scratch/ellen660/encodec/encodec/visualizations/new_original_signal_bwh_{file}.png')
        # plt.close()

# np.random.randn
    #TODO
    #start-end 4 hour
    #rerun the preprocessing w/ start and end indices -> visualze it (10 by 10) and dataset distribution 
    #evaluate tokenizer on L1 bwh 
    #