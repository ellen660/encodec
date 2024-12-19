import os
import numpy as np
import json
from tqdm import tqdm

# def filter_files(ds_dir, 
#     filepath = os.path.join(self.ds_dir, f)
#     breathing, fs = np.load(filepath)['data'], np.load(filepath)['fs']
#     if breathing.shape[0] < max_length:
#         return False
#     return True

max_length = 10 * 60 * 60 * 4

root = "/data/netmit/wifall/ADetect/data"
datasets = ["shhs2_new", "shhs1_new", "mros1_new", "mros2_new", "wsc_new", "cfs"]

fns_to_ignore = []

for ds in datasets:
    data_dir = os.path.join(root, ds, "thorax")

    fns = sorted(os.listdir(data_dir))

    for fn in tqdm(fns):
        filepath = os.path.join(data_dir, fn)
        breathing = np.load(filepath)['data']
        if breathing.shape[0] <= max_length:
            fns_to_ignore.append(fn)
            print(f"ignoring {fn}")
            # break

# save filenames to ignore to a .py file
with open("fns_to_ignore.py", "w") as f:
    f.write(f"fns_to_ignore = {json.dumps(fns_to_ignore)}")
