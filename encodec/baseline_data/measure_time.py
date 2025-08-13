import time
from torch.utils.data import DataLoader
from encodec.baseline_data import init_dataset
from encodec.baseline_data.iter_dataloader import init_iter_dataset
from train import load_config
from tqdm import tqdm

config_dict, config = load_config(filepath=f"encodec/params/baseline_eeg.yaml", schemapath=f"encodec/params/schema.json")

# def measure_loading_time(num_workers, pin):
#     _, loader = init_dataset(config=config, type="training", datasets=config.dataset.datasets, pin_memory=pin) #type: ignore
#     start = time.time()
#     for i, batch in enumerate(tqdm(loader, desc=f"{num_workers} workers, pin {pin}")):
#         pass
#     end = time.time()
#     avg_time_per_batch = (end - start) / len(loader)
#     tqdm.write(f"num_workers={num_workers}, pin = {pin}, avg batch loading time: {avg_time_per_batch:.4f}s")
#     start = time.time()
#     for i, batch in enumerate(tqdm(loader, desc=f"{num_workers} workers, pin {pin}")):
#         pass
#     end = time.time()
#     avg_time_per_batch = (end - start) / len(loader)
#     tqdm.write(f"second time: {avg_time_per_batch:.4f}s")
    
def measure_loading_time_iterator(num_workers, pin):
    # Create loader
    _, loader = init_iter_dataset(
        config=config,
        type="training",
        datasets=config.dataset.datasets,
        pin_memory=pin,
        debug_training=True
    )

    def time_one_epoch():
        start = time.time()
        batch_count = 0
        for data, label in tqdm(loader, desc=f"{num_workers} workers, pin {pin}"):
            batch_count += 1
        end = time.time()
        avg_time = (end - start) / batch_count
        tqdm.write(f"num_workers={num_workers}, pin={pin}, avg batch load: {avg_time:.4f}s")

    # First pass
    time_one_epoch()
    # breakpoint()

    # Must recreate loader for second pass
    _, loader = init_iter_dataset(
        config=config,
        type="training",
        datasets=config.dataset.datasets,
        pin_memory=pin,
        debug_training=True
    )

    # Second pass
    time_one_epoch()

    
# Quick benchmark
for pin in [False, True]:
    measure_loading_time_iterator(num_workers=config.common.num_workers, pin=pin) #type: ignore

# Example usage:
for workers in [0, 1, 2, 4, 8, 12, 16]:
    measure_loading_time_iterator(num_workers=workers, pin=True)