import torchmetrics
import torchmetrics.classification
import torch
# from ipdb import set_trace as bp
import numpy as np 
from dataclasses import dataclass

@dataclass
class MetricsArgs():
    num_datasets: int
    device: str

class Metrics():
    def __init__(self, args: MetricsArgs):
        self.args = args 
        self.used_keys = {}
        self.num_datasets = args.num_datasets # 1 dataset for now
        self.init_metrics()

    def init_metrics(self):
        self.metrics_dict = {
            # "Loss per step": {},
            "Loss Frequency": {},
            "Loss L1": {},
            "Loss L2": {},
            "Loss L1 0": {},
            "Loss L2 0": {},
            "Loss L1 1": {},
            "Loss L2 1": {},
            "Loss L1 2": {},
            "Loss L2 2": {},
            "Loss L1 3": {},
            "Loss L2 3": {},
            "Loss L1 4": {},
            "Loss L2 4": {},
            "Loss L1 5": {},
            "Loss L2 5": {},
            "Loss L1 6": {},
            "Loss L2 6": {},
            "Loss L1 7": {},
            "Loss L2 7": {},
            "Loss commit_loss": {},
            "Loss Frequency L1": {},
            "Loss Frequency L2": {},
            "Frequency Accuracy": {},
            "Loss Discriminator": {},
            "Max Discriminator Gradient": {},
            "Loss Generator": {},
            "Loss Feature": {},
            "Max Gradient": {},
            "Learning Rate": {},
            "Loss": {},
            "Logits Real": {},
            "Logits Fake": {},
        }
        self.metrics = set(self.metrics_dict.keys())
    
    def fill_metrics(self, mapping, epoch):
        for key in mapping.keys():
            assert key in self.metrics
            self.metrics_dict[key][epoch] = mapping[key]
            self.used_keys[key] = True
        
    def compute_and_log_metrics(self):
        metrics = {}
        for item in self.used_keys:
            metrics[item] = sum(self.metrics_dict[item].values()) / len(self.metrics_dict[item])

        return metrics
    
    def clear_metrics(self):
        self.metrics_dict = {
            # "Loss per step": {},
            "Loss Frequency": {},
            "Loss L1": {},
            "Loss L2": {},
            "Loss L1 0": {},
            "Loss L2 0": {},
            "Loss L1 1": {},
            "Loss L2 1": {},
            "Loss L1 2": {},
            "Loss L2 2": {},
            "Loss L1 3": {},
            "Loss L2 3": {},
            "Loss L1 4": {},
            "Loss L2 4": {},
            "Loss L1 5": {},
            "Loss L2 5": {},
            "Loss L1 6": {},
            "Loss L2 6": {},
            "Loss L1 7": {},
            "Loss L2 7": {},
            "Loss commit_loss": {},
            "Loss Frequency L1": {},
            "Loss Frequency L2": {},
            "Frequency Accuracy": {},
            "Loss Discriminator": {},
            "Max Discriminator Gradient": {},
            "Loss Generator": {},
            "Loss Feature": {},
            "Max Gradient": {},
            "Learning Rate": {},
            "Loss": {},
            "Logits Real": {},
            "Logits Fake": {},
        }
        self.used_keys = {}

