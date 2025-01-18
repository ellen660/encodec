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
            "Loss L1 mgh_train_encodec": {},
            "Loss L2 mgh_train_encodec": {},
            "Loss L1 shhs2_new": {},
            "Loss L2 shhs2_new": {},
            "Loss L1 shhs1_new": {},
            "Loss L2 shhs1_new": {},
            "Loss L1 mros1_new": {},
            "Loss L2 mros1_new": {},
            "Loss L1 mros2_new": {},
            "Loss L2 mros2_new": {},
            "Loss L1 wsc_new": {},
            "Loss L2 wsc_new": {},
            "Loss L1 cfs": {},
            "Loss L2 cfs": {},
            "Loss L1 bwh_new": {},
            "Loss L2 bwh_new": {},
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
            "Loss L1 mgh_train_encodec": {},
            "Loss L2 mgh_train_encodec": {},
            "Loss L1 shhs2_new": {},
            "Loss L2 shhs2_new": {},
            "Loss L1 shhs1_new": {},
            "Loss L2 shhs1_new": {},
            "Loss L1 mros1_new": {},
            "Loss L2 mros1_new": {},
            "Loss L1 mros2_new": {},
            "Loss L2 mros2_new": {},
            "Loss L1 wsc_new": {},
            "Loss L2 wsc_new": {},
            "Loss L1 cfs": {},
            "Loss L2 cfs": {},
            "Loss L1 bwh_new": {},
            "Loss L2 bwh_new": {},
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

