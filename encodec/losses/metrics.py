from dataclasses import dataclass
import torch


@dataclass
class MetricsArgs:
    datasets: list[str]
    device: torch.device


class Metrics:
    def __init__(self, args: MetricsArgs):
        self.args = args
        self.datasets = args.datasets
        self.clear_metrics()

    def clear_metrics(self):
        self.metrics_dict = {
            # "Loss per step": {},
            "Loss Frequency": {},
            "Loss L1": {},
            "Loss L2": {},
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
        for dataset in self.datasets:
            self.metrics_dict[f"Loss L1 {dataset}"] = {}
            self.metrics_dict[f"Loss L2 {dataset}"] = {}
        self.used_keys = {}

    def fill_metrics(self, mapping, epoch):
        for key in mapping.keys():
            assert key in self.metrics_dict
            self.metrics_dict[key][epoch] = mapping[key]
            self.used_keys[key] = True

    def compute_and_log_metrics(self):
        metrics = {}
        for item in self.used_keys:
            metrics[item] = sum(self.metrics_dict[item].values()) / len(self.metrics_dict[item])

        return metrics
