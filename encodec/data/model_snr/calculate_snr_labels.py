import pickle
from typing import cast
import numpy as np
from encodec.data import BwhDataset
from encodec.data.all_datasets import MergedDataset
from torch.utils.data import DataLoader 
import os
from tqdm import tqdm
import sys
import torch

class DeepSNRPredictor:
    BATCH_SIZE = 256
    SPEC_WIDTH = 80
    SPEC_HEIGHT = 80

    SPEC_STEP_SEC = 5
    SPEC_WIN_SEC = 60
    SPEC_NPAD = 2
    SPEC_CUTOFF_BPM = 40

    def __init__(self, device="cpu"):
        from sklearn.calibration import CalibratedClassifierCV

        from noise_model import NoiseDetectionModel, NoiseDetectionModelWithSpec

        self.device = device

        self.model_file_path = f"/data/scratch/ellen660/encodec/encodec/data/model_snr/deepsnr_model.pkl"
        self.model_dict = pickle.load(open(self.model_file_path, "rb"))

        self.model = NoiseDetectionModel().to(device)
        self.model.load_state_dict(self.model_dict["model"])
        self.svm_model = cast(CalibratedClassifierCV, self.model_dict["svm_model"])

        self.model_with_spec = NoiseDetectionModelWithSpec().to(device)
        self.model_with_spec.load_state_dict(self.model_dict["model_with_spec"])
        self.svm_model_with_spec = cast(CalibratedClassifierCV, self.model_dict["svm_model_with_spec"])

        self.model.eval()
        self.model_with_spec.eval()

    def predict_batch(self, signals, specs=None, get_feature=False):
        import torch

        signals = torch.tensor(np.array(signals, dtype=np.float32)[:, None], device=self.device)

        if specs is not None:
            specs = torch.tensor(np.array(specs, dtype=np.float32)[:, None], device=self.device)
            features = []
            for i in range(len(signals) // self.BATCH_SIZE + 1):
                batch = signals[i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE]
                spec = specs[i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE]
                if len(batch) > 0:
                    _, x = self.model_with_spec.forward(batch, spec)
                    features.append(x.squeeze().data.cpu().numpy())
            features = np.vstack(features)
            if get_feature:
                return features
            prediction = self.svm_model_with_spec.predict_proba(features)
        else:
            features = []
            for i in range(len(signals) // self.BATCH_SIZE + 1):
                batch = signals[i * self.BATCH_SIZE : (i + 1) * self.BATCH_SIZE]
                if len(batch) > 0:
                    _, x = self.model.forward(batch)
                    features.append(x.squeeze().data.cpu().numpy())
            features = np.vstack(features)
            if get_feature:
                return features
            prediction = self.svm_model.predict_proba(features)

        return prediction[:, 1]

    def predict(self, signals, specs=None, get_feature=False):
        #signal shape: (n, t)
        if signals.shape[1] >= self.model.SignalDuration:
            mid = signals.shape[1] // 2
            half = self.model.SignalDuration // 2
            signals = signals[..., mid - half : mid + half]
        else:
            pad = self.model.SignalDuration - signals.shape[1]
            signals = np.pad(signals, [[0, 0], [pad // 2, pad - pad // 2]], "reflect")
        signals = signals - np.mean(signals, axis=1, keepdims=True)
        signals = signals / np.std(signals, axis=1, keepdims=True)
        signals = np.clip(signals, -self.model.SignalClipLimit, self.model.SignalClipLimit)
        return self.predict_batch(signals, specs, get_feature)

def as_sliding_window(array, window_size, stride):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], stride)]

if __name__ == '__main__':
    root = "/data/netmit/sleep_lab/ali_2/bwh_encodec"
    save_dir = "/data/netmit/sleep_lab/ali_2/bwh_v10_deepsnr_labels"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    done = [f for f in os.listdir(save_dir) if f.endswith(".npy")]
    done = [f.replace(".npy", ".npz") for f in done]
    # breakpoint()
    data = BwhDataset(dataset = "bwh_new", mode = "test", cv = 0, channels = {"thorax": 1.0}, max_length = 10*60*60*4)
    dataset = DataLoader(data, batch_size=1, shuffle=False, num_workers=10)
    print(f'size dataset: {len(dataset)}')
    # test = np.random.rand(10*60*60*4)
    STEP_SIZE = 8 #so one noise label every 80 seconds

    deepsnr_predictor = DeepSNRPredictor() 
    seg_len = deepsnr_predictor.model.SignalDuration
    for i, item in enumerate(tqdm(dataset)):
        breathing = item["x"].squeeze(0).squeeze(0).numpy() # T 
        # print(f'breathing shape: {breathing.shape}')
        #take every two 
        breathing = breathing[::2]
        # print(f'breathing shape: {breathing.shape}')
        #make tensor
        # breathing = torch.tensor(breathing, dtype=torch.float32).unsqueeze(0) # (1, T)
        breathing = breathing.reshape(-1, 600)
        deepsnr = deepsnr_predictor.predict(breathing)
        # print(f'deepsnr shape: {deepsnr.shape}')
        # print(f'deepsnr: {deepsnr}')
        # sys.exit()
        #one label every 2 minutes 

        filename = item["filename"][0]
        # print(f'breathing shape: {breathing.shape}, filename: {filename}')
        if filename in done:
            continue

        # signal_pad = np.pad(breathing, [[seg_len // 2, seg_len // 2 - 1]], mode="reflect")
        # signal_reshaped = as_sliding_window(signal_pad, seg_len, STEP_SIZE) #T // 8, 600
        # signal_reshaped = signal_reshaped.unsqueeze(0)
        # print(f'signal shape after reshape: {signal_reshaped.shape}')
        # deepsnr = deepsnr_predictor(signal_reshaped)
        # #save deepsnr to file
        save_path = os.path.join(save_dir, filename.replace(".npz", ".npy"))
        np.save(save_path, deepsnr)
        print(f'save_path: {save_path}')
        # breakpoint()

        # downsamples by 60 -> one label every 60 seconds 
        # 2min is 10 * 60 * 2 = 1200 samples
