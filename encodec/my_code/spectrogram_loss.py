import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BreathingSpectrogram(nn.Module):
    def __init__(
        self,
        sampling_rate=10,
        n_fft=256,
        hop_length=None,
        win_length=None,
        device='cuda'
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        hop_length = sampling_rate * 1
        # win_length = sampling_rate * 1
        win_length = n_fft
        window = torch.hann_window(win_length, device=device).float()
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.win_length = win_length # 256

    def forward(self, signal):
        # Ensure input dimensions are [B, 1, T]
        if signal.dim() != 3 or signal.size(1) != 1:
            raise ValueError("Input signal must have dimensions [B, 1, T]")

        # Turn each channel into a batch
        signal = signal.squeeze(1)  # Remove channel dimension

        # Padding the signal
        p = (self.n_fft - self.hop_length) // 2
        signal = F.pad(signal, (p, p), "reflect")

        # Compute the STFT
        fft = torch.stft(
            signal,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=False,
            return_complex=True
        )

        # Normalize the STFT
        normalization_factor = self.window.sum()
        fft = fft / normalization_factor

        # Compute the power spectrogram (magnitude squared)
        power_spectrogram = torch.abs(fft) ** 2

        # Convert to log scale (log10) to mimic perceptual scaling
        log_spectrogram = torch.log10(torch.clamp(power_spectrogram, min=1e-5))

        # Restore original shape [B, H, T]
        return log_spectrogram

def create_breathing_frequency_weight(frequency_bins, breathing_frequency, bandwidth=1.0, device='cpu'):
    """
    Creates a weight distribution that assigns higher weights around the breathing frequency.
    The weight decays as you move away from the breathing frequency.
    
    frequency_bins: Tensor of frequency bin indices corresponding to the spectrogram.
    breathing_frequency: The dominant breathing frequency.
    bandwidth: Controls the width of the region around the breathing frequency that has higher weight.

    weight has the shape (batch_size, num_freq, num_frames)
    """
    # Gaussian-like decay weight function centered at the breathing frequency
    weight = torch.exp(-((frequency_bins - breathing_frequency) ** 2) / (2 * bandwidth ** 2) + 1e-8)

    # normalize such that max weight for each frame is 1
    weight = weight / torch.max(weight, dim=1, keepdim=True)[0]

    weight = torch.clamp(weight, min=1e-3, max=1.0)

    weight = weight.to(device)
    return weight


class ReconstructionLoss(nn.Module):
    def __init__(self, sampling_rate=10, n_fft=64, device='cuda'):
        super().__init__()
        self.spectrogram = BreathingSpectrogram(sampling_rate=sampling_rate, n_fft=n_fft, device=device)

        self.bin_freq = (1 / n_fft) / 2

        self.device = device

    def forward(self, x, x_hat):
        # Compute spectrograms
        S_x = self.spectrogram(x)
        S_x_hat = self.spectrogram(x_hat)

        # take the argmax of the S_x
        Sx_breathing_rate = torch.argmax(S_x, dim=1)
        Sx_hat_breathing_rate = torch.argmax(S_x_hat, dim=1) # (batch_size, num_frames)

        # compute the accuracy between the two
        acc = torch.sum(Sx_breathing_rate == Sx_hat_breathing_rate) / Sx_breathing_rate.numel()

        # Keep only the first half of the spectrogram (positive frequencies)
        S_x = S_x[:, :int(0.5/self.bin_freq), :]
        S_x_hat = S_x_hat[:, :int(0.5/self.bin_freq), :]

        # TODO: include weights to penalize lower frequencies. this weight is a 2d matrix, with the same shape as S_x, where the highest weight_value corresponds to indices associated with Sx_breathing_rate
        
        # create a matrix of frequency bins of shape (batch_size, num_freq, num_frames), where each row is the frequency bin index
        batch_size, num_freq, num_frames = S_x.size()

        frequency_bins = torch.arange(num_freq, device=self.device).unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, num_frames)
        breathing_frequency = Sx_breathing_rate.unsqueeze(1).repeat(1, num_freq, 1)

        weight = create_breathing_frequency_weight(frequency_bins, breathing_frequency, bandwidth=1.0, device=self.device)

        # Compute L1 loss in the frequency domain
        l1_loss = F.l1_loss(S_x, S_x_hat, reduction='none')
        l1_loss = torch.mean(l1_loss * weight)

        # Compute L2 loss in the frequency domain
        l2_loss = torch.sqrt(F.mse_loss(S_x, S_x_hat, reduction='none') + 1e-8)
        l2_loss = torch.mean(l2_loss * weight)

        # Combine losses with equal weighting (adjust if necessary)
        total_loss = l1_loss + l2_loss
        
        results = {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'acc': acc,
            'Sx_breathing_rate': Sx_breathing_rate,
            'Sx_hat_breathing_rate': Sx_hat_breathing_rate,
            'S_x': S_x,
            'S_x_hat': S_x_hat
        }

        # return total_loss
        return results

# Test case
def test_reconstruction_loss():
    batch_size = 4
    time_steps = 10*60*60*4 # 4 hours of breathing at 10 Hz
    sampling_rate = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create input and reconstructed signals
    x = torch.randn(batch_size, 1, time_steps, device=device, requires_grad=True)
    x_hat = torch.randn(batch_size, 1, time_steps, device=device, requires_grad=True)

    # Initialize loss function
    loss_fn = ReconstructionLoss(sampling_rate=sampling_rate, n_fft=64, device=device)

    # Compute loss
    loss = loss_fn(x, x_hat)
    loss.backward()
    print(f"Reconstruction loss: {loss.item()}")

if __name__ == "__main__":
    # Run the test
    test_reconstruction_loss()
