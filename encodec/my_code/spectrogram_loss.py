import torch
import torch.nn as nn
import torch.nn.functional as F

class BreathingSpectrogram(nn.Module):
    def __init__(
        self,
        sampling_rate=10,
        n_fft=64,
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
        self.hop_length = hop_length
        self.win_length = win_length

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

class ReconstructionLoss(nn.Module):
    def __init__(self, sampling_rate=10, n_fft=64, device='cuda'):
        super().__init__()
        self.spectrogram = BreathingSpectrogram(sampling_rate=sampling_rate, n_fft=n_fft, device=device)

    def forward(self, x, x_hat):
        # Compute spectrograms
        S_x = self.spectrogram(x)
        S_x_hat = self.spectrogram(x_hat)

        # Compute L1 loss in the frequency domain
        l1_loss = F.l1_loss(S_x, S_x_hat)

        # Compute L2 loss in the frequency domain
        l2_loss = F.mse_loss(S_x, S_x_hat)

        # Combine losses with equal weighting (adjust if necessary)
        total_loss = l1_loss + l2_loss
        return total_loss

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
