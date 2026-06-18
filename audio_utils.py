import torchaudio
import torch.utils.data
import librosa.util as librosa_util
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Union
from scipy.signal import get_window
from librosa.util import pad_center, tiny
from librosa.filters import mel as librosa_mel_fn


def window_sumsquare(
    window,
    n_frames,
    hop_length,
    win_length,
    n_fft,
    dtype=np.float32,
    norm=None,
):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm) ** 2
    win_sq = librosa_util.pad_center(win_sq, n_fft=n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample: min(n, sample + n_fft)
          ] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


def dynamic_range_compression(x, normalize_fun=torch.log, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return normalize_fun(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def mel_spectrogram_transform(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    mel_basis, hann_window = {}, {}
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft,
                             n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax)+'_'+str(y.device)
                  ] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(
        1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def get_mel_from_wav(audio, _stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    log_magnitudes_stft = (
        torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32)
    )
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, log_magnitudes_stft, energy


def _pad_spec(fbank, target_length=1024):
    n_frames = fbank.shape[0]
    p = target_length - n_frames
    # cut and pad
    if p > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, p))
        fbank = m(fbank)
    elif p < 0:
        fbank = fbank[0:target_length, :]

    if fbank.size(-1) % 2 != 0:
        fbank = fbank[..., :-1]

    return fbank


def pad_wav(waveform, segment_length):
    waveform_length = waveform.shape[-1]
    assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    elif waveform_length < segment_length:
        temp_wav = np.zeros((1, segment_length))
        temp_wav[:, :waveform_length] = waveform
    return temp_wav


def normalize_wav(waveform):
    waveform = waveform - np.mean(waveform)
    waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
    return waveform * 0.5


def read_wav_file(filename, segment_length):
    # waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sr, new_freq=16000)
    waveform = waveform.numpy()[0, ...]
    waveform = normalize_wav(waveform)
    waveform = waveform[None, ...]
    waveform = pad_wav(waveform, segment_length)

    waveform = waveform / np.max(np.abs(waveform)+1e-8)
    waveform = 0.5 * waveform

    return waveform


def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
    assert fn_STFT is not None

    # mixup
    waveform = read_wav_file(filename, target_length * 160)  # hop size is 160

    waveform = waveform[0, ...]
    waveform = torch.FloatTensor(waveform)

    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

    fbank = torch.FloatTensor(fbank.T)
    log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform


class STFT(torch.nn.Module):
    """adapted from Prem Seetharaman's https://github.com/pseeth/pytorch-stft"""

    def __init__(self, filter_length, hop_length, win_length, window="hann"):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]),
             np.imag(fourier_basis[:cutoff, :])]
        )

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        if window is not None:
            assert filter_length >= win_length
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, size=filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode="reflect",
        )
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data,
            torch.autograd.Variable(self.forward_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        ).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )

        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            torch.autograd.Variable(self.inverse_basis, requires_grad=False),
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            # remove modulation effects
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.autograd.Variable(
                torch.from_numpy(window_sum), requires_grad=False
            )
            window_sum = window_sum
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]

            # scale by hop ratio
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[:, :, int(
            self.filter_length / 2):]
        inverse_transform = inverse_transform[:,
                                              :, : -int(self.filter_length / 2):]

        return inverse_transform

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        reconstruction = self.inverse(self.magnitude, self.phase)
        return reconstruction


class TacotronSTFT(torch.nn.Module):
    def __init__(
        self,
        filter_length,
        hop_length,
        win_length,
        n_mel_channels,
        sampling_rate,
        mel_fmin,
        mel_fmax,
    ):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)

    def transform(self, input_data):
        """Forward STFT transform to internal STFT implementation.
        Returns (magnitude, phase) consistent with STFT.transform.
        """
        return self.stft_fn.transform(input_data)

    def spectral_normalize(self, magnitudes, normalize_fun):
        output = dynamic_range_compression(magnitudes, normalize_fun)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y, normalize_fun=torch.log):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert torch.min(y.data) >= -1, torch.min(y.data)
        assert torch.max(y.data) <= 1, torch.max(y.data)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output, normalize_fun)
        energy = torch.norm(magnitudes, dim=1)

        log_magnitudes = self.spectral_normalize(magnitudes, normalize_fun)

        return mel_output, log_magnitudes, energy


def generate_soft_masks(
    mixed_audio_path: str,
    stems: Optional[Dict[str, np.ndarray]],
    stft_params: Dict[str, Union[int, float]],
    smooth_sigma: float = 0.0,
    formula: str = 'energy',
) -> Dict[str, torch.Tensor]:
    """
    生成多分轨（vocals/bass/drums/other）的掩码。

    使用物理严谨的能量域Mel聚合公式：
        mask = sqrt( M(|S_k|^2) / M(|X|^2) )

    其中 M 为Mel滤波器组线性算子。

    Args:
        mixed_audio_path: 混合音频路径或包含分轨wav的目录路径。
            当stems为None时，从此目录加载{vocals,bass,drums,other,mixture}.wav
        stems: 分轨波形字典，格式为：
            {'vocals': np.ndarray, 'bass': np.ndarray, 'drums': np.ndarray,
             'other': np.ndarray, 'mixture': np.ndarray}
            波形应为mono，采样率与stft_params['sampling_rate']一致。
        stft_params: STFT/Mel参数字典，包含：
            - sampling_rate: 采样率（默认16000）
            - n_fft: FFT长度（默认1024）
            - hop_length: 跳跃长度（默认160）
            - win_length: 窗长度（默认1024）
            - n_mels: Mel频带数（默认64）
            - mel_fmin/mel_fmax: Mel频率范围

    Returns:
        Dict[str, torch.Tensor]: 分轨软掩码字典，每个掩码形状为(T, n_mels)，
            数值范围[0,1]，float32类型。
    """
    if stems is None:
        import os
        if not isinstance(mixed_audio_path, str) or not os.path.isdir(mixed_audio_path):
            raise ValueError(
                "generate_soft_masks: 当 stems=None 时，mixed_audio_path 必须是包含分轨 wav 的目录路径。"
            )

        expected_files = {
            'vocals': 'vocals.wav',
            'bass': 'bass.wav',
            'drums': 'drums.wav',
            'other': 'other.wav',
        }
        mixture_file = 'mixture.wav'

        # 加载 wav，并重采样至目标采样率
        def _load_wav(path: str, target_sr: int) -> np.ndarray:
            wav, sr0 = torchaudio.load(path)  # (C, T)
            if sr0 != target_sr:
                wav = torchaudio.functional.resample(
                    wav, orig_freq=sr0, new_freq=target_sr)
            # 转 mono：按通道取平均
            wav = wav.mean(dim=0)
            return wav.numpy()

        stems = {}
        for k, fn in expected_files.items():
            fp = os.path.join(mixed_audio_path, fn)
            if not os.path.isfile(fp):
                continue
            stems[k] = _load_wav(
                fp, int(stft_params.get('sampling_rate', 16000)))

        mix_fp = os.path.join(mixed_audio_path, mixture_file)
        if not os.path.isfile(mix_fp):
            raise FileNotFoundError(
                f"generate_soft_masks: 缺少 mixture.wav，无法计算基于 mixture 的掩码。")
        stems['mixture'] = _load_wav(
            mix_fp, int(stft_params.get('sampling_rate', 16000)))
    else:
        if not isinstance(stems, dict) or 'mixture' not in stems or stems.get('mixture') is None:
            raise ValueError(
                "generate_soft_masks: 需要在 stems 中提供 'mixture' 波形以计算掩码。")

    sr = int(stft_params.get('sampling_rate', 16000))
    n_fft = int(stft_params.get('n_fft', 1024))
    hop = int(stft_params.get('hop_length', 160))
    win = int(stft_params.get('win_length', 1024))
    n_mels = int(stft_params.get('n_mels', 64))
    mel_fmin = float(stft_params.get('mel_fmin', 0.0))
    mel_fmax = float(stft_params.get('mel_fmax', 8000.0))

    # 与 VAE 对齐的 Mel STFT
    mel_stft = TacotronSTFT(
        filter_length=n_fft,
        hop_length=hop,
        win_length=win,
        n_mel_channels=n_mels,
        sampling_rate=sr,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    )

    def _wav_to_stft_magnitude(wave_np: np.ndarray) -> torch.Tensor:
        """
        返回 (1, F_lin, T) 的线性STFT幅度谱。
        """
        if wave_np.ndim > 1:
            wave_np = wave_np.reshape(-1)
        wave = torch.from_numpy(wave_np).float().unsqueeze(0)  # (1, T)

        # STFT: 返回 (1, F_lin, T) 幅度谱
        magnitude, _ = mel_stft.stft_fn.transform(wave)
        return magnitude.float()  # (1, F_lin, T)

    def _wav_to_mel_energy(wave_np: np.ndarray) -> torch.Tensor:
        """
        返回 (T, n_mels) 的 Mel 能量谱。
        计算顺序：STFT → 幅度平方（能量）→ Mel聚合 → Mel能量谱
        """
        magnitude = _wav_to_stft_magnitude(wave_np)  # (1, F_lin, T)
        # 先计算能量（幅度平方）
        energy_linear = magnitude.pow(2)  # (1, F_lin, T)
        # 再执行 Mel 聚合：物理严谨的能量域聚合
        mel_energy = torch.matmul(mel_stft.mel_basis, energy_linear)  # (1, n_mels, T)
        mel_energy = mel_energy.squeeze(0).transpose(0, 1).contiguous()  # (T, n_mels)
        return mel_energy.float()

    # 动态选择可用的分轨（跳过缺失或 None）
    available_stems = [
        k for k, v in stems.items() if k != 'mixture' and v is not None]
    if len(available_stems) == 0:
        return {}

    eps = 1e-10 

    if formula == 'energy':
        # 在能量域计算 Mel 聚合
        mel_energy_mix = _wav_to_mel_energy(stems['mixture'])  # (T, n_mels)
        if mel_energy_mix.dim() != 2:
            raise RuntimeError(
                f"_wav_to_mel_energy must return 2D (T, n_mels), got {mel_energy_mix.shape}")

        masks = {}
        for k in available_stems:
            mel_energy_k = _wav_to_mel_energy(stems[k])  # (T, n_mels)

            # mask = sqrt( M(|S_k|^2) / M(|X|^2) )
            ratio = mel_energy_k / (mel_energy_mix + eps)
            mask = ratio.clamp(min=0.0, max=1.0).sqrt()

            # 形状与数值域断言
            if mask.dim() != 2:
                raise ValueError(
                    f"Mask for {k} must be 2D (T,F); got {mask.dim()}D")
            if mask.shape[1] != n_mels:
                raise ValueError(
                    f"Mask for {k} mel bins mismatch: expected {n_mels}, got {mask.shape[1]}")
            if mask.numel() == 0 or mask.shape[0] <= 0:
                raise ValueError(
                    f"Mask for {k} has invalid time length {mask.shape[0]}")
            if torch.any(mask < -1e-6) or torch.any(mask > 1.0 + 1e-6):
                raise ValueError(f"Mask for {k} out of [0,1] range.")

            masks[k] = mask.to(torch.float32)
    else:
        raise ValueError(f"Unsupported formula: {formula}")

    return masks