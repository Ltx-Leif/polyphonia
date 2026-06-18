import numpy as np
import torch
from typing import Callable, Dict, Optional, Union
from pathlib import Path

import librosa
import librosa.display
import torchaudio


def apply_asr(
    *,
    model,
    attn_input: torch.Tensor,
    attention_op: Callable,
    Q_cur: torch.Tensor,
    K_cur: torch.Tensor,
    V_cur: torch.Tensor,
    Q_src: torch.Tensor,
    K_src: torch.Tensor,
    mask_tensor: torch.Tensor,
    scale: Optional[Union[float, torch.Tensor]],
) -> Optional[torch.Tensor]:
    """
    Score 层保留源结构查询 (Q_src*K_src^T)
    Value 始终使用当前步 (V_cur)
    """
    if mask_tensor is None:
        return None

    scale = 1.0 if scale is None else scale
    BH, Lq, _ = Q_cur.shape
    mask_tensor = mask_tensor.to(dtype=Q_cur.dtype, device=Q_cur.device)
    B = mask_tensor.shape[0] if mask_tensor.dim() > 0 else 1
    mask_seq = mask_tensor.reshape(B, -1)
    if mask_seq.shape[1] != Lq:
        raise ValueError(
            f"Mask sequence length {mask_seq.shape[1]} 与注意力长度 {Lq} 不匹配")

    num_heads = max(1, BH // max(1, B))
    head_mask = mask_seq.repeat_interleave(num_heads, dim=0)  # (BH, Lq)

    # Score 计算
    baddbmm_cur = torch.empty(
        BH, Lq, K_cur.shape[1], dtype=Q_cur.dtype, device=Q_cur.device)
    S_cur = torch.baddbmm(
        baddbmm_cur, Q_cur, K_cur.transpose(-1, -2), beta=0, alpha=scale)

    baddbmm_src = torch.empty(
        BH, Lq, K_src.shape[1], dtype=Q_src.dtype, device=Q_src.device)
    S_src = torch.baddbmm(
        baddbmm_src, Q_src, K_src.transpose(-1, -2), beta=0, alpha=scale)

    M_scores = head_mask.unsqueeze(-1)  # (BH, Lq, 1)
    S_final = M_scores * S_cur + (1.0 - M_scores) * S_src
    P_final = torch.softmax(S_final, dim=-1)

    modified_output = attention_op(
        model, attn_input, attention_probs=P_final, value=V_cur)[4]
    return modified_output


# Compute indexs of target word in embeddings 获得word在嵌入中的位置
def compute_token_merge_indices(tokenizer, prompt: str, word: str, word_idx: int = None, offset_idx: int = 0):
    merge_idxs = []
    tokens = tokenizer.tokenize(prompt.lower())
    # New tokenizer uses wordpiece markers.
    tokens = [x.replace('</w>', '') for x in tokens]

    if word_idx is None:
        word = word.lower()
        # New tokenizer uses wordpiece markers.
        search_tokens = [x.replace('</w>', '')
                         for x in tokenizer.tokenize(word)]
        start_indices = [x + offset_idx for x in range(
            len(tokens)) if tokens[x:x + len(search_tokens)] == search_tokens]

        for indice in start_indices:
            merge_idxs += [i + indice for i in range(0, len(search_tokens))]

        if not merge_idxs:
            raise ValueError(f'Search word {word} not found in prompt!')
    else:
        merge_idxs.append(word_idx)

    return [x + 1 for x in merge_idxs], word_idx  # Offset by 1.


def seeds_gen():
    import random

    # 生成200个种子
    seeds = list(range(100, 20000, 100))

    # 将种子写入文件
    with open('seeds.lst', 'w') as file:
        for seed in seeds:
            file.write(f"{seed}\n")


def vis_spect(mel_spectrogram, path):
    import matplotlib.pyplot as plt
    import librosa.display
    # 假设 mel_spectrogram 是一个形状为 (1024, 64) 的 numpy 数组
    # 这里使用随机数据来模拟

    # 将梅尔频谱图转换为分贝单位
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 创建一个图形和坐标轴
    plt.figure(figsize=(12, 4))

    # 使用 librosa.display.specshow 绘制梅尔频谱图
    librosa.display.specshow(
        mel_spectrogram_db, sr=22050, x_axis='time', y_axis='mel')

    # 添加颜色条
    plt.colorbar(format='%+2.0f dB')

    # 设置标题和标签
    plt.title('Mel Spectrogram')
    plt.xlabel('Time')
    plt.ylabel('Mel Frequency')

    # 显示图形
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.1)


def visualize_and_save_spectrogram(
    wav_path: Union[str, Path],
    out_path: Union[str, Path],
    stft_params: Dict[str, Union[int, float]],
    title: str = "Mel Spectrogram",
    vis_mode: str = "mel",  # 'mel' | 'stft'
):
    """加载音频并保存频谱图，支持 Mel 与线性 STFT 两种模式。"""

    import matplotlib.pyplot as plt
    import librosa.display
    from audio_utils import TacotronSTFT

    target_sr = int(stft_params.get('sampling_rate', 16000))
    n_fft = int(stft_params.get('n_fft', 1024))
    hop_length = int(stft_params.get('hop_length', 160))
    win_length = int(stft_params.get('win_length', 1024))

    waveform, sr = torchaudio.load(str(wav_path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=target_sr)

    waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4))

    if vis_mode == "mel":
        n_mels = int(stft_params.get('n_mels', 64))
        mel_fmin = float(stft_params.get('mel_fmin', 0.0))
        mel_fmax = float(stft_params.get('mel_fmax', target_sr // 2))
        stft = TacotronSTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mels,
            sampling_rate=target_sr,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
        )
        magnitude, _ = stft.stft_fn.transform(waveform)  # (1, F, T)
        mel_linear = torch.matmul(stft.mel_basis, magnitude)  # (1, n_mels, T)
        mel_linear = mel_linear.squeeze(0)  # (n_mels, T)

        mel_power = (mel_linear.cpu().numpy().astype(np.float32)) ** 2
        mel_db = librosa.power_to_db(np.maximum(mel_power, 1e-10), ref=np.max)

        img = librosa.display.specshow(
            mel_db,
            sr=target_sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=mel_fmin,
            fmax=mel_fmax,
            cmap='magma',
            ax=ax,
        )
        ax.set_ylabel('Mel Frequency (Hz)')
    else:
        stft = TacotronSTFT(
            filter_length=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=1,
            sampling_rate=target_sr,
            mel_fmin=0.0,
            mel_fmax=target_sr // 2,
        )
        magnitude, _ = stft.transform(waveform)
        magnitude = magnitude.squeeze(0).cpu().numpy().astype(np.float32)
        power_spec = np.maximum(magnitude, 1e-8) ** 2
        power_db = librosa.power_to_db(power_spec, ref=np.max)
        img = librosa.display.specshow(
            power_db,
            sr=target_sr,
            hop_length=hop_length,
            x_axis='time',
            y_axis='log',
            cmap='magma',
            ax=ax,
        )
        ax.set_ylabel('Frequency (log scale, Hz)')

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)



def visualize_soft_masks_on_spectrogram(
    mixture_wav_path: str,
    masks: Dict[str, torch.Tensor],
    stft_params: Dict[str, Union[int, float]],
    out_dir,
    color_map: Dict[str, str] = None,
    alpha: float = 0.35,
):

    import matplotlib.pyplot as plt
    from audio_utils import TacotronSTFT

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if color_map is None:
        color_map = {'vocals': 'r', 'bass': 'g', 'drums': 'b', 'other': 'y'}

    # 1) 读取 mixture 并重采样到目标 SR
    wav, sr0 = torchaudio.load(mixture_wav_path)  # (C, T)
    target_sr = int(stft_params.get('sampling_rate', 16000))
    if sr0 != target_sr:
        wav = torchaudio.functional.resample(
            wav, orig_freq=sr0, new_freq=target_sr)
    wav = wav.mean(dim=0, keepdim=True)  # (1, T) - mono

    # 2) 构造与 masks 对齐的 Mel 频谱
    n_fft = int(stft_params.get('n_fft', 1024))
    hop = int(stft_params.get('hop_length', 160))
    win = int(stft_params.get('win_length', 1024))
    n_mels = int(stft_params.get('n_mels', 64))
    mel_fmin = float(stft_params.get('mel_fmin', 0.0))
    mel_fmax = float(stft_params.get('mel_fmax', 8000.0))

    stft = TacotronSTFT(
        filter_length=n_fft,
        hop_length=hop,
        win_length=win,
        n_mel_channels=n_mels,
        sampling_rate=target_sr,
        mel_fmin=mel_fmin,
        mel_fmax=mel_fmax,
    )
    magnitude, _ = stft.stft_fn.transform(wav)  # (1, F_lin, T) 线性幅度
    mel_linear = torch.matmul(stft.mel_basis, magnitude)  # (1, n_mels, T)
    mel_linear = mel_linear.squeeze(0)  # (n_mels, T)

    mel_power = (mel_linear.cpu().numpy().astype(np.float32)) ** 2
    mel_db = librosa.power_to_db(np.maximum(
        mel_power, 1e-10), ref=np.max)  # (n_mels, T)

    # 3) 绘制与保存三路图像
    def _color_to_cmap(c: str) -> str:
        return {'r': 'Reds', 'g': 'Greens', 'b': 'Blues', 'y': 'Oranges', 'm': 'Purples'}.get(c, 'Reds')

    for name in ['vocals', 'bass', 'drums', 'other']:
        if name not in masks:
            continue
        mask_tf = masks[name]  # (T, n_mels)
        if mask_tf.dim() != 2:
            raise ValueError(
                f"Mask {name} must be 2D (T,F); got {mask_tf.shape}")

        mask_np = mask_tf.detach().cpu().numpy().astype(np.float32)  # (T, n_mels)

        plt.figure(figsize=(12, 4))
        ax = plt.gca()

        # 3.1) 绘制背景频谱图 (使用 Hz 坐标)
        # mel_db 形状为 (n_mels, T)，specshow 期望 (F, T)
        librosa.display.specshow(
            mel_db,
            sr=target_sr,
            hop_length=hop,
            x_axis='time',
            y_axis='mel',
            fmin=mel_fmin,
            fmax=mel_fmax,
            cmap='gray_r',
            ax=ax
        )

        # 3.2) 绘制前景掩码 (使用 Hz 坐标)
        # mask_np 形状为 (T, n_mels)，需要转置为 (n_mels, T) 以匹配 specshow
        cmap = _color_to_cmap(color_map.get(name, 'r'))
        librosa.display.specshow(
            mask_np.T,
            sr=target_sr,
            hop_length=hop,
            x_axis='time',
            y_axis='mel',
            fmin=mel_fmin,
            fmax=mel_fmax,
            cmap=cmap,
            alpha=alpha,
            ax=ax
        )
        ax.set_ylabel('Mel Frequency (Hz)')
        ax.set_xlabel('Time (s)')

        plt.title(f"Soft Mask Overlay: {name}")
        plt.tight_layout()
        out_path = out_dir / f"softmask_overlay_{name}.png"
        plt.savefig(str(out_path), dpi=150)
        plt.close()


if __name__ == "__main__":
    seeds_gen()