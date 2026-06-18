import torch
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import copy
import os
import sys
import random
import matplotlib.pyplot as plt
import torchaudio
from diffusers.utils.torch_utils import randn_tensor
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
    VitsModel,
    VitsTokenizer,
)
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput

from utils import *
from audio_utils import wav_to_fbank, TacotronSTFT
import cv2 as cv
from tqdm import tqdm
from my_audioldm2 import load_audioldm2, encode_latent, decode_latent, get_text_embedding, get_attn_layers, attention_op


class Edit():
    """
    Polyphonia 音频编辑器核心类。

    基于AudioLDM2的多音轨音频编辑扩散模型，支持：
    - 软掩码引导的分轨定向编辑
    - 条件化注意力机制（ASR策略）
    - 文本驱动的音频风格转换

    Attributes:
        device: 计算设备
        unet: U-Net噪声预测器
        vae: 变分自编码器
        text_encoder/text_encoder_2: 双文本编码器
        vocoder: 声码器
        projection_model: 投影模型
        language_model: GPT2语言模型
        scheduler: 扩散调度器
    """

    def __init__(self,
                 vae,
                 tokenizer,
                 tokenizer_2,
                 text_encoder,
                 text_encoder_2,
                 vocoder, projection_model,
                 language_model,
                 unet, scheduler,
                 device,
                 edit_params=None,
                 ):

        self.device = device

        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.vocoder = vocoder
        self.projection_model = projection_model
        self.language_model = language_model
        self.scheduler = scheduler

        self.vae_scale_factor = 2 ** (
            len(self.vae.config.block_out_channels) - 1)
        self.num_channels_latents = self.unet.config.in_channels
        self.vocoder_upsample_factor = np.prod(
            self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        self.edit_params_default = {
            'self_layers': {
                'up': [],
                'mid': [], 
                'down': [],
            },
            'cross_layers': {
                'up': [],
                'mid': [],  # 要改哪个直接写数字，up: 1-9, down: 1-6, mid: 1, 例如'up': [1, 3, 4]就是改up的1，3，4层，从1开始
                'down': [],
            },
            'steps': 200,
            'guidance_scale': 3.5,
            't_start': 1000,
            'tgt_block': 2,
            'use_attention_mask': True,
            'audio_length_in_s': self.unet.config.sample_size * self.vae_scale_factor * self.vocoder_upsample_factor,
            'swap_self_q': True,
            'swap_self_k': True,
            'swap_self_v': True,
            'swap_cross_q': True,
            'swap_break_point': 0,
            'cross_related': False,
            'self_related': True
        }
        if edit_params is not None:
            self.edit_params_default.update(edit_params)

        # self.audio_length_in_s = self.unet.config.sample_size * self.vae_scale_factor * self.vocoder_upsample_factor
        self.audio_length_in_s = self.edit_params_default['audio_length_in_s']
        self.height = int(self.audio_length_in_s /
                          self.vocoder_upsample_factor)
        self.original_waveform_length = int(
            self.audio_length_in_s * self.vocoder.config.sampling_rate)

        self.cur_t = None
        self.cur_i = 0
        self.t_start = self.edit_params_default['t_start']
        self.swap_self_q = self.edit_params_default['swap_self_q']
        self.swap_self_k = self.edit_params_default['swap_self_k']
        self.swap_self_v = self.edit_params_default['swap_self_v']
        self.swap_break_point = self.edit_params_default['swap_break_point']

        self.trigger_get_map = False
        self.trigger_swap_map = False
        self.trigger_cross_related = self.edit_params_default['cross_related']
        self.trigger_self_related = self.edit_params_default['self_related']

        self.self_queries = {}
        self.self_keys = {}
        self.self_values = {}
        self.self_features = {}

        # self.cross_features = {}
        self.cross_queries = {}
        self.cross_keys = {}
        self.cross_values = {}

        self.get_c_hooks = []
        self.get_s_hooks = []
        self.swap_c_hooks = []
        self.swap_s_hooks = []

        self.setting()

        self.enable_softmask: bool = False
        self.mask_bias_strength: float = 3.0
        self._skip_clear_state_once: bool = False
        # {'vocals','bass','drums'}: (T,F) float32
        self.soft_masks_base: Dict[str, torch.Tensor] | None = None
        self.group_order: List[str] = ['vocals', 'bass', 'drums', 'other']
        # 需编辑的分轨
        self.edit_stem: List[str] = []

    def setting(self):
        self.unsetting()
        attn = get_attn_layers(self.unet)
        self.attn_keys = list(attn.keys())

        preserve_cache = getattr(self, "_preserve_attn_cache", False)

        for key in attn.keys():
            if not preserve_cache:
                self.self_queries[key] = {}
                self.self_keys[key] = {}
                self.self_values[key] = {}
                # self.cross_features[key] = {}
                self.cross_queries[key] = {}
                self.cross_keys[key] = {}
                self.cross_values[key] = {}

            self.self_queries.setdefault(key, {})
            self.self_keys.setdefault(key, {})
            self.self_values.setdefault(key, {})
            self.cross_queries.setdefault(key, {})
            self.cross_keys.setdefault(key, {})
            self.cross_values.setdefault(key, {})

            for i in range(len(attn[key])):
                if i % 3 == 0:
                    continue
                self.cross_queries[key].setdefault(i, {})
                self.cross_keys[key].setdefault(i, {})
                self.cross_values[key].setdefault(i, {})
                hook = attn[key][i].transformer_blocks[0].attn2.register_forward_hook(
                    self.__get_cross_attention(key, i), with_kwargs=True)
                self.get_c_hooks.append(hook)
            for i in range(len(attn[key])):
                self.self_queries[key].setdefault(i, {1: {}, 2: {}})
                self.self_keys[key].setdefault(i, {1: {}, 2: {}})
                self.self_values[key].setdefault(i, {1: {}, 2: {}})
                hook_1 = attn[key][i].transformer_blocks[0].attn1.register_forward_hook(
                    self.__get_self_attention(key, i, 1))
                self.get_s_hooks.append(hook_1)
                if i % 3 == 0:
                    hook_2 = attn[key][i].transformer_blocks[0].attn2.register_forward_hook(
                        self.__get_self_attention(key, i, 2))
                    self.get_s_hooks.append(hook_2)
            for i in self.edit_params_default['self_layers'][key]:
                index = i-1
                hook_1 = attn[key][index].transformer_blocks[0].attn1.register_forward_hook(
                    self.__swap_self_attention(key, index, 1))
                if index % 3 == 0:
                    hook_2 = attn[key][index].transformer_blocks[0].attn2.register_forward_hook(
                        self.__swap_self_attention(key, index, 2))
                self.swap_s_hooks.append(hook_1)
                if index % 3 == 0:
                    self.swap_s_hooks.append(hook_2)
            for i in self.edit_params_default['cross_layers'][key]:
                index = i-1
                hook = attn[key][index].transformer_blocks[0].attn2.register_forward_hook(
                    self.__swap_cross_attention(key, index), with_kwargs=True)
                self.swap_c_hooks.append(hook)

    def unsetting(self):
        for h in self.get_c_hooks:
            h.remove()
        for h in self.get_s_hooks:
            h.remove()
        for h in self.swap_c_hooks:
            h.remove()
        for h in self.swap_s_hooks:
            h.remove()
        self.get_c_hooks = []
        self.get_s_hooks = []
        self.swap_c_hooks = []
        self.swap_s_hooks = []

    def set_soft_masks(self, masks: Dict[str, torch.Tensor]):
        """
        设置分轨软掩码的基础网格（T_mel, F_mel）。键限定在 {'vocals','bass','drums','other'}；张量为 float32。
        约束：
        - 每个 mask 必须是 2D，维度 (T,F)；F 应与 VAE/Mel 配置的 n_mels 一致。
        - 所有 mask 的 F 必须相同，内部记录为 self.softmask_n_mels，供后续对齐检查。
        - 允许过滤后为空，此时关闭软掩码分支。
        """
        filtered: Dict[str, torch.Tensor] = {}
        n_mels_ref = None
        for k, v in masks.items():
            if k not in ['vocals', 'bass', 'drums', 'other']:
                continue
            if v.dim() != 2:
                raise ValueError(
                    f"Soft mask {k} must be 2D (T,F); got {v.dim()}D")
            if v.shape[0] <= 0 or v.shape[1] <= 0:
                raise ValueError(
                    f"Soft mask {k} has invalid shape {tuple(v.shape)}")
            if n_mels_ref is None:
                n_mels_ref = v.shape[1]
            elif v.shape[1] != n_mels_ref:
                raise ValueError(
                    f"Soft mask {k} mel bins mismatch: expected {n_mels_ref}, got {v.shape[1]}")
            filtered[k] = v.detach().to(torch.float32)

        if len(filtered) == 0:
            # 无有效软掩码时关闭 softmask 分支
            self.soft_masks_base = {}
            self.softmask_n_mels = None
            return

        self.soft_masks_base = filtered
        self.softmask_n_mels = n_mels_ref

    def enable_softmasking(self, enabled: bool = True):
        """开关：是否启用掩码化交叉注意力。"""
        self.enable_softmask = bool(enabled)

    def set_edit_stem(self, stems: Optional[List[str]]):
        """设置需要编辑（使用当前特征覆盖）的分轨列表。"""
        self.edit_stem = list(stems) if stems is not None else []

    def _build_conditional_asr_mask(self, hidden_states: torch.Tensor, target_dtype: torch.dtype, device: torch.device) -> Optional[torch.Tensor]:
        """
        构造与输入形状对齐的条件化 ASR 掩码：
        - 4D latent：返回 (B,1,H,W)，其中 H 视作时间轴，W 视作频率轴；(T,F) 掩码将通过 F.interpolate(mask, size=(H,W)) 线性映射到 latent。
        - 3D tokens：返回 (B,Lq,1)，频率轴被压缩到序列长度维度。
        """
        mask_bank = getattr(self, 'soft_masks_base', None)
        if not isinstance(mask_bank, dict):
            return None

        edit_names = [
            stem for stem in getattr(self, 'edit_stem', [])
            if stem in mask_bank]
        if len(edit_names) == 0:
            return None

        mask_candidates: List[torch.Tensor] = []
        if hidden_states.dim() == 4:
            B, _, H, W = hidden_states.shape
            for name in edit_names:
                base_mask = mask_bank.get(name, None)
                if base_mask is None:
                    continue
                mask = base_mask.to(target_dtype).to(
                    device).unsqueeze(0).unsqueeze(0)
                mask_hw = F.interpolate(mask, size=(H, W), mode='bilinear',
                                        align_corners=False).squeeze(0).squeeze(0)
                mask_candidates.append(
                    mask_hw.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1))
        elif hidden_states.dim() == 3:
            B, Lq_in, _ = hidden_states.shape
            for name in edit_names:
                base_mask = mask_bank.get(name, None)
                if base_mask is None:
                    continue
                mask = base_mask.to(target_dtype).to(
                    device).unsqueeze(0).unsqueeze(0)
                mask_1d = F.interpolate(
                    mask, size=(Lq_in, 1), mode='bilinear', align_corners=False).squeeze().view(Lq_in)
                mask_candidates.append(
                    mask_1d.unsqueeze(0).unsqueeze(-1).repeat(B, 1, 1))
        else:
            return None

        if len(mask_candidates) == 0:
            return None

        return torch.stack(mask_candidates, dim=0).max(dim=0).values

    def get_text_condition_grouped(self,
                                   target_prompt: Dict[str, str],
                                   negative_prompt=None,
                                   max_new_tokens=None,
                                   num_waveforms_per_prompt: int = 1):
        """
        用 baseline_prompt 进行一次完整编码，保持全局语义连贯；
        对每个分轨短语在基准句 token 中做子序列定位，记录 token span；
        仍返回单条基准句的条件给 U-Net。
        """
        device = self.device

        # 0) 解析基准句
        baseline_prompt = target_prompt.get('__baseline__', '').strip()
        if len(baseline_prompt) == 0:
            raise ValueError(
                "get_text_condition_grouped: baseline_prompt 不能为空。")

        # 1) 识别启用的分轨
        enabled_by_prompt: List[str] = [
            g for g in self.group_order if target_prompt.get(g, '').strip() != '']
        mask_bank = self.soft_masks_base or {}
        if len(mask_bank) > 0:
            enabled_names: List[str] = [
                g for g in enabled_by_prompt if g in mask_bank]
            missing_masks = [
                g for g in enabled_by_prompt if g not in mask_bank]
            if len(missing_masks) > 0:
                print(
                    f"[SoftMask][warning] prompt 中声明但缺失掩码的分轨将被跳过: {missing_masks}")
        else:
            # 软掩码为空时允许退化运行，仅按 prompt 启用
            enabled_names = enabled_by_prompt

        # 2) 基准句一次性编码（保持与原 get_text_condition 完全一致的路径与形状）
        self._skip_clear_state_once = True
        base_kwargs = self.get_text_condition(
            prompt=baseline_prompt,
            negative_prompt=negative_prompt,
            max_new_tokens=max_new_tokens,
            num_waveforms_per_prompt=num_waveforms_per_prompt,
        )

        # 3) 使用正式编码路径 + offset_mapping 进行 token 级定位
        primary_tok = self.tokenizer_2 if isinstance(
            self.tokenizer_2, (RobertaTokenizer, RobertaTokenizerFast, T5Tokenizer, T5TokenizerFast)) else self.tokenizer
        token_indices_raw: Dict[str, List[int]] = {}
        text_seq_len = None
        base_tok_len = 1
        try:
            base_inputs = primary_tok(
                baseline_prompt,
                padding="max_length" if isinstance(
                    primary_tok, (RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer)) else True,
                truncation=True,
                max_length=primary_tok.model_max_length if hasattr(
                    primary_tok, 'model_max_length') else None,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
        except Exception as e:
            raise RuntimeError(
                f"get_text_condition_grouped: tokenizer 编码失败，无法建立文本-分轨对齐：{e}")

        offsets = base_inputs.get('offset_mapping', None)
        input_ids = base_inputs.get('input_ids', None)
        attention_mask_tok = base_inputs.get('attention_mask', None)
        if offsets is None or input_ids is None:
            raise ValueError(
                "get_text_condition_grouped: tokenizer 未返回 offset_mapping，无法对齐分轨。")

        offsets_list = offsets[0].tolist()
        attn_mask_list = attention_mask_tok[0].tolist(
        ) if attention_mask_tok is not None else [1] * len(offsets_list)
        text_seq_len = int(input_ids.shape[1])
        base_tok_len = max(1, text_seq_len)

        # 构建归一空白的 baseline 及其映射
        bl_lower = baseline_prompt.lower()
        norm_chars = []
        norm_to_orig = []
        k = 0
        while k < len(bl_lower):
            ch = bl_lower[k]
            if ch.isspace():
                j = k
                while j < len(bl_lower) and bl_lower[j].isspace():
                    j += 1
                norm_chars.append(' ')
                norm_to_orig.append(k)
                k = j
            else:
                norm_chars.append(ch)
                norm_to_orig.append(k)
                k += 1
        bl_norm = ''.join(norm_chars)

        import re
        for g in enabled_names:
            phrase = target_prompt.get(g, '').strip()
            if len(phrase) == 0:
                continue
            pl_norm = re.sub(r"\s+", " ", phrase.lower()).strip()
            char_start_norm = bl_norm.find(pl_norm)
            if char_start_norm < 0:
                # 找不到字面子串（归一空白后），跳过
                continue
            char_end_norm = char_start_norm + len(pl_norm)
            if char_end_norm > len(norm_to_orig):
                continue
            # 映射回原始 baseline 的字符区间
            orig_start = norm_to_orig[char_start_norm]
            orig_end = norm_to_orig[char_end_norm - 1] + 1

            span_raw_indices: List[int] = []
            for i, (s, e) in enumerate(offsets_list):
                if i < len(attn_mask_list) and attn_mask_list[i] == 0:
                    continue
                if e <= s:
                    continue
                if not (e <= orig_start or s >= orig_end):
                    span_raw_indices.append(i)
            if len(span_raw_indices) > 0:
                token_indices_raw[g] = span_raw_indices

        # 记录到实例，供交叉注意力阶段使用
        self.token_group_names_in_order = enabled_names
        self.token_indices_per_group = token_indices_raw
        self.text_seq_len = int(
            text_seq_len) if text_seq_len is not None else base_tok_len

        # 4) 直接返回基准句条件
        return base_kwargs

    def _masked_cross_attention(self,
                                attn_module,
                                hidden_states: torch.Tensor,
                                encoder_hidden_states: torch.Tensor,
                                attention_mask: Optional[torch.Tensor],
                                layer_ctx: Optional[tuple] = None):
        """
        重构版本：基于软加权的交叉注意力引导机制
        """
        # 1) 计算原始注意力分数
        _, Q, K, _, _, scale = attention_op(
            attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)

        # 基础维度信息
        BH, Lq, Dh = Q.shape
        B = hidden_states.shape[0]
        num_heads = BH // B if B > 0 else 1
        Lk = K.shape[1]

        # 2) 处理attention_mask并计算原始分数矩阵
        if attention_mask is None:
            baddbmm_input = torch.empty(
                Q.shape[0], Q.shape[1], K.shape[1], dtype=Q.dtype, device=Q.device
            )
            beta = 0
        else:
            seq_len_k = K.shape[1]
            bsz = B
            prepared_mask = attn_module.prepare_attention_mask(
                attention_mask, seq_len_k, bsz)
            if prepared_mask is None:
                baddbmm_input = torch.empty(
                    Q.shape[0], Q.shape[1], K.shape[1], dtype=Q.dtype, device=Q.device
                )
                beta = 0
            else:
                baddbmm_input = prepared_mask.to(
                    dtype=Q.dtype, device=Q.device)
                # 对齐到 [BH, Lq, Lk]
                if baddbmm_input.shape[0] == B and B > 0:
                    baddbmm_input = baddbmm_input.repeat_interleave(
                        num_heads, dim=0)
                if baddbmm_input.shape[1] == 1 and Lq != 1:
                    baddbmm_input = baddbmm_input.expand(
                        baddbmm_input.shape[0], Lq, K.shape[1]).contiguous()
                beta = 1

        # 计算原始注意力分数 S: (BH, Lq, Lk)
        S = torch.baddbmm(
            baddbmm_input,
            Q,
            K.transpose(-1, -2),
            beta=beta,
            alpha=scale,
        )

        # 3) 检查软掩码与 token 原始索引是否可用，并限制在文本分支
        branch_tag = getattr(attn_module, "_melodia_branch", None)
        token_indices_raw_per_group = getattr(
            self, 'token_indices_per_group', None)
        has_token_indices = isinstance(
            token_indices_raw_per_group, dict) and len(token_indices_raw_per_group) > 0
        mask_bank = getattr(self, 'soft_masks_base', None)
        if (not mask_bank) or (branch_tag == "gpt"):
            P = torch.softmax(S, dim=-1)
            return attention_op(attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_probs=P)[-2]
        if not has_token_indices:
            P = torch.softmax(S, dim=-1)
            return attention_op(attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_probs=P)[-2]

        # 4) 生成分轨软掩码并下采样到当前层分辨率
        enabled_names = getattr(self, 'token_group_names_in_order', [])
        if len(enabled_names) == 0:
            # 无启用分轨，回退到标准注意力
            P = torch.softmax(S, dim=-1)
            return attention_op(attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_probs=P)[-2]

        # 根据hidden_states维度处理软掩码
        soft_masks_processed = {}  # {group_name: tensor(B, Lq)}

        if hidden_states.ndim == 4:
            # 4D情况: (B, C, H, W)
            _, _, H, W = hidden_states.shape
            for g in enabled_names:
                base_mask = mask_bank.get(g, None)
                if base_mask is None:
                    # 无掩码时跳过该分轨
                    continue
                # 下采样到当前层分辨率
                mask_resized = base_mask.to(Q.dtype).to(
                    Q.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T, F)
                mask_hw = F.interpolate(
                    mask_resized, size=(H, W), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)  # (H, W)
                mask_flat = mask_hw.flatten().unsqueeze(0).repeat(B, 1)  # (B, H*W)
                soft_masks_processed[g] = mask_flat

        elif hidden_states.ndim == 3:
            # 3D情况: (B, Lq, C)，即为当前 AudioLDM2 配置
            for g in enabled_names:
                base_mask = mask_bank.get(g, None)
                if base_mask is None:
                    continue
                # 下采样到当前序列长度
                mask_resized = base_mask.to(Q.dtype).to(
                    Q.device).unsqueeze(0).unsqueeze(0)  # (1, 1, T, F)
                mask_1d = F.interpolate(
                    mask_resized, size=(Lq, 1), mode='bilinear', align_corners=False
                ).squeeze().view(Lq)  # (Lq,)
                mask_flat = mask_1d.unsqueeze(0).repeat(B, 1)  # (B, Lq)
                soft_masks_processed[g] = mask_flat
        else:
            raise ValueError(
                f"Unsupported hidden_states ndim: {hidden_states.ndim}")

        # 若全部被过滤，直接回退
        if len(soft_masks_processed) == 0:
            P = torch.softmax(S, dim=-1)
            return attention_op(attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_probs=P)[-2]

        # 5) 基于 query 侧软掩码与 key 侧原始 token 索引的外积
        S_guided = S.clone()  # (BH, Lq, Lk)
        
        bias_val = getattr(self, 'mask_bias_strength', 3.0)
        if isinstance(bias_val, dict):
            # 获取当前时间步
            current_t = int(getattr(self, 'cur_t', 0))
            bias_strength = float(bias_val.get(current_t, 0.0))
        else:
            bias_strength = float(bias_val)
        has_bias = False

        for g_name in enabled_names:
            idx_list = token_indices_raw_per_group.get(
                g_name, None) if has_token_indices else None
            if not idx_list:
                continue

            # 获取该分轨的软掩码 (B, Lq)
            group_mask = soft_masks_processed.get(g_name, None)
            if group_mask is None:
                continue

            # 扩展到多头维度 (BH, Lq)
            group_mask_heads = group_mask.repeat_interleave(num_heads, dim=0)

            # 构造 key 侧 token one-hot 掩码 (Lk,)
            token_mask_k = torch.zeros(
                Lk, dtype=Q.dtype, device=Q.device)
            valid_idx = [i for i in idx_list if (
                i is not None and 0 <= int(i) < Lk)]
            if len(valid_idx) == 0:
                continue
            token_mask_k[torch.tensor(
                valid_idx, device=Q.device, dtype=torch.long)] = 1.0

            # 外积得到 (BH, Lq, Lk) 的加性偏置
            group_bias = group_mask_heads.unsqueeze(-1) * \
                token_mask_k.view(1, 1, -1)
            S_guided = S_guided + bias_strength * group_bias
            has_bias = True

        if not has_bias:
            P = torch.softmax(S, dim=-1)
            return attention_op(attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_probs=P)[-2]

        P_guided = torch.softmax(S_guided, dim=-1)

        return attention_op(attn_module, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_probs=P_guided)[-2]

    def update_t_start(self, edit_params):
        if edit_params is not None:
            self.edit_params_default.update(edit_params)
        self.t_start = self.edit_params_default['t_start']

    def update_swap_break_point(self, edit_params):
        if edit_params is not None:
            self.edit_params_default.update(edit_params)
        self.swap_break_point = self.edit_params_default['swap_break_point']

    def update_audio_length_in_s(self, audio_length_in_s):
        self.audio_length_in_s = audio_length_in_s
        self.height = int(self.audio_length_in_s /
                          self.vocoder_upsample_factor)
        self.original_waveform_length = int(
            self.audio_length_in_s * self.vocoder.config.sampling_rate)

    def __get_self_attention(self, name, layer_idx, attention_idx):
        def hook(model, input, output):

            if self.trigger_get_map and self.trigger_self_related and int(self.cur_t) > self.swap_break_point:

                _, query, key, value, _, _ = attention_op(model, input[0])

                self.self_queries[name][layer_idx][attention_idx][int(
                    self.cur_t)] = query.detach()
                self.self_keys[name][layer_idx][attention_idx][int(
                    self.cur_t)] = key.detach()
                self.self_values[name][layer_idx][attention_idx][int(
                    self.cur_t)] = value.detach().cpu()

        return hook

    def __get_cross_attention(self, name, layer_idx):
        def hook(model, input, kwargs, output):

            if self.trigger_get_map and self.trigger_cross_related and int(self.cur_t) > self.swap_break_point and layer_idx % 3 != 0:

                _, query, key, value, _, _ = attention_op(
                    model, input[0], **kwargs)

                # self.cross_features[name][layer_idx][int(self.cur_t)] = attention_probs.detach()
                self.cross_queries[name][layer_idx][int(
                    self.cur_t)] = query.detach().cpu()
                self.cross_keys[name][layer_idx][int(
                    self.cur_t)] = key.detach().cpu()
                self.cross_values[name][layer_idx][int(
                    self.cur_t)] = value.detach().cpu()

        return hook

    def __swap_cross_attention(self, name, layer_idx):
        def hook(model, input, kwargs, output):

            if self.trigger_swap_map and self.trigger_cross_related and layer_idx % 3 != 0 and int(self.cur_t) > self.swap_break_point:
                branch_tag = getattr(model, "_melodia_branch", None)
                token_indices_raw = getattr(
                    self, 'token_indices_per_group', None)
                has_raw_indices = isinstance(
                    token_indices_raw, dict) and any(len(v) > 0 for v in token_indices_raw.values())
                is_text_branch = branch_tag == "text"
                is_gpt_branch = branch_tag == "gpt"
                # 若启用软掩码且有对齐的原始 token 索引，并且当前模块属于文本分支
                if self.enable_softmask and (self.soft_masks_base is not None) and has_raw_indices and is_text_branch:
                    # 文本分支
                    modified_output = self._masked_cross_attention(
                        model,
                        input[0],
                        kwargs.get('encoder_hidden_states', None),
                        kwargs.get('attention_mask', None),
                        layer_ctx=(name, layer_idx)
                    )
                    return modified_output

                # GPT 分支
                cond_asr_enabled = (
                    getattr(self, 'enable_softmask', False)
                    and getattr(self, 'soft_masks_base', None) is not None
                    and isinstance(getattr(self, 'edit_stem', []), list)
                    and len(getattr(self, 'edit_stem', [])) > 0
                )
                if cond_asr_enabled and is_gpt_branch:
                    cur_t = int(self.cur_t)
                    device = input[0].device
                    cache_q = self.cross_queries.get(name, {}).get(
                        layer_idx, {})
                    cache_k = self.cross_keys.get(name, {}).get(
                        layer_idx, {})
                    cache_v = self.cross_values.get(name, {}).get(
                        layer_idx, {})
                    if cur_t in cache_q and cur_t in cache_k and cur_t in cache_v:
                        Q_src = cache_q[cur_t].to(device)
                        K_src = cache_k[cur_t].to(device)
                        V_src = cache_v[cur_t].to(device)

                        hidden_states = input[0]
                        _, Q_cur, K_cur, V_cur, _, scale = attention_op(
                            model,
                            hidden_states,
                            encoder_hidden_states=kwargs.get(
                                'encoder_hidden_states', None),
                            attention_mask=kwargs.get(
                                'attention_mask', None),
                        )

                        mask_tensor = self._build_conditional_asr_mask(
                            hidden_states, Q_cur.dtype, device)
                        if mask_tensor is not None:
                            modified_output = apply_asr(
                                model=model,
                                attn_input=hidden_states,
                                attention_op=attention_op,
                                Q_cur=Q_cur,
                                K_cur=K_cur,
                                V_cur=V_cur,
                                Q_src=Q_src,
                                K_src=K_src,
                                mask_tensor=mask_tensor,
                                scale=scale,
                            )                          
                            if modified_output is not None:
                                return modified_output

                if name in self.cross_queries.keys():
                    if layer_idx in self.cross_queries[name].keys():
                        if int(self.cur_t) in self.cross_queries[name][layer_idx].keys():
                            queries = self.cross_queries[name][layer_idx][int(
                                self.cur_t)]
                            keys = self.cross_keys[name][layer_idx][int(
                                self.cur_t)]
                            _, _, _, _, modified_output, _ = attention_op(
                                model, input[0], query=queries, key=keys, **kwargs)
                            return modified_output

        return hook

    def __swap_self_attention(self, name, layer_idx, attention_idx):
        def hook(model, input, output):
            cur_t = int(self.cur_t)
            if not (self.trigger_swap_map and self.trigger_self_related and cur_t > self.swap_break_point):
                return

            if name not in self.self_queries:
                return
            if layer_idx not in self.self_queries[name]:
                return

            attn_cache = self.self_queries[name][layer_idx][attention_idx]
            if cur_t not in attn_cache:
                return

            device = input[0].device

            cond_asr_enabled = (
                getattr(self, 'enable_softmask', False)
                and getattr(self, 'soft_masks_base', None) is not None
                and isinstance(getattr(self, 'edit_stem', []), list)
                and len(getattr(self, 'edit_stem', [])) > 0
            )

            if cond_asr_enabled:
                hidden_states = input[0]
                _, Q_cur, K_cur, V_cur, _, scale = attention_op(
                    model, hidden_states)

                Q_src = attn_cache[cur_t].to(device)
                K_src = self.self_keys[name][layer_idx][attention_idx][cur_t].to(
                    device)
                V_src = self.self_values[name][layer_idx][attention_idx][cur_t].to(
                    device)

                mask_tensor = self._build_conditional_asr_mask(
                    hidden_states, Q_cur.dtype, device)
                if mask_tensor is not None:
                    modified_output = apply_asr(
                        model=model,
                        attn_input=hidden_states,
                        attention_op=attention_op,
                        Q_cur=Q_cur,
                        K_cur=K_cur,
                        V_cur=V_cur,
                        Q_src=Q_src,
                        K_src=K_src,
                        mask_tensor=mask_tensor,
                        scale=scale,
                    )
                    if modified_output is not None:
                        return modified_output

            query = attn_cache[cur_t].to(device) if self.swap_self_q else None
            key = self.self_keys[name][layer_idx][attention_idx][cur_t].to(
                device) if self.swap_self_k else None
            value = self.self_values[name][layer_idx][attention_idx][cur_t].to(
                device) if self.swap_self_v else None

            _, _, _, _, modified_output, _ = attention_op(
                model, input[0], query=query, key=key, value=value)
            return modified_output
        return hook

    def prepare_inputs_for_generation(
        self,
        inputs_embeds,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        if past_key_values is not None:
            inputs_embeds = inputs_embeds[:, -1:]

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
        }

    def generate_language_model(
        self,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 8,
        **model_kwargs,
    ):
        """

        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        model_kwargs = self.language_model._get_initial_cache_position(
            inputs_embeds, model_kwargs)
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = self.language_model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat(
                [inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self.language_model._update_model_kwargs_for_generation(
                output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

    def get_text_condition(self,
                           prompt=None,
                           negative_prompt=None,
                           max_new_tokens=None,
                           num_waveforms_per_prompt=1,
                           ):
        r"""
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide audio generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The negative prompt or prompts to guide audio generation.
            max_new_tokens (`int`, *optional*, defaults to None):
                The number of new tokens to generate with the GPT2 language model.
            num_waveforms_per_prompt (`int`, defaults to 1):
                number of waveforms that should be generated per prompt
        """
        device = self.device
        if not isinstance(prompt, dict):
            if getattr(self, '_skip_clear_state_once', False):
                # 跳过一次清理
                self._skip_clear_state_once = False
            else:
                self.token_indices_per_group = {}
                self.token_group_names_in_order = []
                self.text_seq_len = 0
                self.current_prompt_type = "baseline"
        # 若 prompt 为 dict（{'vocals','bass','drums'}），走分组条件路径并返回
        if isinstance(prompt, dict):
            self.current_prompt_type = "grouped"
            return self.get_text_condition_grouped(
                target_prompt=prompt,
                negative_prompt=negative_prompt,
                max_new_tokens=max_new_tokens,
                num_waveforms_per_prompt=num_waveforms_per_prompt,
            )
            
        if prompt is None:
            return {}, None, None
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise TypeError(
                f"`prompt` should be `str` or `list`, but got {type(prompt)}")
        tokenizers = [self.tokenizer, self.tokenizer_2]
        is_vits_text_encoder = isinstance(self.text_encoder_2, VitsModel)
        if is_vits_text_encoder:
            text_encoders = [self.text_encoder,
                             self.text_encoder_2.text_encoder]
        else:
            text_encoders = [self.text_encoder, self.text_encoder_2]

        # prompt embeds encoding
        prompt_embeds_list = []
        attention_mask_list = []

        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            use_prompt = isinstance(
                tokenizer, (RobertaTokenizer, RobertaTokenizerFast,
                            T5Tokenizer, T5TokenizerFast)
            )
            if not use_prompt:
                raise NotImplementedError(
                    "`tokenizer_2` can only support T5Tokenizer, T5TokenizerFast yet")

            text_inputs = tokenizer(
                prompt,
                padding="max_length"
                if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer))
                else True,
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            attention_mask = text_inputs.attention_mask
            untruncated_ids = tokenizer(
                prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1: -1])

            text_input_ids = text_input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)

            if text_encoder.config.model_type == "clap":
                prompt_embeds = text_encoder.get_text_features(
                    text_input_ids,
                    attention_mask=attention_mask,
                )
                # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                prompt_embeds = prompt_embeds[:, None, :]
                # make sure that we attend to this single hidden-state
                attention_mask = attention_mask.new_ones((batch_size, 1))
            elif is_vits_text_encoder:
                # Add end_token_id and attention mask in the end of sequence phonemes
                for text_input_id, text_attention_mask in zip(text_input_ids, attention_mask):
                    for idx, phoneme_id in enumerate(text_input_id):
                        if phoneme_id == 0:
                            text_input_id[idx] = 182
                            text_attention_mask[idx] = 1
                            break
                prompt_embeds = text_encoder(
                    text_input_ids, attention_mask=attention_mask, padding_mask=attention_mask.unsqueeze(
                        -1)
                )
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = text_encoder(
                    text_input_ids,
                    attention_mask=attention_mask,
                )
                prompt_embeds = prompt_embeds[0]

            prompt_embeds_list.append(prompt_embeds)
            attention_mask_list.append(attention_mask)

        projection_output = self.projection_model(
            hidden_states=prompt_embeds_list[0],
            hidden_states_1=prompt_embeds_list[1],
            attention_mask=attention_mask_list[0],
            attention_mask_1=attention_mask_list[1],
        )
        projected_prompt_embeds = projection_output.hidden_states
        projected_attention_mask = projection_output.attention_mask

        generated_prompt_embeds = self.generate_language_model(
            projected_prompt_embeds,
            attention_mask=projected_attention_mask,
            max_new_tokens=max_new_tokens,
        )

        prompt_embeds = prompt_embeds.to(
            dtype=self.text_encoder_2.dtype, device=self.device)
        attention_mask = (
            attention_mask.to(device=self.device)
            if attention_mask is not None
            else torch.ones(prompt_embeds.shape[:2], dtype=torch.long, device=self.device)
        )
        generated_prompt_embeds = generated_prompt_embeds.to(
            dtype=self.language_model.dtype, device=self.device)

        bs_embed, seq_len, hidden_size = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size)

        # duplicate attention mask for each generation per prompt
        attention_mask = attention_mask.repeat(1, num_waveforms_per_prompt)
        attention_mask = attention_mask.view(
            bs_embed * num_waveforms_per_prompt, seq_len)

        bs_embed, seq_len, hidden_size = generated_prompt_embeds.shape
        # duplicate generated embeddings for each generation per prompt, using mps friendly method
        generated_prompt_embeds = generated_prompt_embeds.repeat(
            1, num_waveforms_per_prompt, 1)
        generated_prompt_embeds = generated_prompt_embeds.view(
            bs_embed * num_waveforms_per_prompt, seq_len, hidden_size
        )

        # negative prompt embeds encoding
        uncond_tokens: List[str]
        if negative_prompt is None:
            uncond_tokens = [""] * batch_size
        elif type(prompt) is not type(negative_prompt):
            raise TypeError(
                f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                f" {type(prompt)}."
            )
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        elif batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )
        else:
            uncond_tokens = negative_prompt

        negative_prompt_embeds_list = []
        negative_attention_mask_list = []
        max_length = prompt_embeds.shape[1]
        for tokenizer, text_encoder in zip(tokenizers, text_encoders):
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=tokenizer.model_max_length
                if isinstance(tokenizer, (RobertaTokenizer, RobertaTokenizerFast, VitsTokenizer))
                else max_length,
                truncation=True,
                return_tensors="pt",
            )

            uncond_input_ids = uncond_input.input_ids.to(device)
            negative_attention_mask = uncond_input.attention_mask.to(device)

            if text_encoder.config.model_type == "clap":
                negative_prompt_embeds = text_encoder.get_text_features(
                    uncond_input_ids,
                    attention_mask=negative_attention_mask,
                )
                # append the seq-len dim: (bs, hidden_size) -> (bs, seq_len, hidden_size)
                negative_prompt_embeds = negative_prompt_embeds[:, None, :]
                # make sure that we attend to this single hidden-state
                negative_attention_mask = negative_attention_mask.new_ones(
                    (batch_size, 1))
            elif is_vits_text_encoder:
                negative_prompt_embeds = torch.zeros(
                    batch_size,
                    tokenizer.model_max_length,
                    text_encoder.config.hidden_size,
                ).to(dtype=self.text_encoder_2.dtype, device=self.device)
                negative_attention_mask = torch.zeros(batch_size, tokenizer.model_max_length).to(
                    dtype=self.text_encoder_2.dtype, device=self.device
                )
            else:
                negative_prompt_embeds = text_encoder(
                    uncond_input_ids,
                    attention_mask=negative_attention_mask,
                )
                negative_prompt_embeds = negative_prompt_embeds[0]

            negative_prompt_embeds_list.append(negative_prompt_embeds)
            negative_attention_mask_list.append(negative_attention_mask)

        projection_output = self.projection_model(
            hidden_states=negative_prompt_embeds_list[0],
            hidden_states_1=negative_prompt_embeds_list[1],
            attention_mask=negative_attention_mask_list[0],
            attention_mask_1=negative_attention_mask_list[1],
        )
        negative_projected_prompt_embeds = projection_output.hidden_states
        negative_projected_attention_mask = projection_output.attention_mask

        negative_generated_prompt_embeds = self.generate_language_model(
            negative_projected_prompt_embeds,
            attention_mask=negative_projected_attention_mask,
            max_new_tokens=max_new_tokens,
        )

        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(
            dtype=self.text_encoder_2.dtype, device=device)
        negative_attention_mask = (
            negative_attention_mask.to(device=device)
            if negative_attention_mask is not None
            else torch.ones(negative_prompt_embeds.shape[:2], dtype=torch.long, device=device)
        )
        negative_generated_prompt_embeds = negative_generated_prompt_embeds.to(
            dtype=self.language_model.dtype, device=device
        )

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        negative_prompt_embeds = negative_prompt_embeds.repeat(
            1, num_waveforms_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(
            batch_size * num_waveforms_per_prompt, seq_len, -1)

        # duplicate unconditional attention mask for each generation per prompt
        negative_attention_mask = negative_attention_mask.repeat(
            1, num_waveforms_per_prompt)
        negative_attention_mask = negative_attention_mask.view(
            batch_size * num_waveforms_per_prompt, seq_len)

        # duplicate unconditional generated embeddings for each generation per prompt
        seq_len = negative_generated_prompt_embeds.shape[1]
        negative_generated_prompt_embeds = negative_generated_prompt_embeds.repeat(
            1, num_waveforms_per_prompt, 1)
        negative_generated_prompt_embeds = negative_generated_prompt_embeds.view(
            batch_size * num_waveforms_per_prompt, seq_len, -1
        )

        if self.edit_params_default['use_attention_mask']:
            denoise_kwargs = {
                'encoder_hidden_states_1': torch.cat([negative_prompt_embeds, prompt_embeds]),
                'encoder_attention_mask_1': torch.cat([negative_attention_mask, attention_mask]),
                'encoder_hidden_states': torch.cat([negative_generated_prompt_embeds, generated_prompt_embeds]),
                'return_dict': False
            }
        else:
            denoise_kwargs = {
                'encoder_hidden_states_1': torch.cat([negative_prompt_embeds, prompt_embeds]),
                'encoder_attention_mask_1': torch.cat([negative_attention_mask, torch.ones_like(attention_mask)]),
                'encoder_hidden_states': torch.cat([negative_generated_prompt_embeds, generated_prompt_embeds]),
                'return_dict': False
            }

        return denoise_kwargs

    def prepare_latents(self, batch_size=1, generator=None, latents=None, shape=None):
        if shape is None:
            shape = (
                batch_size,
                self.num_channels_latents,
                int(self.height) // self.vae_scale_factor,
                int(self.vocoder.config.model_in_dim) // self.vae_scale_factor,
            )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=self.device, dtype=self.unet.dtype)
        else:
            latents = latents.to(self.device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def mel_spectrogram_to_waveform(self, mel_spectrogram):
        if mel_spectrogram.dim() == 4:
            mel_spectrogram = mel_spectrogram.squeeze(1)
        mel_spectrogram_list = [mel_spectrogram]
        while mel_spectrogram_list[-1].shape[1] > 1536:
            temp = mel_spectrogram_list[-1][:, 1024:, :]
            mel_spectrogram_list[-1] = mel_spectrogram_list[-1][:, :1024, :]
            mel_spectrogram_list.append(temp)
        waveform_list = []
        for mel_spectrogram in mel_spectrogram_list:
            waveform = self.vocoder(mel_spectrogram)
            waveform = waveform.detach().float()
            waveform_list.append(waveform)
        waveform = torch.cat(waveform_list, dim=1)
        del waveform_list
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16

        return waveform

    def mel_spectrogram_to_audio(self, mel_spectrogram,):
        audio = self.mel_spectrogram_to_waveform(mel_spectrogram)
        audio = audio[:, :self.original_waveform_length]
        audio = audio.detach().cpu().numpy()
        return AudioPipelineOutput(audio)

    def load_wav(self, audio_path, audio_length_in_s=None, device=None):
        audio, sample_rate = torchaudio.load(audio_path)
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(audio)
        vocoder_upsample_factor = np.prod(
            self.vocoder.config.upsample_rates) / self.vocoder.config.sampling_rate

        if audio_length_in_s is None:
            audio_length_in_s = audio_16k.shape[1] / 16000
        height = int(audio_length_in_s / vocoder_upsample_factor)
        if height % self.vae_scale_factor != 0:
            height = int(np.ceil(height / self.vae_scale_factor)
                         ) * self.vae_scale_factor

        original_waveform_length = int(
            audio_length_in_s * self.vocoder.config.sampling_rate)

        fn_STFT = TacotronSTFT(
            1024,  # config["preprocessing"]["stft"]["filter_length"],
            160,  # config["preprocessing"]["stft"]["hop_length"],
            1024,  # config["preprocessing"]["stft"]["win_length"],
            64,  # config["preprocessing"]["mel"]["n_mel_channels"],
            16000,  # config["preprocessing"]["audio"]["sampling_rate"],
            0,  # config["preprocessing"]["mel"]["mel_fmin"],
            8000,  # config["preprocessing"]["mel"]["mel_fmax"],
        )

        mel, _, _ = wav_to_fbank(
            audio_path, target_length=int(audio_length_in_s * 102.4), fn_STFT=fn_STFT
        )
        mel = mel.unsqueeze(0).unsqueeze(0).to(torch.float16).to(device)

        # audio_test = self.mel_spectrogram_to_audio(mel).audios[0]
        audio_test = None

        return mel, original_waveform_length, audio_test, audio_length_in_s

    def load_npy(self, audio_path):
        mel = np.load(audio_path).T
        mel = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(device)

        return mel

    def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
        pred_type = self.scheduler.config.prediction_type
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]

        beta_prod_t = 1 - alpha_prod_t

        if pred_type == "epsilon":
            return model_output
        elif pred_type == "sample":
            return (sample - alpha_prod_t ** (0.5) * model_output) / beta_prod_t ** (0.5)
        elif pred_type == "v_prediction":
            return (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {pred_type} must be one of `epsilon`, `sample`, or `v_prediction`"
            )

    def auto_corr_loss(self, hidden_states, generator=None):
        reg_loss = 0.0
        for i in range(hidden_states.shape[0]):
            for j in range(hidden_states.shape[1]):
                noise = hidden_states[i: i + 1, j: j + 1, :, :]
                while True:
                    roll_amount = torch.randint(
                        noise.shape[2] // 2, (1,), generator=generator).item()
                    reg_loss += (noise * torch.roll(noise,
                                 shifts=roll_amount, dims=2)).mean() ** 2
                    reg_loss += (noise * torch.roll(noise,
                                 shifts=roll_amount, dims=3)).mean() ** 2

                    # if noise.shape[2] <= 8:
                    #     break
                    break  # We first consider full latent
                    # it is the problem
                    noise = F.avg_pool2d(noise, kernel_size=2)
        return reg_loss

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)

    def invert_process(self, latents, denoise_kwargs):
        device = self.device
        pred_images = []
        pred_latents = []

        decode_kwargs = {'vae': self.vae}
        guidance_scale = self.edit_params_default['guidance_scale']
        do_classifier_free_guidance = guidance_scale > 1

        num_reg_steps = 5
        num_auto_corr_rolls = 5
        lambda_auto_corr = 20.0
        lambda_kl = 20.0

        # Prepare timesteps
        self.scheduler.set_timesteps(
            self.edit_params_default['steps'], device=device)

        # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
        timesteps = reversed(self.scheduler.timesteps)
        num_inference_steps = len(self.scheduler.timesteps)

        # Reverse diffusion process
        for i in tqdm(range(0, num_inference_steps)):

            t = timesteps[i]

            # setting t (for saving time step)
            self.cur_t = t.item()

            with torch.no_grad():

                # For text condition on stable diffusion

                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                noisy_residual = self.unet(latent_model_input, t.to(
                    latent_model_input.device), **denoise_kwargs)[0]

                # For text condition on stable diffusion
                if do_classifier_free_guidance:
                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noisy_residual.chunk(
                        2)
                    noisy_residual = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                # regularization for noise prediction
                # with torch.enable_grad():
                #     for _ in range(num_reg_steps):
                #         if lambda_auto_corr > 0:
                #             for _ in range(num_auto_corr_rolls):
                #                 var = torch.autograd.Variable(noisy_residual.detach().clone(), requires_grad=True)

                #                 # Derive epsilon from model output before regularizing to IID standard normal
                #                 var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                #                 l_ac = self.auto_corr_loss(var_epsilon)
                #                 l_ac.backward()

                #                 grad = var.grad.detach() / num_auto_corr_rolls
                #                 noisy_residual = noisy_residual - lambda_auto_corr * grad

                #         if lambda_kl > 0:
                #             var = torch.autograd.Variable(noisy_residual.detach().clone(), requires_grad=True)

                #             # Derive epsilon from model output before regularizing to IID standard normal
                #             var_epsilon = self.get_epsilon(var, latent_model_input.detach(), t)

                #             l_kld = self.kl_divergence(var_epsilon)
                #             l_kld.backward()

                #             grad = var.grad.detach()
                #             noisy_residual = noisy_residual - lambda_kl * grad

                #         noisy_residual = noisy_residual.detach()

                current_t = max(0, t.item() - (1000//num_inference_steps))  # t
                # min(999, t.item() + (1000//num_inference_steps)) # t+1
                next_t = t
                alpha_t = self.scheduler.alphas_cumprod[current_t]
                alpha_t_next = self.scheduler.alphas_cumprod[next_t]

                # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
                latents = (latents - (1-alpha_t).sqrt()*noisy_residual)*(
                    alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noisy_residual
                pred_latents.append(latents)
                # pred_images.append(decode_latent(latents, **decode_kwargs))

        # last_image = decode_latent(latents, **decode_kwargs)
        last_image = None

        return pred_images, pred_latents, last_image

    def reverse_process(self, latents, denoise_kwargs, cfg=None):
        device = self.device
        pred_images = []
        pred_latents = []

        decode_kwargs = {'vae': self.vae}
        guidance_scale = self.edit_params_default['guidance_scale']
        if cfg is not None:
            guidance_scale = cfg
        do_classifier_free_guidance = guidance_scale > 1

        # Prepare timesteps
        self.scheduler.set_timesteps(
            self.edit_params_default['steps'], device=device)
        # Reverse diffusion process
        for t in tqdm(self.scheduler.timesteps):

            # setting t (for saving time step)
            self.cur_t = t.item()
            if self.cur_t > self.t_start:
                continue

            with torch.no_grad():

                # For text condition on stable diffusion

                latent_model_input = torch.cat(
                    [latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t)
                noisy_residual = self.unet(latent_model_input, t.to(
                    latent_model_input.device), **denoise_kwargs)[0]

                # For text condition on stable diffusion
                if do_classifier_free_guidance:
                    # perform guidance
                    noise_pred_uncond, noise_pred_text = noisy_residual.chunk(
                        2)
                    noisy_residual = noise_pred_uncond + guidance_scale * \
                        (noise_pred_text - noise_pred_uncond)

                # coef * P_t(e_t(x_t)) + D_t(e_t(x_t))
                prev_noisy_sample = self.scheduler.step(
                    noisy_residual, t, latents).prev_sample
                pred_original_sample = self.scheduler.step(
                    noisy_residual, t, latents).pred_original_sample    # D_t(e_t(x_t))

                latents = prev_noisy_sample

                # pred_latents.append(pred_original_sample)
                # pred_images.append(decode_latent(
                #     pred_original_sample, **decode_kwargs))

        last_image = decode_latent(latents, **decode_kwargs)

        return pred_images, pred_latents, last_image

    @torch.no_grad()
    def invert_pipeline(
        self,
        init_mel,
        init_audio_length,
        prompt=None,
        negative_prompt=None,
    ):
        self.update_audio_length_in_s(init_audio_length)
        enc_latent = encode_latent(init_mel, self.vae)
        denoise_kwargs = self.get_text_condition(
            prompt=prompt, negative_prompt=negative_prompt)
        pred_mels, pred_latents, last_mel = self.invert_process(
            enc_latent, denoise_kwargs)
        return pred_latents

    @torch.no_grad()
    def reverse_pipeline(
        self,
        prompt,
        negative_prompt=None,
        audio_length=10.0,
        init_latent=None,
    ):
        latents = self.prepare_latents(latents=init_latent)
        denoise_kwargs = self.get_text_condition(
            prompt=prompt, negative_prompt=negative_prompt)
        pred_mels, pred_latents, last_mel = self.reverse_process(
            latents, denoise_kwargs)
        return last_mel

    def get_attention_probs(self, query, key):
        baddbmm_input = torch.empty(
            query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
        )
        beta = 0
        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=64**-0.5,
        ).detach()
        A = attention_scores.softmax(dim=-1).detach()
        return A

    def get_svd_self_maps(self, k=6):
        res = {}
        for part in self.self_queries.keys():
            res[part] = {}
            for layer in tqdm(self.self_queries[part].keys()):
                res[part][layer] = {}
                A = None
                for cur_t in self.self_queries[part][layer][1].keys():
                    query = self.self_queries[part][layer][1][cur_t]
                    key = self.self_keys[part][layer][1][cur_t]
                    A_t = self.get_attention_probs(query, key)
                    if A is None:
                        A = A_t
                    else:
                        A += A_t
                A /= len(self.self_queries[part][layer][1].keys())
                A_k = torch.zeros([A.shape[0], A.shape[1], k])

                # 进行SVD分解
                U, S, VT = torch.svd(A.to(torch.float32))

                # 选择前6个奇异值进行降维
                U_k = U[:, :, :k]
                # S_k = S[:, :k]
                # VT_k = VT[:, :, :k]

                # 将降维后的矩阵存储到结果张量中
                # A_k = torch.matmul(S_k, VT)
                A_k = U_k.to(torch.float16)

                res[part][layer] = A_k
        return res

    def get_cross_maps(self, tgt_block, key_len):
        res = {}
        for part in self.cross_queries.keys():
            res[part] = {}
            if tgt_block is not None and key_len is not None:
                tgt_indexs = [i for i in range(tgt_block, key_len, 3)]
            else:
                tgt_indexs = None
            for layer in tqdm(self.cross_queries[part].keys()):
                if tgt_indexs is not None:
                    if layer in tgt_indexs:
                        continue
                A = None
                for cur_t in self.cross_queries[part][layer].keys():
                    query = self.cross_queries[part][layer][cur_t]
                    key = self.cross_keys[part][layer][cur_t]
                    A_t = self.get_attention_probs(query, key)
                    if A is None:
                        A = A_t
                    else:
                        A += A_t
                A /= len(self.cross_queries[part][layer].keys())
                res[part][layer] = A
        return res

    def save_maps(self, save_dir, save_self=False, save_cross=True):
        import os.path as osp
        if save_self:
            torch.save(self.get_svd_self_maps(), osp.join(
                save_dir, 'self_attn_maps.pth'))
        if save_cross:
            CA_maps_all = self.get_cross_maps(
                self.edit_params_default['tgt_block'], len(self.attn_keys))
            torch.save(CA_maps_all, osp.join(save_dir, 'cross_attn_maps.pth'))

    def load_maps(self, load_dir):
        import os.path as osp
        cross_path = osp.join(load_dir, 'cross_attn_maps.pth')
        self_path = osp.join(load_dir, 'self_attn_maps.pth')
        if osp.exists(cross_path):
            self.cross_features = torch.load(cross_path)
        if osp.exists(self_path):
            self.self_features = torch.load(self_path)


if __name__ == "__main__":
    import scipy
    from my_audioldm2 import encode_latent

    device = "cuda"
    vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vocoder, projection_model, language_model, unet, scheduler = load_audioldm2(
        device=device)

    edit_params = {
        'steps': 200,
        'use_attention_mask': False,
        'tgt_block': 2,
        'self_layers': {
            'up': [],
            'mid': [],
            'down': [],
        },
        'cross_layers': {
            'up': [],
            'mid': [],
            'down': [i for i in range(1, 19)],
        },
        'swap_break_point': 200
    }
    edit = Edit(vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vocoder,
                projection_model, language_model, unet, scheduler, device, edit_params=edit_params)
    # language_model, unet, scheduler = load_audioldm2(device=device)
    prompt = ""
    prompt_edit = "a solo violin music."
    # 种子也会有影响
    random.seed(50)
    torch.manual_seed(50)

    edit.trigger_cross_related = True
    edit.trigger_self_related = False
    edit.trigger_get_map = True
    edit.trigger_swap_map = False

    mel, _, audio_loaded, duration = edit.load_wav(
        '/DATA6_6T/yy/musicmagus_demo_data/saxophone_2/audio.wav', audio_length_in_s=None, device=device)
    inv_latents = edit.invert_pipeline(mel, duration, prompt=prompt)
    inv_latent = inv_latents[-1]

    edit.trigger_get_map = False
    edit.trigger_swap_map = True
    out_mel = edit.reverse_pipeline(
        prompt_edit, audio_length=duration, init_latent=inv_latent)

    audio_guitar2bass = edit.mel_spectrogram_to_audio(out_mel).audios[0]
    # scipy.io.wavfile.write(f"inverse_loaded.wav", rate=16000, data=audio_loaded)
    # scipy.io.wavfile.write(f"inverse_inved.wav", rate=16000, data=audio_loaded)
    scipy.io.wavfile.write(f"inverse_reverse_output.wav",
                           rate=16000, data=audio_guitar2bass)
