#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单样本音乐编辑实验脚本
--------------------------
基于 AudioLDM2 与 Polyphonia 编辑框架，
针对单个 sample/ 目录执行免训练多轨音乐编辑。

# 基础运行（编辑 sample/ 下所有任务）
python run_sample_exp.py --sample_dir ./sample --output_dir ./sample/output --visualize

# 仅执行 vocals2violin 任务
python run_sample_exp.py --task_key vocals2violin --device cuda --seed 928
"""

from __future__ import annotations
import argparse
import copy
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from audio_utils import generate_soft_masks
from edit import Edit
from my_audioldm2 import load_audioldm2
from utils import visualize_and_save_spectrogram

logger = logging.getLogger("run_sample_exp")


def setup_logging() -> None:
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(console_handler)

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    logging.getLogger("audioldm2").setLevel(logging.WARNING)


BEST_EDIT_PARAMS: Dict[str, Any] = {
    "steps": 100,
    "use_attention_mask": False,
    "tgt_block": 2,
    "swap_break_point": 0,
    "guidance_scale": 3.5,
    "t_start": 1000,
    "mask_bias_strength": 2.5,
    "self_layers": {
        "down": [i for i in range(1, 19)],
        "up": [],
        "mid": [],
    },
    "cross_layers": {
        "down": [i for i in range(1, 19)],
        "up": [],
        "mid": [],
    },
}


DEFAULT_STFT_PARAMS = {
    "sampling_rate": 16000,
    "n_fft": 1024,
    "hop_length": 160,
    "win_length": 1024,
    "n_mels": 64,
    "mel_fmin": 0.0,
    "mel_fmax": 8000.0,
}


def select_device(device_arg: str) -> str:
    if device_arg == "cuda":
        return "cuda"
    if device_arg == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_precision(precision_arg: str, device: str) -> torch.dtype:
    if precision_arg == "fp16":
        return torch.float16
    if precision_arg == "fp32":
        return torch.float32
    return torch.float16 if device == "cuda" else torch.float32


def parse_prompt_file(prompt_json: Path, task_key: Optional[str] = None) -> Dict[str, Any]:
    try:
        with open(prompt_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning(f"读取 prompt 失败: {prompt_json}, err={exc}")
        return {}

    if not isinstance(data, dict) or not data:
        return {}

    tasks: Dict[str, Dict[str, Any]] = {}
    key_mapping = {
        "Vocals Softmask Prompt": "vocals",
        "Bass Softmask Prompt": "bass",
        "Drums Softmask Prompt": "drums",
        "Others Softmask Prompt": "other",
        "Other Softmask Prompt": "other",
    }

    for task_name, block in data.items():
        if not isinstance(block, dict):
            continue
        baseline = block.get("Baseline Prompt", "") or ""
        target_prompts: Dict[str, str] = {}
        for src_key, alias in key_mapping.items():
            val = block.get(src_key, "")
            if isinstance(val, str) and val.strip():
                target_prompts[alias] = val.strip()

        tasks[task_name] = {
            "baseline_prompt": baseline.strip(),
            "target_prompt": target_prompts,
            "raw": block,
        }

    if task_key:
        tasks = {task_key: tasks[task_key]} if task_key in tasks else {}

    return tasks


def infer_edit_stem(task_name: str) -> List[str]:
    """
    根据任务名称推断应该被编辑的乐器分轨。
    例如："vocals2violin" -> ["vocals"], "drums2percussion" -> ["drums"]
    """
    name = (task_name or "").lower()
    stems: List[str] = []
    if "vocals2" in name:
        stems.append("vocals")
    if "drums2" in name:
        stems.append("drums")
    if "bass2" in name:
        stems.append("bass")
    if not stems:
        stems.append("other")
    return stems


def save_audio_safe(path: Path, audio_np: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(str(path), audio_np, sr)
    except Exception:
        try:
            import scipy.io.wavfile as wavfile
            wavfile.write(str(path), sr, (audio_np * 32767).astype(np.int16))
        except Exception as e:
            logger.error(f"[保存失败] {path}: {e}")


def reconfigure_editor(ed: Edit, params: Dict[str, Any], preserve_cache: bool = False) -> None:
    setattr(ed, "_preserve_attn_cache", preserve_cache)
    ed.unsetting()
    ed.edit_params_default.update(params)
    ed.t_start = params.get("t_start", ed.t_start)
    ed.swap_break_point = params.get("swap_break_point", ed.swap_break_point)
    if "mask_bias_strength" in params:
        ed.mask_bias_strength = params["mask_bias_strength"]
    ed.setting()
    setattr(ed, "_preserve_attn_cache", False)


def prepare_editor(device: str, precision_t: torch.dtype, init_params: Dict[str, Any]) -> Edit:
    logger.info("正在加载 AudioLDM2 模型...")
    vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2, vocoder, projection_model, language_model, unet, scheduler = load_audioldm2(
        precision_t=precision_t, device=device
    )
    ed = Edit(
        vae, tokenizer, tokenizer_2, text_encoder, text_encoder_2,
        vocoder, projection_model, language_model, unet, scheduler,
        device, edit_params=copy.deepcopy(init_params),
    )
    ed.edit_params_default.update({"t_start": init_params["t_start"]})
    ed.t_start = init_params["t_start"]
    ed.mask_bias_strength = init_params["mask_bias_strength"]
    ed.swap_break_point = init_params["swap_break_point"]
    return ed


def build_prompt_for_generation(prompt_cfg: Dict[str, Any]) -> Any:
    baseline = prompt_cfg.get("baseline_prompt", "") or ""
    target = prompt_cfg.get("target_prompt", {}) or {}
    if baseline.strip() and isinstance(target, dict) and target:
        prompt = {"__baseline__": baseline.strip()}
        prompt.update(target)
        return prompt
    if baseline.strip():
        return baseline.strip()
    if isinstance(target, dict) and target:
        prompt = {"__baseline__": ""}
        prompt.update(target)
        return prompt
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="单样本音乐编辑实验脚本")
    parser.add_argument("--sample_dir", type=str, default="./sample",
                        help="输入样本目录，需包含 mixture.wav、prompt_multi.json 及各分轨 wav")
    parser.add_argument("--output_dir", type=str, default="./sample/output",
                        help="输出结果目录")
    parser.add_argument("--device", type=str, default="auto",
                        help="设备选择: cuda/cpu/auto")
    parser.add_argument("--precision", type=str, default="auto",
                        help="精度选择: fp16/fp32/auto")
    parser.add_argument("--seed", type=int, default=928,
                        help="随机种子")
    parser.add_argument("--visualize", action="store_true",
                        help="是否保存频谱图可视化")
    parser.add_argument("--mask_type", type=str, default="energy",
                        help="软掩码类型")
    parser.add_argument("--task_key", type=str, default=None,
                        help="可选：指定 prompt_multi.json 中的任务键，仅执行该任务")
    args = parser.parse_args()

    sample_dir = Path(args.sample_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    mixture_wav = sample_dir / "mixture.wav"
    prompt_json = sample_dir / "prompt_multi.json"

    if not mixture_wav.exists():
        logger.error(f"mixture.wav 不存在: {mixture_wav}")
        return
    if not prompt_json.exists():
        logger.error(f"prompt_multi.json 不存在: {prompt_json}")
        return

    setup_logging()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    precision_t = select_precision(args.precision, device)

    logger.info("========== 单样本音乐编辑启动 ==========")
    logger.info(f"样本目录: {sample_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"设备/精度: {device} / {precision_t}")
    logger.info(f"随机种子: {args.seed}")

    init_params = copy.deepcopy(BEST_EDIT_PARAMS)
    try:
        ed = prepare_editor(device, precision_t, init_params)
    except Exception as exc:
        logger.critical(f"模型加载失败: {exc}", exc_info=True)
        return

    # 解析 prompt_multi.json
    prompt_tasks = parse_prompt_file(prompt_json, args.task_key)
    if not prompt_tasks:
        logger.error("未解析到有效的编辑任务，退出。")
        return
    logger.info(f"发现 {len(prompt_tasks)} 个编辑任务: {list(prompt_tasks.keys())}")

    # 生成软掩码
    try:
        masks = generate_soft_masks(
            str(sample_dir), None, DEFAULT_STFT_PARAMS, 0.0, args.mask_type
        )
    except Exception as exc:
        logger.error(f"软掩码生成失败: {exc}", exc_info=True)
        return

    if not masks:
        logger.error("软掩码为空，退出。")
        return
    logger.info(f"软掩码已生成，包含分轨: {list(masks.keys())}")

    # 反演: 对 mixture.wav 执行一次 Inversion 获取初始 Latent
    try:
        reconfigure_editor(ed, init_params, preserve_cache=False)
        ed.set_soft_masks(masks)
        ed.enable_softmasking(True)
        ed.trigger_cross_related = True
        ed.trigger_self_related = True
        ed.trigger_get_map, ed.trigger_swap_map = True, False

        mel, _, _, audio_len = ed.load_wav(str(mixture_wav), None, ed.device)
        inv_latents = ed.invert_pipeline(mel, audio_len, prompt="")
        inv_latent = inv_latents[-1]
        ed.trigger_get_map = False
        logger.info(f"反演完成，音频长度: {audio_len:.2f}s")
    except Exception as exc:
        logger.error(f"反演失败: {exc}", exc_info=True)
        return

    # 配置编辑器用于生成（保留反演阶段缓存的注意力图）
    reconfigure_editor(ed, init_params, preserve_cache=True)
    ed.set_soft_masks(masks)
    ed.enable_softmasking(True)
    ed.trigger_cross_related = True
    ed.trigger_self_related = True

    # 遍历每个任务执行 Reverse 生成
    success_count = 0
    fail_count = 0

    for task_name, prompt_cfg in prompt_tasks.items():
        logger.info(f"--> 执行任务: {task_name}")

        out_wav = output_dir / f"{task_name}.wav"

        if out_wav.exists() and out_wav.stat().st_size > 1024:
            logger.info(f"[跳过] 已存在: {out_wav}")
            success_count += 1
            continue

        try:
            edit_stems = infer_edit_stem(task_name)
            ed.set_edit_stem(edit_stems)

            ed.trigger_get_map = False
            ed.trigger_swap_map = True

            prompt_to_use = build_prompt_for_generation(prompt_cfg)
            last_mel = ed.reverse_pipeline(
                prompt=prompt_to_use, audio_length=audio_len, init_latent=inv_latent
            )
            audio_np = ed.mel_spectrogram_to_audio(last_mel).audios[0]
            save_audio_safe(out_wav, audio_np)
            logger.info(f"[成功] 已保存音频: {out_wav}")

            if args.visualize:
                out_png = output_dir / f"{task_name}.png"
                visualize_and_save_spectrogram(
                    out_wav, out_png, DEFAULT_STFT_PARAMS,
                    title=f"{task_name}"
                )
                logger.info(f"[成功] 已保存频谱图: {out_png}")

            success_count += 1
        except Exception as exc:
            logger.error(f"[失败] 任务 {task_name}: {exc}", exc_info=True)
            fail_count += 1
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    logger.info("========== 单样本音乐编辑完成 ==========")
    logger.info(f"成功: {success_count} | 失败: {fail_count}")


if __name__ == "__main__":
    main()
