<div align="center">

# Polyphonia 🎶  
### Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1%2Bcu124-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2605.10203-b31b1b.svg)](https://arxiv.org/abs/2605.10203)
[![Demo](https://img.shields.io/badge/Demo-Project_Page-4b44ce.svg)](https://polyphonia2026.github.io/polyphonia-demo/)
[![Conference](https://img.shields.io/badge/ICML-2026-0066cc.svg)](https://icml.cc/)

</div>

---

## Description

This repository provides the official implementation of **Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration**, accepted to **ICML 2026**. Polyphonia is a zero-shot music editing framework for **stem-specific timbre transfer** in polyphonic mixtures. Instead of relying only on semantic cross-attention, which often lacks sufficient spectral resolution in dense music mixtures, Polyphonia introduces **Acoustic-Informed Attention Calibration** to inject probabilistic acoustic priors into the diffusion editing process. This enables precise target-stem timbre transfer while preserving non-target accompaniment and overall musical coherence.

🎧 **Demo page:** [https://polyphonia2026.github.io/polyphonia-demo/](https://polyphonia2026.github.io/polyphonia-demo/)

---

## Installation

We recommend using a clean Conda environment.

```bash
git clone https://github.com/Ltx-Leif/polyphonia.git
cd polyphonia

conda create -n polyphonia python=3.8.20 -y
conda activate polyphonia

pip install -r requirements.txt
```

The provided `requirements.txt` pins the main dependencies, including:

```text
torch==2.4.1+cu124
torchaudio==2.4.1+cu124
torchvision==0.19.1+cu124
diffusers==0.32.0
transformers==4.46.3
```

The current dependency file is configured for **CUDA 12.4 PyTorch wheels**. If your CUDA version or hardware environment differs, please install the PyTorch build that matches your system first, and then adjust the PyTorch-related entries in `requirements.txt` accordingly.

---

## Code Structure

```text
polyphonia/
├── run_sample_exp.py      # inference script for single-sample editing
├── edit.py                # Core Polyphonia editor and attention calibration mechanisms
├── my_audioldm2.py        # AudioLDM2 loading
├── audio_utils.py         # Audio loading and acoustic mask generation
├── utils.py               # Attention calibration tools and visualization helpers
└── requirements.txt     
```

---

## Quick Start

### 1. Prepare input files

The default inference script expects a sample directory with the following structure:

```text
sample/
├── mixture.wav
├── vocals.wav
├── bass.wav
├── drums.wav
├── other.wav
└── prompt_multi.json
```

- `mixture.wav` is required.
- Stem files such as `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav` are used to construct acoustic soft masks.
- `prompt_multi.json` defines one or more editing tasks.

A `prompt_multi.json` example:

```json
{
  "vocals2violin": {
    "Baseline Prompt": "A recording of vocals with bass, drums, and accompaniment.",
    "Vocals Softmask Prompt": "A recording of a gentle violin with bass, drums, and accompaniment."
  }
}
```

The task name is used to infer the edited stem. For example:

- `vocals2violin` edits the `vocals` stem.
- `drums2percussion` edits the `drums` stem.
- `bass2synth` edits the `bass` stem.
- Other task names default to the `other` stem.

---

### 2. Run all tasks in a sample directory

```bash
python run_sample_exp.py \
  --sample_dir ./sample \
  --output_dir ./sample/output \
  --device cuda \
  --precision fp16 \
  --seed 928 \
  --visualize
```

This command will:

1. Load `./sample/mixture.wav`.
2. Parse all editing tasks from `./sample/prompt_multi.json`.
3. Generate acoustic soft masks from the available stem files.
4. Perform AudioLDM2 inversion once for the input mixture.
5. Run Polyphonia editing for each task.
6. Save edited audio files to `./sample/output/`.
7. Save spectrogram visualizations when `--visualize` is enabled.

Expected outputs:

```text
sample/output/
├── vocals2violin.wav
└── vocals2violin.png      # only generated when --visualize is enabled
```

---

### 3. Run a specific editing task

```bash
python run_sample_exp.py \
  --sample_dir ./sample \
  --output_dir ./sample/output \
  --task_key vocals2violin \
  --device cuda \
  --precision fp16 \
  --seed 928 \
  --mask_type energy \
  --visualize
```

Only the task named `vocals2violin` in `prompt_multi.json` will be executed.

---

## Citation

If you find this repository useful for your research, please cite our paper:

```bibtex
@misc{li2026polyphoniazeroshottimbretransfer,
      title={Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration}, 
      author={Haowen Li and Tianxiang Li and Yi Yang and Boyu Cao and Qi Liu},
      year={2026},
      eprint={2605.10203},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2605.10203}, 
}
```

---

## Acknowledgments

This project builds upon several excellent open-source projects and pretrained models. In particular, we thank the authors and maintainers of **AudioLDM2**, **Diffusers**, **Transformers**, **PyTorch**, **Librosa**, and related audio-generation toolkits for making their code and models publicly available.

We also thank the broader music generation and audio editing research community for providing valuable open-source implementations and evaluation resources that have supported reproducible research in zero-shot music editing.
