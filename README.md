<p align="center">
  <img src="assets/DARE_logo.png" style="width:60%; height:auto;">
</p>

<div align="center">

<h2>DARE: dLLM Alignment and Reinforcement Executor</h2>

<img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
<img src="https://visitor-badge.laobi.icu/badge?page_id=yjyddq.DARE" />
<img src="https://img.shields.io/github/stars/yjyddq/DARE?style=flat-square&logo=github&label=Stars" alt="Stars">
<img src="https://img.shields.io/github/issues/yjyddq/DARE?color=red&label=Issues" alt="Open Issues">
<img src="https://img.shields.io/github/issues-closed/yjyddq/DARE?color=success&label=Issues" alt="Closed Issues">

</div>


## 🎯 Overview

We introduce **DARE** (**d**LLM **A**lignment and **R**einforcement **E**xecutor), a flexible and efficient supervised-finetuning (SFT) and reinforcement learning (RL) training framework designed specifically for diffusion large language models (dLLMs). DARE also integrates dLLMs into a comprehensive evaluation platform. It aims to be both flexible and user-friendly to use with:
- Easy extension of diverse RL algorithms for dLLMs
- Easy extension of extra benchmark evaluations for dLLMs
- Easy integration of popular and upcoming dLLM infras and HuggingFace weights

DARE is a work in progress, we plan to support more models and algorithm for training and evaluation. **We warmly welcome the research community to collaborations, give feedback and share suggestions.** Let's advance the diffusion large language models together !!!👊

<!-- **Optimization Plan in RL Pipeline**
> [!TIP]
> For MDLMs (LLaDA or Dream), we decouple the attention backend used during training from that used during rollout. Rollout uses `flash_attn_func` or `flash_attn_with_kvcache` for KV-cache, training adopts `flash_attn_varlen_func` to skip meaningless computation on padding tokens. The entire pipeline speed-up approximately ⚡️ **4×**.

<p align="center">
  <img src="assets/optimization_plan_mdlm.png" style="width:85%; height:auto;">
</p>

> [!TIP]
> For BDLMs like SDAR: We rollout with compatible lmdeploy inference and adopt SDAR's logits-free `fused_linear_cross_entropy` to cut memory usage, enable online weights update for rollout policy. The entire pipeline will be accelerated more than ⚡️ **14×**.

<p align="center">
  <img src="assets/optimization_plan_bdlm.png" style="width:85%; height:auto;">
</p> -->


## 📢 News
- [2026-03-03]: Support mdpo for LLaDA/Dream.
- [2026-03-02]: Support SGLang for SDAR rl rollout.
- [2026-02-28]: Several errors/bugs/updates for LLaDA/Dream sequence parallel have been fixed/adapted.
- [2026-02-27]: Support evaluation of SDAR with SGLang.
- [2026-02-26]: Update LLaDA cj-grpo and add Dream cj-grpo.
- [2025-12-28]: Several errors/bugs/updates in dp_actor_algorithm have been fixed/adapted.
- [2025-12-24]: Support online rl (online weight update of rollout) for SDAR.
- [2025-12-23]: Support vrpo (preference optimization) for Dream.
- [2025-12-16]: Support vrpo (preference optimization) for LLaDA.
- [2025-12-12]: Support sft/peft of SDAR.
- [2025-12-11]: Support evaluation of LLaDAMoE and LLaDA2.0-mini with SGLang.
- [2025-12-08]: Support coupled-grpo, cj-grpo and spg algorithm.
- [2025-12-03]: Support sequence parallel to enable longer generation ability for dLLMs.
- [2025-12-01]: We initialize the codebase of DARE (dLLM Alignment and Reinforcement Executor), including faster sft/peft/rl (d1, bgpo) training (LLaDA/Dream) and evaluation (LLaDA/Dream/SDAR).


## 🔍 Catalogue

- [🎯 Overview](#-overview)
- [🏆 Key Features](#-key-features)
- [🛠️ Installation and Setup](#️-installation-and-setup)
- [🏋️ Training](#️-training)
- [📊 Evaluation](#-evaluation)
- [📈 Performance](#-performance)
- [📦 Supported Models](#-supported-models)
- [🌱 Supported RL Algorithms](#-supported-rl-algorithms)
- [📧 Contact](#-contact)
- [📚 Citation](#-citation)
- [🙏 Acknowledgments](#-acknowledgments)


## 🏆 Key Features

- **Acceleration Inference/Rollout for dLLMs**
  - Block cache ([Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)) for LLaDAs and Dreams 2.2x faster rollout
  - Inference engine ([lmdeploy](https://github.com/InternLM/lmdeploy), [sglang](https://github.com/sgl-project/sglang)) for SDARs 2-4× faster rollout
- **Parallelism for dLLMs**
  - Support sequence parallel
- **Attention Backend**
  - Support flash_attn backend
  - Support flash_attn_varlen backend
  - Support flash_attn_with_kvcache backend
- **Model Diversity**
  - Masked diffusion language models (e.g., LLaDA/Dream)
  - Block diffusion language model (e.g., SDAR/LLaDA2.0)
- **Comprehensive Evaluation for dLLMs**
  - Integrate faster dLLM evaluation in [opencompass](https://github.com/open-compass/opencompass)
- **Upcoming Features**
  - Support MoE, Multi-Modal, Omni, etc.


## 🛠️ Installation and Setup

Our training framework is built on top of [verl](https://github.com/volcengine/verl), providing a robust foundation for supervised finetuning and reinforcement learning experiments, and our evaluation framework is built on the top of [opencompass](https://github.com/open-compass/opencompass), providing a comprehensive and fast evaluations.

> [!NOTE]
> Due to some **irreconcilable dependency conflicts** between packages, we **strongly recommend using two separate virtual environments**, for training and evaluation, respectively.


### 🚀 Quick Installation

Clone the DARE repo:
```bash
git clone https://github.com/yjyddq/DARE
```

Build training vitual environment:

```bash
# Create and activate environment
conda create -n DARE python=3.10 -y
conda activate DARE

# Install dependencies
cd DARE
pip install -r requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
# or (Recommend)
# install from whl
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# pip install flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

Build evaluation vitual environment:

```bash
# Create and activate environment
conda create --name opencompass python=3.10 -y
conda activate opencompass

# Install dependencies
cd DARE/opencompass
pip install -e .

# For HumanEval evaluation, install the additional dependency:
git clone https://github.com/open-compass/human-eval.git
cd human-eval && pip install -e .
cd ..

# For Math evaluation, pip install the additional dependency:
pip install math_verify latex2sympy2_extended

## Full installation (with support for more datasets)
# pip install "opencompass[full]"

## Environment with model acceleration frameworks
# pip install "opencompass[lmdeploy]"
# or
# pip install lmdeploy==0.10.1
```

### 🔧 Model Setup

After downloading [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), replace the source files with our modified versions to enable several key features:

```bash
# Copy modified files to your LLaDA model directory
cp models/xxx/* <path_to_llada_model>/
```

Or you can move the model weights (`.safetensors`) to `model/xxx/*`

```bash
# Copy weights to models/xxx/ directory
cp <path_to_llada_model>/*.safetensors models/xxx/
```

Also for Dream, SDAR, etc.

> [!NOTE]
> Since optimization plan in RL pipeline (various attention-computation backend), this step is indispensable.


### 🗄️ Dataset Setup

Preprocessed datasets is under `data/preprocessed`. Please refer `verl.utils.preprocess` to organize datasets. 


## 🏋️ Training

### 🚀 SFT Quick Start

```bash
bash scripts/run_sft.sh # | scripts/run_sft_peft.sh
```

Alternatively, use/write scripts in recipe/xxx/run_xxx.sh

```bash
# peft for llada_8b_instruct
bash recipe/run_sft_peft_llada_8b_instruct.sh 

# sft for dream_7b_instruct
bash recipe/run_sft_dream_7b_instruct.sh 

# peft for sdar_8b_chat
bash recipe/run_sft_peft_sdar_8b_chat.sh 
```

### 🚀 RL Quick Start

```bash
# online rl for llada_8b_instruct
bash recipe/run_d1_llada_8b_instruct.sh --task math # use Fast-dLLM for rollout acceleration

# online rl for dream_7b_instruct
bash recipe/run_coupled_grpo_dream_7b_instruct.sh --task math # use Fast-dLLM for rollout acceleration

# online rl for sdar_8b_chat
bash recipe/run_bgpo_sdar_8b_chat.sh --task math # use lmdeploy engine for rollout acceleration
```

### 🚀 DPO/VRPO Quick Start

Run an example for preference optimization. First download [argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned), then run `scripts/preprocess_dpo_dataset.sh` to save `ultrafeedback.parquet` under `data/preprocessed/dpo/train` and `data/preprocessed/dpo/test`

```bash
# preference optimization for llada_8b_instruct
bash recipe/run_vrpo_llada_8b_instruct.sh --task ultrafeedback

# preference optimization for dream_7b_instruct
bash recipe/run_vrpo_dream_7b_instruct.sh --task ultrafeedback
```


## 📊 Evaluation

### 🚀 Convert FSDP Sharded Checkpoints to HuggingFace Safetensors


### 🚀 Eval on OpenCompass's Bench Quick Start

First, please follow [opencompass](https://github.com/open-compass/opencompass) for benchmark dataset preparation. Then, you need to specify the model path in `opencompass/opencompass/configs/models/dllm/*`. For example `llada_instruct_8b.py`:

```bash
from opencompass.models import LLaDAModel

models = [
    dict(
        type=LLaDAModel,
        abbr='llada-8b-instruct',
        path='/TO/YOUR/PATH', # Need to modify
        max_out_len=1024,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    )
]
```

Evaluation of LLaDA-8B-Instruct on mmlu with hf backend:

```bash
bash scripts/eval_llada.sh --task mmlu
```

Evaluation of SDAR-8B-Chat on mmlu with lmdeploy backend:

```bash
bash scripts/eval_sdar_8b_chat.sh --task mmlu --engine lmdeploy
```

### 🚀 Eval on Local Bench Quick Start

If you want to add more benchmarks, models, or custom datasets, please refer to the [Evaluation Guideline](https://github.com/yjyddq/DARE/blob/main/opencompass/README.md).




## 📦 Supported Models

| Model | Params | Training Support | Evaluation Support | Inference Acceleration |
|-------|------------|------------------|--------------------|-------------------------------|
| **LLaDA-8B-Base** | 8B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **LLaDA-8B-Instruct** | 8B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **LLaDA-1.5** | 8B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **Dream-7B-Instruct** | 7B | sft/rl | ✅ | hf [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) |
| **SDAR-1.7B-Chat** | 1.7B | sft/rl | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |
| **SDAR-4B-Chat** | 4B | sft/rl | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |
| **SDAR-8B-Chat** | 8B | sft/rl | ✅ | [lmdeploy](https://github.com/InternLM/lmdeploy) [SGLang](https://github.com/sgl-project/sglang) |


## 🌱 Supported RL Algorithms

| Algorithm | Arxiv | Source Code |
|-------|------------|------------------------|
| **d1** | [2504.12216](https://arxiv.org/pdf/2504.12216) | [dllm-reasoning/d1](https://github.com/dllm-reasoning/d1) |
| **vrpo** | [2505.19223](https://arxiv.org/abs/2505.19223) | [ML-GSAI/LLaDA-1.5](https://github.com/ML-GSAI/LLaDA-1.5) (closed source) |
| **coupled-grpo** | [2506.20639](https://arxiv.org/pdf/2506.20639) | [apple/ml-diffucoder](https://github.com/apple/ml-diffucoder) |
| **mdpo** | [2508.13148](https://arxiv.org/pdf/2508.13148) | [autonomousvision/mdpo](https://github.com/autonomousvision/mdpo) |
| **cj-grpo** | [2509.23924](https://arxiv.org/pdf/2509.23924) | [yjyddq/EOSER-ASS-RL](https://github.com/yjyddq/EOSER-ASS-RL) |
| **spg** | [2510.09541](https://arxiv.org/pdf/2510.09541) | [facebookresearch/SPG](https://github.com/facebookresearch/SPG) |
| **bgpo** | [2510.11683](https://arxiv.org/pdf/2510.11683) | [THU-KEG/BGPO](https://github.com/THU-KEG/BGPO) |


## 📈 Performance

**Evaluation Result Reproduction**

| Bench\Model | LLaDA-8B | LLaDA-8B + Fast-dLLM | Dream-7B | SDAR-8B | SDAR-8B + lmdeploy | SDAR-8B + SGLang |
|-------|------------|------------------------|-------|------------|--------------------|----------------|
| **MMLU** | 65.24 | 65.17 | 66.83 | 76.66 | 73.66 | 77.23 |
| **MMLU-Pro** | 36.82 | 34.58 | 31.89 | 56.49 | 47.39 | 55.38 |
| **Hellaswag** | 75.30 | 74.41 | 63.23 | 84.07 | 87.59 | 81.78 |
| **ARC-C** | 87.80 | 87.80 | 81.36 | 75.59 | 86.78 | 76.95 |
| **GSM8k** | 79.68 | 78.39 | 83.24 | 91.36 | 91.21 | 90.83 |
| **MATH** | 41.08 | 40.58 | 48.02 | 78.40 | 61.80 | 77.00 |
| **GPQA** | 30.81 | 31.82 | 26.77 | 33.33 | 41.40 | 29.80 |
| **AIME24** | 0.83 | 2.08 | 0.83 | 8.75 | 6.67 | 13.33 |
| **AIME25** | 0.42 | 0.00 | 0.00 | 12.50 | 6.67 | 16.67 |
| **Olympiad** | 8.95 | 9.70 | 12.22 | 24.93 | 17.35 | 23.88 |
| **HumanEval** | 46.34 | 43.29 | 78.05 | 79.88 | 75.61 | 75.00 |
| **MBPP** | 38.80 | 20.00 | 56.40 | 66.20 | 67.32 | 71.60 |

**Algorithm Comparison (Same Block Decoding Strategy)**

| Bench\Algo | Baseline (LLaDA-8B-Instruct) | d1 | Coupled-GRPO | VRPO | CJ-GRPO | SPG | BGPO |
|------------|----------|-----|-------------|------|---------|-----|-----|
|           |      | **Mathematics** |      |      |         |     |    |
| **GSM8k** | 76.5 | 83.7 | 85.3 | 81.9 | 85.6 | 83.5 | 82.3 |
| **MATH** | 34.6 | 40.6 | 41.0 | 35.8 | 39.2 | 40.6 | 40.0 |
|           |      | **Coding** |      |      |         |     |    |
| **HumanEval** | 46.9 | 47.6 | 45.1 | 52.4 | 45.1 | 48.8 | 45.1 |
| **MBPP** | 37.9 | 39.1 | 38.1 | 42.8 | 40.9 | 41.9 | 40.3 |
|           |      | **Planning** |      |      |         |     |    |
| **Countdown** | 16.8 | 10.7 | 77.9 | 21.5 | 41.1 | 10.1 | 10.0 |
| **Sudoku** | 26.2 | 31.8 | 21.3 | 29.0 | 25.0 | 27.9 | 42.6 |


| Bench\Algo | Baseline (Dream-7B-Instruct) | d1 | Coupled-GRPO | CJ-GRPO | SPG | BGPO |
|------------|----------|-----|-------------|---------|-----|-----|
|           |      | **Mathematics** |      |      |        |     |
| **GSM8k** | 77.2 | 82.5 | 80.3 | 85.7 | 59.4 | 83.9 |
| **MATH** | 39.6 | 49.7 | 40.4 | 50.7 | 25.2 | 48.9 |
|           |      | **Coding** |      |         |      |      |
| **HumanEval** | 57.9 | 60.7 | 61.6 | 58.5 | 17.7 | 56.7 |
| **MBPP** | 56.2 | 56.5 | 60.3 | 57.5 | 54.4 | 58.7 |


## 📧 Contact

For any questions or collaboration inquiries, feel free to reach out Jingyi Yang at: [yangjingyi946@gmail.com](yangjingyi946@gmail.com).


## 👷‍♂️ Contributor

Waiting for your joining and contribution.



## 📚 Citation

If you find our work useful, please consider citing:

```bibtex
@article{yang2025dare,
  title={DARE: dLLM Alignment and Reinforcement Executor},
  author={Yang, Jingyi, Jiang Yuxian, Hu Xuhao, Shao Jing},
  journal={URL https://github.com/yjyddq/DARE}
}

@article{yang2025taming,
  title={Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step},
  author={Yang, Jingyi and Chen, Guanxu and Hu, Xuhao and Shao, Jing},
  journal={arXiv preprint arXiv:2509.23924},
  year={2025}
}
```


## 🙏 Acknowledgments

We thank the open-source community for their wonderful work and valuable contributions:
- Models Source: [GSAI-ML](https://huggingface.co/GSAI-ML), [Dream-org](https://huggingface.co/Dream-org), [JetLM](https://huggingface.co/JetLM), [huggingface](https://huggingface.co/) for model supply or hosting
- Algorithm: [d1](https://github.com/dllm-reasoning/d1), [CJ-GRPO](https://github.com/yjyddq/EOSER-ASS-RL), [MDPO](https://github.com/autonomousvision/mdpo), [Coupled-GRPO](https://github.com/apple/ml-diffucoder), [SPG](https://github.com/facebookresearch/SPG)
- Training Framework: [verl](https://github.com/volcengine/verl), [BGPO](https://github.com/THU-KEG/BGPO), [Dream](https://github.com/DreamLM/Dream)
- Inference Acceleration/Engine: [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM), [lmdeploy](https://github.com/InternLM/lmdeploy)
- Evaluation Framework: [opencompass](https://github.com/open-compass/opencompass), [LLaDA](https://github.com/ML-GSAI/LLaDA)


