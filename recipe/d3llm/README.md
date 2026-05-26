# d3LLM Dream-Coder × DARE BGPO

## Phase 0：模型可运行性

`models/finetune_d3LLM` 为全量权重（约 7.6B），需在目录内具备 Dream 建模代码方可离线加载。

后续 BGPO 训练计划见 [`docs/d3llm-dream-clean-plan.md`](../../docs/d3llm-dream-clean-plan.md)（阶段 1 起在 Dream HF rollout 上扩展 multiblock，不新增 `d3llm_dream` model name）。

### 一次性准备

```bash
export D3LLM_ROOT=/path/to/d3LLM   # multiblock 绑定需要
bash recipe/d3llm/setup_finetune_d3llm_model_code.sh
```

脚本会从 `d3LLM/utils/utils_Dream/model/` 复制 `configuration_dream.py`、`modeling_dream.py`、`generation_utils.py`，并修正 `config.json` 的 `auto_map`。

### 验证

```bash
export HF_HUB_OFFLINE=1
export D3LLM_ROOT=/path/to/d3LLM

# 仅加载权重
python recipe/d3llm/verify_finetune_d3llm.py --mode load

# Dream 原版 diffusion（entropy）
python recipe/d3llm/verify_finetune_d3llm.py --mode vanilla --max-new-tokens 128

# d3LLM multi-block（entropy_threshold，阶段 1 rollout 目标语义）
python recipe/d3llm/verify_finetune_d3llm.py --mode multiblock --max-new-tokens 128

# 全流程（load + vanilla + multiblock）
python recipe/d3llm/verify_finetune_d3llm.py --mode all --max-new-tokens 128

# LoRA 适配器冒烟（可选）
python recipe/d3llm/verify_finetune_d3llm.py --mode multiblock --lora-path /path/to/adapter
```

### 说明

- **阶段 0（离线）**：`recipe/d3llm/d3llm_multiblock.py` 通过 `D3LLM_ROOT` 绑定，不 import verl。
- **阶段 1（训练）**：`verl/workers/rollout/dream_multiblock.py` + vendored `d3llm_dream_generate_util.py`（源自 d3LLM 官方实现），`model.name=dream`，`rollout.dllm_decode=multiblock`。
- 训练脚本：`bash recipe/dream/run_bgpo_dream_coder_d3llm.sh`（可加 `--smoke`）。

### 阶段 1 训练

```bash
bash recipe/d3llm/setup_finetune_d3llm_model_code.sh   # 若尚未执行
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke
```
