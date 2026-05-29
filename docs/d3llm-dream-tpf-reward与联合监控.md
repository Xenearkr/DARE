# d3LLM Dream-Coder：TPF 效率奖励与联合监控

> 背景：第一次全量 BGPO（见 [第二阶段文档 §11](./第二阶段D3LLM-Dream-Coder+BGPO兼容与debug内容.md#11-第一次全量-bgpo-训练复盘效果不佳)）出现 **HumanEval acc 先升后跌**，且离线分析显示 **acc 略升时 TPF（tokens per forward）反而下降**——解码变「更犹豫」、实际能力退化。  
> 本文记录 2026-05-29 起的改动思路与实现。

---

## 1. 核心思路

### 1.1 为什么用 TPF 而不是 NFE

- **NFE**（number of forward evaluations）是绝对步数，与 `max_response_length`、block 完成度强相关，不同样本不可比。
- **TPF** = `gen_tokens / NFE`（有效生成 token 数 / 前向次数），衡量 **每步前向「产出」多少 token**，与 `recipe/d3llm/benchmark_humaneval_pipelines.py` 一致。
- 效率奖励应对 **通过测试的样本** 鼓励更高 TPF（更少犹豫、更自信的块解码），而非单纯惩罚 NFE 大。

### 1.2 Baseline 必须排除失败样本

失败样本往往 NFE 高、TPF 低（反复 mask/unmask 仍错），若纳入 baseline 会：

1. 把 baseline 拉低，给「瞎猜但偶尔 pass」的样本虚高 efficiency bonus；  
2. 干扰 EMA 对「健康解码」的刻画。

因此 **`TpfBaselineTracker.observe_passed()` 仅接受 `acc=True` 且 `tpf>0` 的样本**。

### 1.3 奖励分解

| 分量 | 训练 | 验证 |
|------|------|------|
| `pass_reward` | 0/1（EvalPlus 判题） | 同左 |
| `efficiency_reward` | `coef * (tpf/baseline - 1)`，裁剪后；**仅 passed** | **固定 0**（只监控） |
| `reward` | `pass_reward + efficiency_reward` | = `pass_reward` |

默认超参（`run_bgpo_dream_coder_d3llm.sh`）：

```bash
+reward_model.reward_kwargs.enable_tpf_efficiency=True
+reward_model.reward_kwargs.tpf_efficiency_coef=0.1
+reward_model.reward_kwargs.tpf_baseline_ema_alpha=0.1
+reward_model.reward_kwargs.tpf_baseline_initial=2.0
+reward_model.reward_kwargs.tpf_efficiency_max_bonus=0.25   # 默认
+reward_model.reward_kwargs.tpf_efficiency_max_penalty=0.25
```

---

## 2. 代码改动清单

### 2.1 奖励与 TPF 工具

| 文件 | 内容 |
|------|------|
| `verl/utils/reward_score/code_efficiency.py` | `compute_tpf`、`normalize_rollout_nfe`、`TpfBaselineTracker` |
| `verl/workers/reward_manager/dllm.py` | `DLLMRewardManager`：两阶段（先 pass → 更新 baseline → 加 efficiency）；`reward_extra_info` 含 `pass_reward`、`reward`、`efficiency_reward`、`tpf`、`tpf_baseline` |
| `verl/utils/reward_score/code_reward.py` | 返回显式 `pass_reward` |
| `verl/utils/reward_score/__init__.py` | `dllm_rm` 透传 TPF 相关字段 |

### 2.2 Rollout 传递解码统计

| 文件 | 字段 |
|------|------|
| `verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py` | `non_tensor_batch["rollout_nfe"]`（SGLang `meta_info.nfe` 求和）、`rollout_gen_tokens`（response mask 计数） |
| `verl/workers/rollout/dream_multiblock.py` | 每 sample `nfe` → `gen_kwargs["_sample_nfes"]` |
| `verl/workers/rollout/fast_dream_rollout.py` | HF 路径写入 `non_tensor_batch` |

### 2.3 W&B / 验证监控

| 文件 | 指标 |
|------|------|
| `verl/trainer/ppo/dllm_metric_utils.py` | `compute_reward_extra_metrics`：`pass_reward/mean`、`reward/mean`、`efficiency_reward/mean`、`tpf/mean`、`tpf_passed/mean`、`tpf_baseline/mean`、`rollout_nfe/mean`、`rollout_tpf/mean` |
| `verl/trainer/ppo/dllm_ray_trainer.py` | 训练 loop 调用 `compute_reward_extra_metrics` |
| 验证 | `process_validation_metrics` 自动汇总 `pass_reward`、`reward`、`tpf` 等到 `val-aux/*`；`acc` 仍在 `val-core/*` |

---

## 3. 与 EvalPlus 数据/提取修复的关系

本次 TPF 改动与下列项 **正交但应一起使用**：

1. **混训数据** `code_evalplus_mix_1.parquet`（mbpp + taco + lcbv5 + primeintellect，EvalPlus 续写 prompt）  
2. **验证集** `humaneval_evalplus_1.parquet` / smoke `humaneval_evalplus_smoke_8.parquet`  
3. **代码提取** `extract_code_from_model`：支持 prompt 含开围栏、response 仅闭围栏或裸代码  

否则 pass_reward 本身不可信，TPF baseline 也会被错误样本污染。

---

## 4. 系数调参建议

| 参数 | 默认 | 调参方向 |
|------|------|----------|
| `tpf_efficiency_coef` | 0.1 | smoke 看 `efficiency_reward/mean` 量级；若 \|eff\| ≫ pass_reward 则降至 0.05；若几乎为 0 可升至 0.15 |
| `tpf_baseline_initial` | 2.0 | 接近离线 HF multiblock 典型 TPF（可用 benchmark 脚本估） |
| `tpf_baseline_ema_alpha` | 0.1 | 越大 baseline 跟当前 batch 越紧；全量训练可保持 0.05–0.1 |
| `max_bonus/penalty` | 0.25 | 防止 efficiency 项压过 pass_reward（0/1） |

**健康 smoke 信号**（1 train step 后）：

- `pass_reward/mean` 与 `reward/mean` 同时出现且 `reward ≥ pass_reward`  
- `tpf/mean`、`rollout_nfe/mean` 非 0  
- `efficiency_reward/mean` 在 passed 子集上有非零值  
- `tpf_passed/mean` 与 `val-aux/humaneval/tpf/mean@1` 量级合理（通常 1.5–4，视 block 与长度而定）

---

## 5. 仍待实验

- `use_kl_loss=True`（`kl_loss_coef=0.005~0.01`）抑制 step 120+ 漂移  
- `train/val temperature=0.0` 与当前 0.2 的对照  
- 全量重训：同时看 `val-core/humaneval/acc` **与** `val-aux/humaneval/tpf` 是否同向改善  

---

## 6. Smoke 验证结果（2026-05-29）

日志：`logs/DARE/dream-code-d3llm-bgpo-sglang-smoke-bsz4-n4-prompt1024-response1024-bl32-lr5e-7-temp0.2-gpu4-20260529_135721/`

| 指标 | step0 (val) | step1 (train) | step1 (val) |
|------|-------------|---------------|-------------|
| `val-core/humaneval/acc` | **100%** | — | **100%** |
| `pass_reward/mean` | 1.0 (val-aux) | 0.5 | 1.0 (val-aux) |
| `reward/mean` | ~0.991 | **0.494** | ~0.997 |
| `efficiency_reward/mean` | ~-0.009* | **-0.006** | ~-0.003* |
| `tpf/mean` | 2.79 (val) | 1.43 | 2.82 (val) |
| `tpf_passed/mean` | — | **1.93** | — |
| `rollout_nfe/mean` | 48.6 | 51.7 | 47.3 |

\* step0/1 val 上 `efficiency_reward` 非零为 bug（`test_batch.meta_info` 缺 `validate=True`），已在 `dllm_ray_trainer._validate` 修复；修复后 val 的 `reward` 应等于 `pass_reward`。

**系数结论（`tpf_efficiency_coef=0.1`）**：

- 训练步 `|efficiency_reward/mean|≈0.006` ≪ `pass_reward/mean=0.5`，塑形强度适中，**暂不调低**。
- passed 样本 TPF（1.93）略低于 baseline（2.21），效率项小幅为负，符合预期。
- 若全量训练中出现「acc 升、TPF 持续跌」，可先将 coef 降至 **0.05**；若 efficiency 几乎为 0 可升至 **0.15**。

---

## 7. 相关命令

```bash
# smoke（含 TPF 奖励 + W&B）
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine sglang

# 离线 TPF 对照
python recipe/d3llm/benchmark_humaneval_pipelines.py  # 见脚本内说明
```
