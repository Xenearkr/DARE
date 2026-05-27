# d3LLM Dream-Coder × SGLang：问题总结与排查方案

> 最后更新：2026-05-27  
> 关联 smoke：`dream-code-d3llm-bgpo-sglang-smoke-...-20260527_{105104,112919}`  
> 训练入口（可用）：`recipe/dream/run_bgpo_dream_coder_d3llm.sh`  
> 规划文档：[d3llm-dream-sglang-plan.md](./d3llm-dream-sglang-plan.md)

---

## 总览结论

| 主题 | 结论 |
|------|------|
| **训练主链路** | BGPO + `SGLangDreamRollout` smoke 可跑通；checkpoint / reward / rollout 日志在修复 metadata 后正常。 |
| **问题 1：双 nfe + 变慢** | 主因：`pad_full_generation` 时 **page 向下取整** 拆成两轮 staging（已修）；辅以 `chunked_prefill` / KV budget。 |
| **问题 2：HumanEval ~0.19** | **SGLang 验证路径** 上 base 就偏低（~17–20%），不是 1-step RL 训坏；与 **HF multiblock 单卡离线**（同权重约 **58%**）差距大，根因在 **解码/评测链路不一致**。 |
| **离线三路对比** | 曾尝试 `benchmark_humaneval_decode_paths.py`，因与 Ray 训练环境不一致（单卡、Engine 参数、依赖）未跑通，**已删除**；后续用下方「推荐工具」在 **DARE conda + 4 卡训练同环境** 下排查。 |

---

## 问题 1：`nfe` 为两个值（如 `[201, 197]`），rollout 耗时约翻倍

### 现象

```text
[dream-sglang] sample 1/4 prompt_tokens=442 done elapsed=59.48s nfe=[201, 197]
[dream-sglang] sample 2/4 prompt_tokens=442 done elapsed=26.57s nfe=177
```

- `len(nfe)>1` 时耗时常为单标量 nfe 的 ~2×。
- 每个 sample 在 VERL 里仍只调 **一次** `async_generate`。

### 根因（2026-05-27 修复）

1. **`PrefillAdder._add_dllm_req`** 对全长 Dream 序列做 `trunc_len = len // page_size * page_size`（954→928），余下 token 走 **第二轮 staging** → 两个 nfe、耗时翻倍。
2. 每轮 scheduler 结束 `append` 一次 nfe；VERL 仍只调一次 `async_generate`。

**代码修复**：`schedule_policy.py`（`pad_full_generation` 不 page-floor）、`sglang_dream_rollout.py`（`chunked_prefill_size=-1`、`max_prefill_tokens`）、smoke 脚本 `enable_chunked_prefill=False`。

### smoke 配置（修复后）

| 项 | 值 |
|----|-----|
| `enable_chunked_prefill` | **False** |
| `max_num_batched_tokens` | 4096 |
| `chunked_prefill_size`（SGLang Engine） | **-1** |
| `mem_fraction_static` | 0.32（与 FSDP 同卡） |

### 验收

同 prompt 下 `nfe` 为单值（或 `nfe_rounds=1`），耗时与 nfe 量级线性。

---

## 问题 2：HumanEval pass@1 ≈ 0.17–0.19，低于 HF 预期 ~0.5

**详细分析**：[d3llm-dream-humaneval-quality.md](./d3llm-dream-humaneval-quality.md)（失败 taxonomy、HF×SGLang 交叉表、修复优先级）。

### 已测数据

| 来源 | pass@1 | 说明 |
|------|--------|------|
| `val_generations/0.jsonl`（训前） | **29/164 ≈ 17.7%** | SGLang val，`temperature=0` |
| `val_generations/1.jsonl`（1 step 后） | **32/164 ≈ 19.5%** | 同上 |
| 单卡 HF multiblock 离线（164 题，未完成 SGLang 路） | **95/164 ≈ 57.9%** | `models/finetune_d3LLM`，`verl.workers.rollout.dream_multiblock` 同族逻辑；**非训练 Ray 路径** |

**解读**：同一本地权重在 **HF multiblock** 上可达 ~0.58，在 **当前 SGLang val 管线** 上仅 ~0.19 → 优先查 **SGLang 解码 + stop/finalize + 温度**，而非权重损坏或 1-step RL。

### 失败样本（SGLang val，约）

- 提取后 `SyntaxError`、残缺代码块、LCB 风格前言（`To solve this problem`）仍常见。
- 奖励侧 `extract_code_from_model` 取第一个含 `def` 的代码块。

### 与 ~0.5 的可能差异

1. **路径**：训练验证 = SGLang `FullAttnMultiBlock` + val_kwargs；论文/本地基线常为 HF `dream_multiblock`。
2. **超参**：val `temperature=0` vs train `0.2`；stop / finalize / chunked prefill / KV 预算（亦影响问题 1）。
3. **分布**：LCB 训练 vs HumanEval 补全格式。
4. **步数**：smoke 1–3 step 不能代表 full BGPO 上限。

### 建议排查（勿再用已删的 bulk benchmark）

1. **单条对齐**（与训练一致，在 **4×Ray worker / DARE env** 下）：  
   `recipe/d3llm/compare_sglang_train_vs_val_path.py --humaneval --row N`  
   对比 train / val 解码与 `finish_reason`、nfe、文本是否一致。

2. **HF 冒烟**：  
   `recipe/d3llm/verify_finetune_d3llm.py --mode multiblock`（单卡 sanity）。

3. **全量 HumanEval**：仍以 **`run_bgpo_dream_coder_d3llm.sh --smoke`** 的 `val_generations/*.jsonl` 为准；若要对齐 HF 全量，应在 **4 卡 data parallel** 下复用 `dream_multiblock.execute_dream_multiblock_generation`（与 `tensor_model_parallel_size=1`、每 rank 一样本循环一致），而不是单进程单卡 SGLang Engine 脚本。

4. 逐项对齐：temperature、stop_token_ids、`_finalize_dream_response_tensor`、`enable_chunked_prefill`、`mem_fraction_static`。

### 验收

- SGLang val 与 HF multiblock 在相同 164 题上差距缩小到可解释范围（&lt;5pp），或文档化不可对齐项。
- 或：在仅 SGLang 训练的前提下，证明 val 配置与 HF 等价。

---

## 已完成的训练侧修复（供对照）

- Train/val 共用 `_finalize_dream_response_tensor` + `stop_token_ids`（`sglang_dream_rollout.py`）。
- `gen_batch` 携带 `data_source` / `extra_info`，rollout 调试 reward 可算。
- `format_nfe_for_log`：多轮 nfe 打 WARN。

---

## 推荐工具（保留）

| 文件 | 用途 |
|------|------|
| `recipe/dream/run_bgpo_dream_coder_d3llm.sh` | **正式 / smoke 训练**（4 GPU Ray） |
| `recipe/d3llm/compare_sglang_train_vs_val_path.py` | 单题 SGLang train vs val |
| `recipe/d3llm/verify_finetune_d3llm.py` | HF 加载与 multiblock 冒烟 |
| `verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py` | 训练 rollout 实现 |

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05-27 | 初版：问题 1 / 2 与排查步骤 |
| 2026-05-27 | 汇总：HF 单卡 ~58% vs SGLang val ~19%；删除未跑通的 `benchmark_humaneval_decode_paths.py` 等 |
| 2026-05-27 | 新增 HumanEval 质量专文；smoke `save_freq=0` |
