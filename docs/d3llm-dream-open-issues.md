# d3LLM Dream-Coder × SGLang：待解决问题与排查方案

> 记录时间：2026-05-27  
> 关联 smoke：`dream-code-d3llm-bgpo-sglang-smoke-...-20260527_112919`  
> 关联实现：`SGLangDreamRollout`、`FullAttnMultiBlock`、`dllm_ray_trainer` 验证路径

---

## 问题 1：`nfe` 出现两个值（如 `[201, 197]`），且 rollout 明显变慢

### 现象

`rollout_debug` 中常见：

```text
[dream-sglang] sample 1/4 prompt_tokens=442 done elapsed=59.48s nfe=[201, 197]
[dream-sglang] sample 2/4 prompt_tokens=442 done elapsed=26.57s nfe=177
```

- `nfe` 为 **长度 2 的列表** 时，耗时往往 ~2× 于单标量 `nfe=177`（同 prompt 长度）。
- 用户怀疑「跑了两遍」——**方向正确**：不是 VERL 对同一样本调了两次 `async_generate`，而是 **SGLang 内对同一请求做了多轮 DLLM scheduler round**。

### 机制（根因）

1. **Dream 在 SGLang 中标记为 `needs_full_prefill=True`**（`third_party/sglang/python/sglang/srt/dllm/config.py`）：双向模型不能复用 prefix KV，每轮 scheduler 都要带全长上下文做 forward。

2. **`FullAttnMultiBlock` 在一次 `run()` 内**用 `while` 循环完成 block diffusion，结束时设置  
   `customized_info = {"nfe": [nfe] * batch_size}`（单次 forward 内的总 NFE）。

3. **SGLang DLLM 调度**（`dllm/mixin/scheduler.py`）在 **每个 scheduler round** 结束后调用  
   `maybe_collect_customized_info()`，把该轮的 `nfe` **append** 到 `req.customized_info["nfe"]`（见 `scheduler_output_processor_mixin.py`）。

4. 当 **单条请求因 KV / token budget 被拆成多轮 staging**（`add_dllm_staging_req` 对 `extend_input_len` 做截断）时：
   - 第 1 轮：部分 mask 解码 → 记录 `nfe≈201`
   - 第 2 轮：继续解码 → 再记录 `nfe≈197`
   - 客户端 `meta_info["nfe"]` = `[201, 197]`（列表原样透出）

5. VERL 侧 **每个 sample 仍只调用一次** `async_generate`；慢是因为 **引擎内多轮 forward**，不是 Python 循环调了两次 API。

### 与配置的关系（优先排查）

| 配置项 | smoke 当前值 | 可能影响 |
|--------|--------------|----------|
| `enable_chunked_prefill` | `True` | 与 DLLM staging 截断交互，长 prompt+response 更易多轮 |
| `max_num_batched_tokens` | `4096`（smoke） | `PrefillAdder.rem_dllm_tokens` 上界；不足时 staging 截断 |
| `mem_fraction_static` | `0.32`（SGLang smoke） | KV 池小 → `rem_total_tokens` 不足 → 多轮 |
| `prompt_tokens` + `max_new_tokens` | 442 + 512 | 总长 ~954，接近 budget 边界时易拆 2 轮 |

### 后续排查步骤（按顺序）

1. **确认多轮**（日志）
   - 在 `meta_info` 中除 `nfe` 外记录 `completion_tokens`、finish_reason。
   - 若 `len(nfe)>1`，在 rollout 日志打印 `nfe_rounds=len(nfe) nfe_total=sum(nfe)`（已在 `format_nfe_for_log` 中做）。

2. **对照实验：强制单轮全长**
   - smoke 中设 `actor_rollout_ref.rollout.enable_chunked_prefill=False`。
   - 提高 `max_num_batched_tokens`（如 8192）或 `mem_fraction_static`（在显存允许下）。
   - 期望：`nfe` 变为标量或单元素列表，单 sample 耗时下降。

3. **SGLang 侧验证**（离线）
   - 固定一条 `prompt_tokens≈442`、`max_new_tokens=512` 的输入，只跑 Engine generate。
   - 扫描 scheduler 日志 / 打印 `req.is_chunked`、`extend_input_len` 每轮变化。

4. **长期修复（可选）**
   - upstream：Dream + `pad_full_generation` 时合并多轮 `customized_info["nfe"]` 为 `sum` 或仅保留最后一轮。
   - VERL：对 `len(nfe)>1` 打 WARN，便于发现 budget 不足。

### 验收标准

- 同 prompt 长度下，`nfe` 为 **单标量**（或 `nfe_rounds=1`）。
- 单 sample rollout 耗时与 `nfe` 量级线性，无「列表长度=2 → 时间翻倍」。

---

## 问题 2：HumanEval pass@1 ≈ 0.17–0.19，低于 `finetune_d3LLM` 预期（~0.5）

### 现象

| 阶段 | pass@1 | 说明 |
|------|--------|------|
| 训练前（`val_generations/0.jsonl`） | **17.7%** (29/164) | base 权重，未经 RL |
| step1 后（`val_generations/1.jsonl`） | **19.5%** (32/164) | 仅 1 step smoke |
| 此前 3-step smoke step3 | ~19% | 与 base 接近，RL 步数过少 |

说明：**低分主要不是 3 步 RL 把模型训坏**，而是 **评测链路 / 解码路径** 与「HF 上测得的 ~0.5」不一致。

### 失败样本特征（`val_generations/3.jsonl` 统计）

| 类型 | 约占比（错题中） |
|------|------------------|
| 提取后 `SyntaxError` | ~53/133 |
| 提取残缺 / `bad_extract` | ~39/133 |
| 前言 + markdown（`The completed Python code...`） | 少量 |
| 仍有 ``` 代码块 | 140/164 |

典型错误：扩散噪声 token（如 `Mechanic.append`）、两行 `return` 粘在一起、docstring 复述占满上下文。

### 与「~0.5」差异的可能原因

1. **评测路径不是 HF multiblock**
   - 生产验证：`SGLang` + `validate=True` + `temperature=0.0` + `FullAttnMultiBlock`。
   - 论文/本地常见：`verify_finetune_d3llm.py` / HF `diffusion_generate` + `d3llm_multiblock`。
   - **温度、stop、block 阈值、是否多轮 staging 均可能不同。**

2. **后处理**
   - 奖励与 pass@1 使用 `extract_code_from_model`（第一个含 `def` 的 ``` 块）。
   - 前言 + 语法错误块仍会被执行 → 0 分。

3. **数据分布**
   - 训练：LCB 风格（长解释、`To solve this problem`）。
   - 验证：HumanEval 补全 stub。

4. **步数**
   - smoke 仅 1–3 step，不能代表 full BGPO 后的上限。

### 后续排查步骤（按顺序）

1. **跑对比脚本（必做）**  
   使用 `recipe/d3llm/benchmark_humaneval_decode_paths.py`（见下）在同一 parquet、同一提取器上对比：
   - **A. HF multiblock**（`verify_finetune_d3llm` 同配置）
   - **B. SGLang val 路径**（与 trainer 一致：`temperature=0`、`stop_token_ids`、finalize）
   - **C. SGLang train 路径**（`temperature=0.2`、per-sample seed）

   **若 A≈0.5 且 B≈0.19** → 根因在 SGLang/验证配置，而非权重。  
   **若 A≈0.19** → 权重或本地 `models/finetune_d3LLM` 与预期 checkpoint 不一致。  
   **若 B 对齐 A 后上升** → 逐项回退 val 与 HF 差异（温度、stop、max_new_tokens、chunked prefill）。

2. **对齐 val 与 train 后处理**
   - 验证路径已接 `_finalize_dream_response_tensor` + `stop_token_ids`（与 train 一致）。
   - 确认 `enable_chunked_prefill`、KV 预算不引入多轮 nfe（见问题 1）。

3. **扩大训练再评**
   - full 脚本 + `val_before_train` + 每 N step `test_freq`。
   - 不以 1-step smoke 判断上限。

4. **可选：HumanEval 专用提取**
   - 去前言、取最后一个合法 `def` 块（仅用于 debug 对比，改 reward 需与论文一致）。

### 验收标准

- 对比表：HF / SGLang-val / SGLang-train 三条 pass@1 差距 < 5pp，或明确文档化不可对齐项。
- 若 HF≈0.5：训练主路径以 SGLang 为准时，验证配置必须与 HF 对齐或证明等价。

---

## 相关文件

| 文件 | 用途 |
|------|------|
| `recipe/d3llm/benchmark_humaneval_decode_paths.py` | HumanEval 多路径 pass@1 对比 |
| `recipe/d3llm/compare_sglang_train_vs_val_path.py` | 单条 prompt train vs val 引擎对比 |
| `recipe/d3llm/verify_finetune_d3llm.py` | HF 离线 multiblock 冒烟 |
| `verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py` | 训练 rollout + stop/finalize |
| `third_party/sglang/.../full_attn_multi_block.py` | NFE 产生处 |
| `third_party/sglang/.../dllm/mixin/scheduler.py` | 多轮 scheduler / nfe append |

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05-27 | 初版：问题 1 多轮 nfe / 问题 2 HumanEval 基线；排查步骤与对比脚本 |
