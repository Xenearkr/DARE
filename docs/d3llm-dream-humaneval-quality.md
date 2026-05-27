# d3LLM Dream-Coder：HumanEval 质量偏低根因分析

> 最后更新：2026-05-27  
> 权重：`models/finetune_d3LLM`  
> 关联 smoke：`dream-code-d3llm-bgpo-sglang-smoke-...-20260527_{112919,130203}`  
> 总览：[d3llm-dream-open-issues.md](./d3llm-dream-open-issues.md)

---

## 结论（Executive Summary）

| 判断 | 说明 |
|------|------|
| **权重未“训坏”** | 训前 `val_generations/0.jsonl` 与训后 `1.jsonl` 仅差 ~2pp；同一权重在 **HF multiblock 离线** 上 pass@1 ≈ **58%**。 |
| **主因是评测/解码链路** | 训练验证走 **SGLang `FullAttnMultiBlock` + val 批处理路径**；HF 基线走 **`dream_multiblock` / `D3LLMMultiBlock`**。二者在 **70/95 道 HF 已通过题** 上仍失败，说明差距主要来自 **解码实现差异**，而非 HumanEval 任务本身学不会。 |
| **次要但可修的因素** | 未闭合 markdown 围栏导致提取失败、`_opening_prefix_ids` 误截断、`max_response_length=512` 截断、val `temperature=0` 与 HF/训练 `0.2` 不一致、train **逐条** vs val **批处理** 两条 SGLang 路径。 |
| **nfe 双轮问题** | 已在 `130203` smoke 中修复（单轮 nfe、耗时恢复正常）；**不解释** HumanEval 与 HF 的 ~40pp 差距。 |

**当前 smoke 配置**：`save_freq=0`（不保存 checkpoint），见 `recipe/dream/run_bgpo_dream_coder_d3llm.sh`。

---

## 1. 数据与复现来源

### 1.1 pass@1 对照

| 来源 | pass@1 | 配置要点 |
|------|--------|----------|
| SGLang val，`112919` / `0.jsonl`（训前） | **29/164 = 17.7%** | `val_kwargs.temperature=0`，`max_response_length=512` |
| SGLang val，`130203` / `0.jsonl`（nfe 修复后） | **36/164 = 22.0%** | 同上 |
| SGLang val，`112919` / `1.jsonl`（1 step 后） | **32/164 = 19.5%** | 1-step BGPO 几乎不动基线 |
| HF multiblock 单卡离线 | **95/164 = 57.9%** | `logs/benchmarks/humaneval_decode_paths_20260527.json`，`train_temperature=0.2`，`threshold=0.5` |

离线三路脚本 `benchmark_humaneval_decode_paths.py` 中 **SGLang 路径未跑完**（单卡 `mem_fraction_static=0.32` OOM），JSON 里仅有 `hf_multiblock` 完整结果；SGLang 全量对比仍以 **Ray smoke 的 `val_generations/*.jsonl`** 为准。

### 1.2 与 HF 的逐题交叉表（`112919` val0）

| 组合 | 题数 |
|------|------|
| HF ✓，SGLang val ✗ | **70** |
| HF ✓，SGLang val ✓ | 25 |
| HF ✗，SGLang val ✓ | 4 |
| HF ✗，SGLang val ✗ | 65 |

**70 题**在相同权重、相同 parquet 下 HF 已过、SGLang val 未过——这是“链路不一致”的直接证据。

对这 70 题的粗分类（SGLang 侧失败形态）：

| 类型 | 题数 | 含义 |
|------|------|------|
| 提取后仍 `SyntaxError` | 25 | 与 HF 生成文本不同（截断、缺 import、坏字符串等） |
| 未闭合 ` ``` ` 围栏 | 13 | `extract_code_from_model` 返回 `None` → `pred=""` |
| 无 markdown 围栏 | 10 | 同上或整段非代码 |
| 语法可解析但测例失败 | 13 | 解码结果与 HF 不同逻辑 |
| 输出很长仍失败 | 9 | 冗长/重复/部分截断 |

---

## 2. 失败样本 taxonomy（SGLang val，`112919` / `0.jsonl`）

对 **135 道未通过** 题的分类（与 reward 中 `extract_code_from_model` + `humaneval_check_correctness` 一致）：

| 类别 | 题数 | 说明 |
|------|------|------|
| `syntax_error:SyntaxError` | 34 | 提取出的代码无法执行 |
| `syntax_ok_but_test_fail` | 30 | 能 parse，但 assert 不通过 |
| `no_codeblock_in_output` | 28 | 模型未按 ``` 格式输出 |
| `extract_empty_but_has_fence` | 25 | 有围栏但正则未匹配（常见：**未闭合围栏**） |
| `code_without_leading_fence` | 9 | 非标准围栏形态 |
| `syntax_error:IndentationError` | 8 | |
| `lcb_style_preamble` | 1 | 输出含 LCB 风格前言 |

### 2.1 输出长度与截断

- 失败样本中 **57** 道 `len(output) > 480`（接近 `max_response_length=512`）。
- 全量 **29** 道 `unclosed_fence`（开闭 ` ``` ` 数量为奇数）。
- 失败里约 **26** 道属于「截断在 code block 中间」。

典型样例（`0.jsonl` 第 0 行）：`output` 在 `for j in range(i + 1, len` 处被截断，无闭合 ` ``` ` → `pred=""` → 0 分；而 HF 同题通过且输出完整函数。

### 2.2 提取逻辑

```21:47:verl/utils/reward_score/code_reward.py
def extract_code_from_model(model_response: str):
    code_blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    ...
```

- **必须**有闭合的 ` ``` `；未闭合则整题 0 分（约 **25+13** 道与围栏相关）。
- HumanEval 评测只用 **提取后的 completion**，不把 parquet 里的 `prompt` 拼进执行（与 OpenCompass「prompt + completion」习惯不同，但 train/HF/SGLang 三条路一致，不是 58% vs 19% 的主因）。

### 2.3 评测执行

`humaneval_check_correctness` 在提取代码上跑 `ground_truth` 里的 `check(candidate)`；逻辑错、缺行、错误分支（如 `elif` 写成 `if`）归入 `syntax_ok_but_test_fail` 或运行期 assert 失败。

---

## 3. 根因分解（按优先级）

### 3.1 【P0】SGLang 与 HF multiblock 解码不一致（~40pp）

**现象**：同权重 HF 58%，SGLang val 18–22%；70 题 HF 过、SGLang 不过。

**可能机制**（需 `compare_sglang_train_vs_val_path.py` 逐题对齐验证）：

1. **引擎实现**：SGLang `FullAttnMultiBlock` vs `verl.workers.rollout.dream_multiblock` / `recipe/d3llm/d3llm_multiblock.py` 在 mask 调度、early stop、confidence threshold、NFE 停止条件上不一致。
2. **温度来源分裂**：
   - Engine 启动时 `dllm_algorithm_config` 写入 **`temperature: rollout.temperature`（训练 0.2）**（`sglang_dream_rollout.py`）。
   - Val 批处理另设 `val_kwargs.temperature=0`（`sglang_rollout.py` 的 `update_sampling_params`），**是否传入 DLLM 算法需实测**；若仍用 YAML 的 0.2，则文档写的 “val temperature=0” 与真实解码不符。
3. **Train vs Val 路径分裂**（同文件）：
   - **Train rollout**：`do_sample=True` → `_per_sample_generate_sequences`（逐条 `async_generate`，`temperature=0.2`，独立 seed）。
   - **Val**：`validate=True` → `super()._batch_level_generate_sequences`（批处理 + `val_kwargs`），再 `_apply_dream_response_finalize`。
   - 两条路径的 `finish_reason`、批大小、stop 处理可能不同。

### 3.2 【P1】后处理误伤：`_opening_prefix_ids`

```292:293:verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py
    def _opening_prefix_ids(self) -> List[int]:
        return self.tokenizer.encode("To solve this problem", add_special_tokens=False)
```

`_finalize_dream_response_tensor` 若在 response 中 **第二次** 出现该子串，会 **提前截断**（为 LCB 长前言设计）。HumanEval 偶发类似措辞或 token 碰撞时，会把合法代码截掉 → 未闭合围栏或语法残缺。

**建议**：HumanEval / `data_source=humaneval` 时禁用该截断，或改为仅匹配完整 chat 回合边界。

### 3.3 【P1】长度与 early stop

- `max_response_length=512`：部分题 HF 生成更长或更紧凑；SGLang 在 `finish_reason=length` 时填满 512 token，再经 stop 截断，易截在 code block 中间。
- Dream early stop（pad/mask/eos）在 SGLang 与 HF 的触发时机可能不同，导致 **有效解码长度** 不同。

### 3.4 【P2】格式与分布

- **训练**：`lcbv5-K8_1.parquet`（LiveCodeBench 风格）；**验证**：`humaneval_1.parquet`（补全 + ```python）。
- 模型在 LCB 上学会长前言、`To solve...`；HumanEval 期望短补全。smoke 中仅 **1** 道明显 LCB 前言，不是 70 题差距主因，但会影响少数题。

### 3.5 【排除】非主因

| 假设 | 结论 |
|------|------|
| 1-step BGPO 训坏 | 0.jsonl vs 1.jsonl 几乎不变 |
| 权重损坏 | HF 同路径 58% |
| nfe 双轮 | 修后 HumanEval 仅 112919→130203 +4pp，说明不是主因 |
| reward 未拿到 `data_source` | 已修；且 `pred` 非空时仍能测，问题在生成质量 |

---

## 4. 典型样例

### 4.1 截断 + 提取失败（HF ✓，SGLang ✗）

- **Row 0**：SGLang `output` 止于 `...for j in range(i + 1, len`，无闭合 fence，`pred=""`；HF multiblock 完整 `def has_close_elements...` 并通过。

### 4.2 提取成功但逻辑错（两边解码不同）

- **Row 1**：`separate_paren_groups` 中 `elif char == '('` 后误用 `if char == ')'`，测例失败；HF 同题通过（解码文本不同）。

### 4.3 提取成功且短代码（真过）

- **Row 2**：`truncate_number` 三行 + 闭合 fence，`acc=1`。

---

## 5. 配置清单（对齐检查表）

| 项 | Train rollout | Val（当前 smoke） | HF offline benchmark |
|----|---------------|-------------------|----------------------|
| 解码栈 | SGLang FullAttnMultiBlock | 同上（批处理） | `dream_multiblock` / D3LLM |
| temperature（sampling_params） | 0.2 | **0.0** (`val_kwargs`) | N/A（HF 侧 multiblock cfg **0.2**） |
| temperature（dllm yaml） | 0.2（Engine 初始化） | 同上（**可能覆盖 val_kwargs**） | 0.2 |
| `max_new_tokens` / response | 512 | 512 | 512 |
| `threshold` / block | 0.5 / 32 | 同左 | 0.5 / 32 |
| stop | pad, eos, im_end, mask | 同左 + finalize | HF early stop on EOS |
| 路径 | per-sample async | batch async | 单进程 HF |
| opening_prefix 截断 | 有 | 有（val finalize） | **无** |

---

## 6. 建议的验证与修复顺序

1. **逐题 diff（P0）**  
   在 DARE conda、与训练相同 Engine 参数下：  
   `recipe/d3llm/compare_sglang_train_vs_val_path.py --humaneval --row N`  
   对比 decode 文本、nfe、`finish_reason`、是否触发 opening_prefix 截断。

2. **统一温度（P0）**  
   - 验证 val 时 DLLM 实际使用的 temperature；  
   - 试验 val `temperature=0.2` 与 HF 对齐，或 HF 也跑 `temperature=0` 做对照。

3. **Val 走 train 同路径（P0）**  
   考虑 val 也走 `_per_sample_generate_sequences`（仅 `n=1`、`temperature=val`），消除 batch vs per-sample 差异。

4. **禁用或按 data_source 关闭 opening_prefix 截断（P1）**

5. **提取器容错（P1）**  
   对未闭合 fence 取第一个 ` ```python\n` 至 EOS 的启发式（需评估是否误提取）。

6. **全量对齐实验（P0 验收）**  
   在 **4 卡 Ray** 上对 164 题同时 dump HF multiblock 与 SGLang val 文本，目标差距 &lt;5pp 或文档化不可对齐项。

7. **勿依赖已删单卡 benchmark 脚本**；若重做全量对比，须在 **与 smoke 相同的 Ray + mem_fraction** 下跑 SGLang。

---

## 7. 附录：命令与路径

```bash
# smoke（不存 ckpt：save_freq=0）
conda activate DARE
./recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine sglang

# 单题 train vs val
python recipe/d3llm/compare_sglang_train_vs_val_path.py --humaneval --row 0

# HF 权重冒烟
python recipe/d3llm/verify_finetune_d3llm.py --mode multiblock
```

| 路径 | 内容 |
|------|------|
| `data/preprocessed/rl/test/humaneval_1.parquet` | 164 题，`data_source=humaneval` |
| `logs/.../val_generations/0.jsonl` | SGLang 验证 dump |
| `logs/benchmarks/humaneval_decode_paths_20260527.json` | HF 164 题明细 |
| `verl/utils/reward_score/code_reward.py` | 提取 + humaneval 测例 |

---

## 变更记录

| 日期 | 说明 |
|------|------|
| 2026-05-27 | 初版：taxonomy、HF vs SGLang 交叉表、根因与修复顺序 |
| 2026-05-27 | smoke `save_freq=0` |
