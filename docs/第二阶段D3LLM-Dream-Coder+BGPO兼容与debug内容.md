# 自 `8984846` 以来的主要改动思路（d3LLM Dream-Coder × BGPO）

> 基线：`8984846`（2026-05-26，*modifying gitignore (before adding d3llm)*）  
> 当前：`39e4168`（2026-05-27，`final fix (start full bgpo)`）  
> 分支：`feat/d3llm-dream-clean`（自 SDAR 4 卡 SGLang 成果之上重做 d3LLM，**未**在旧 `feat/sglang-from-cb8ce6b` 的 d3LLM commit 链上继续堆叠）  
> 前置阶段 1 文档：[第一阶段SDAR+BGPO兼容与debug内容.md](./第一阶段SDAR+BGPO兼容与debug内容.md)

---

## 一句话

在 **不新增 `model.name=d3llm_dream`** 的前提下，把 **d3LLM Dream-Coder 的 multiblock 解码** 接到现有 **Dream + BGPO** 训练栈：先 **HF + FSDP 变长 NFE 可训**，再 **SGLang 独立 Engine 推理** 与 SDAR 4 卡模式对齐；并修 **双 nfe / val 路径 / HumanEval 评测一致性** 等阻断或误导指标的问题。

---

## 0. 与第一阶段的关系

| 维度 | 第一阶段（SDAR） | 第二阶段（d3LLM Dream-Coder） |
|------|------------------|------------------------------|
| 模型 | `model.name=sdar` | **`model.name=dream`**，权重 `models/finetune_d3LLM` |
| 解码 | SGLang `LowConfidence` + temperature 采样 | **multiblock / `FullAttnMultiBlock`**（entropy_threshold） |
| 默认 rollout | SGLang 内嵌 | 阶段 1：`hf`；阶段 3：**`sglang`** |
| 复用 | — | **`FSDPSGLangSDARShardingManager`**、`SGLangRollout` 基类、4 卡 Ray/`unset PYTORCH_CUDA_ALLOC_CONF` 等 |
| 丢弃 | — | 旧分支 `2738878…9f9ff07` 整段（`d3llm_dream` 命名、`D3LLM_ROOT` runtime、`sglang_d3llm_dream_rollout.py` 等） |

**思路**：第一阶段解决「4 卡 + SGLang + GRPO 要有随机性」；第二阶段解决「Dream-Coder 变长 diffusion 步数 + 同一套 BGPO 能训、能评、能和 HF 基线对照」。

---

## 1. 设计原则（干净重做）

来源：**自写**规划文档 `docs/d3llm-dream-clean-plan.md`（commit `376e041` 起）。

1. **`model.name` 始终为 `dream`**，用 `rollout.dllm_decode=multiblock` 区分 vanilla / d3LLM，不引入 `d3llm_dream`。
2. **不在 `dllm_main_ppo.py` 注入 `D3LLM_ROOT`**；训练路径 vendored 生成逻辑，离线验证才用 `recipe/d3llm/d3llm_multiblock.py` + 环境变量。
3. **SGLang 复用 SDAR 的 sharding / Engine 生命周期**，单独写 `SGLangDreamRollout`，不 fork 一整份 SDAR 训练脚本逻辑到 `recipe/d3llm/`。
4. **分阶段验收**：0 离线加载 → 1 HF smoke → 3A–3E SGLang → 再 HumanEval / 全量 BGPO。

---

## 2. 时间线（按 commit 意图）

```
8984846  SDAR SGLang 4 卡成果 + gitignore（d3LLM 接入前）
    ↓
376e041  阶段 0：离线 verify + setup 脚本 + clean-plan 文档
    ↓
d1d2b7a  阶段 1：HF multiblock 接入 verl（dream_multiblock + vendored util）
    ↓
d63a3b1  rollout debug、parquet fallback、global_step meta
    ↓
73790be  HF FSDP 变长 NFE：fsdp_rollout_inference_context + multiblock 同步
    ↓
a59ff34  third_party/sglang → PR #20615（Dream + FullAttnMultiBlock）
    ↓
cce889c  3A 文档 + verify_sglang_dream_rollout
0d1fc34  3B verify_sglang_engine_smoke + 并行脚本
c3d850d  3C SGLangDreamRollout + dllm_fsdp_workers 路由
2f2e3b7  3D run_bgpo_dream_coder_d3llm.sh SGLang 参数
    ↓
830ffd1…9ace7ca  rollout debug、smoke、日志
bf0c578  修复 double rollout（val n 重复）
ca1cfdf  val 走 per-sample 路径（对齐 train）
d16c0a6  validation 逻辑收尾
39e4168  W&B / debug / 全量 BGPO 启动前最后一轮脚本修
```

---

## 3. 阶段 0 — 权重与离线可运行（仅 `recipe/d3llm`，不进 verl 训练热路径）

| 文件 | 来源 | 原因与内容 |
|------|------|------------|
| `recipe/d3llm/setup_finetune_d3llm_model_code.sh` | **自写**（参考 d3LLM 仓库布局） | `finetune_d3LLM` 目录需自带 `configuration_dream.py` / `modeling_dream.py` / `generation_utils.py` 与 `auto_map`，否则 HF 无法 `trust_remote_code` 加载全量 7.6B。 |
| `recipe/d3llm/verify_finetune_d3llm.py` | **自写** | `load` / `vanilla`（entropy）/ `multiblock`（entropy_threshold）冒烟，确认权重与解码语义再改 verl。 |
| `recipe/d3llm/d3llm_multiblock.py` | **自写**（逻辑对齐 d3LLM `run_code_eval.sh`） | 通过 `D3LLM_ROOT` **绑定**官方 `d3llm_dream_generate_util`，仅用于离线；训练不 import 此文件。 |
| `recipe/d3llm/README.md` | **自写** | 阶段 0 使用说明。 |
| `docs/d3llm-dream-clean-plan.md` | **自写** | 旧实现删除/合并清单 + 分阶段文件边界。 |

---

## 4. 阶段 1 — HF multiblock + BGPO 闭环

**目标**：在 **同一 FSDP actor** 上做 multiblock rollout + BGPO update，先证明算法闭环，再换 SGLang。

### 4.1 解码与生成核心

| 文件 | 来源 | 原因与内容 |
|------|------|------------|
| `verl/workers/rollout/d3llm_dream_generate_util.py` | **从 d3LLM 官方 vendored**（~1400 行） | 实现 `DreamGenerationMixin` / `entropy_threshold` multiblock；避免训练依赖外部 repo 路径。 |
| `verl/workers/rollout/dream_multiblock.py` | **自写封装**（参考 `recipe/d3llm/d3llm_multiblock.py` + d3LLM eval 默认超参） | `execute_dream_multiblock_generation`：左 padding 剥离 → 生成 → 恢复布局；`block_length=32`、`threshold=0.5`、`cache_delay_iter=32`、`early_stop=True`。 |
| `verl/workers/rollout/rollout_utils.py` | **小改自写** | `execute_fastdream_generation` 在 `dllm_decode=="multiblock"` 时路由到 `dream_multiblock`。 |
| `verl/workers/rollout/fast_dream_rollout.py` | **改现有 Dream rollout** | 读取 `dllm_decode`、`d3llm_*` 阈值、`per_sample_seed`；`fsdp_rollout_inference_context` 包裹 forward。 |
| `recipe/dream/run_bgpo_dream_coder_d3llm.sh` | **自写**（fork `run_bgpo_dream_7b_instruct.sh` + 借鉴 SDAR 4 卡脚本） | `task=code`、`model.path=finetune_d3LLM`、`dllm_decode=multiblock`；`--smoke` / `--engine hf\|sglang`。 |

### 4.2 FSDP 变长 NFE（阶段 1 最关键的正确性改动）

| 文件 | 来源 | 原因与内容 |
|------|------|------------|
| `verl/utils/fsdp_utils.py` | **自写**（模式 **参考** `hf_rollout` 的 summon/unshard） | 新增 `fsdp_rollout_inference_context`：multiblock 每 rank **NFE 不同**，不能在共享 FSDP 上异步 forward，否则 collective 死锁；快 rank `barrier` 等慢 rank 再 reshard。 |
| `verl/workers/rollout/d3llm_dream_generate_util.py` | **自写补丁**（在 vendored 代码上） | `set_fsdp_rollout_sync` / 必要时 dummy forward，与 `dream_multiblock` 配合（commit `73790be`）。 |
| `recipe/d3llm/test_fsdp_rollout_unshard.py` | **自写** | 单测/诊断 unshard 路径。 |
| `recipe/d3llm/benchmark_long_prompt.py`、`benchmark_multiblock_padding.py` | **自写** | 长 prompt / padding 行为排查。 |

**思路**：HF 路径的瓶颈不是「算得慢」 alone，而是 **变长 diffusion 与 FSDP 集体通信不兼容**；先修同步语义，再考虑 Engine 外置。

### 4.3 小修复（cherry-pick 风格，与 d3LLM 算法正交）

| 文件 | 来源 | 内容 |
|------|------|------|
| `verl/models/transformers/dream.py` | **小改自写** | 无 `DreamFlashAttention` 时跳过 Ulysses flash patch（Dream-Coder 仅 `DreamSdpaAttention`）。 |
| `verl/utils/dataset/rl_dataset.py` | **自写** | parquet 缓存含废弃 `List` feature 时 **pandas fallback**。 |
| `verl/trainer/ppo/dllm_ray_trainer.py` | **自写** | `gen_batch.meta_info["global_step"]` 供 rollout debug。 |
| `verl/workers/rollout/dream_rollout_debug.py` | **自写** | rank 级 `rollout_debug/rank*.rollout.log`、train `prompt_group` 汇总。 |

---

## 5. 阶段 3 — SGLang 独立推理（对齐 SDAR 4 卡架构）

规划：**自写** `docs/d3llm-dream-sglang-plan.md`。

### 5.1 上游与子模块

| 项 | 来源 | 原因与内容 |
|----|------|------------|
| `third_party/sglang` @ `c795ddb` | **拉取** [sglang PR #20615](https://github.com/sgl-project/sglang/pull/20615)（`feat/dllm-llada-dream-support`） | 上游提供 `DreamModel`、`FullAttnMultiBlock` dLLM 算法；commit `a59ff34`。 |
| `third_party/sglang/.../schedule_policy.py` | **自写补丁**（在子模块内） | `pad_full_generation=True` 时 **不再**对全长做 `// page_size * page_size` 截断，避免 **一轮请求拆成两轮 staging → 双 nfe、耗时约 2×**（见 `docs/d3llm-dream-open-issues.md`）。 |

### 5.2 VERL 集成

| 文件 | 来源 | 原因与内容 |
|------|------|------------|
| `verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py` | **自写**（**参考** `sglang_sdar_rollout.py` + `SGLangRollout`） | `SGLangDreamRollout`：`dllm_algorithm=FullAttnMultiBlock`、YAML `dllm_algorithm_config`（threshold/block_size/temperature/early_stop 等）；`chunked_prefill_size=-1`；`memory_saver` + `release_memory_occupation`。 |
| `verl/workers/dllm_fsdp_workers.py` | **改现有** | `rollout.name=sglang` 且 `model.name=dream` 且 `dllm_decode=multiblock` → `SGLangDreamRollout` + **`FSDPSGLangSDARShardingManager`（复用 SDAR，未新写 sharding）**。 |
| `verl/workers/rollout/sglang_rollout/__init__.py` | **小改** | 导出 `SGLangDreamRollout`。 |
| `recipe/dream/run_bgpo_dream_coder_d3llm.sh` | **自写扩写** | SGLang 分支：`unset PYTORCH_CUDA_ALLOC_CONF`；`mem_fraction_static` / `torch_native` or `fa3`；`enable_chunked_prefill=False`；`actor_rollout_ref.rollout.dllm_algorithm=FullAttnMultiBlock` 等。 |

### 5.3 采样与路径对齐（算法信号 + 评测）

| 改动 | 来源 | 原因 |
|------|------|------|
| `_per_sample_generate_sequences` | **自写**（**参考** SDAR `_lmdeploy_style_batch_level_generate_sequences`） | 每样本独立 `async_generate` + `sampling_seed`；避免 batch `n>1` 绑死随机性。 |
| val 与 train 均走 per-sample | **自写**（`ca1cfdf`、`d16c0a6`） | 原先 val 走父类 `_batch_level_generate_sequences` → 与 train 解码路径不一致，HumanEval 被系统性压低。 |
| val 不乘 `n_rollout` | **自写**（`bf0c578`） | dataloader 已按 `val_kwargs.n` repeat，再乘 train `n` → **double rollout**。 |
| `_finalize_dream_response_tensor` / stop tokens | **自写** | 对齐 SGLang stop（pad/eos/im_end/mask）；处理 `finish_reason=length` 时 `_opening_prefix_ids` 误截断。 |
| `val_temperature=0.2`、`val_do_sample=True` | **脚本自写** | 与 train / HF smoke 一致；greedy val 曾 ~20% 而 HF multiblock 离线 ~58%（见 `docs/d3llm-dream-humaneval-quality.md`）。 |

### 5.4 离线验证脚本（recipe，不进训练）

| 文件 | 来源 | 作用 |
|------|------|------|
| `recipe/d3llm/verify_sglang_engine_smoke.py` | **自写** | 3A：单卡 Engine 起服 + 一条生成。 |
| `recipe/d3llm/run_verify_sglang_parallel.sh` | **自写** | 3B：多卡并行 HF vs SGLang 对齐。 |
| `recipe/d3llm/verify_sglang_dream_rollout.py` | **自写** | HF multiblock vs SGLang token 对比（精简版）。 |
| `recipe/d3llm/benchmark_humaneval_pipelines.py` | **自写** | HumanEval 多路径 pass@1 对照（HF / SGLang train vs val）。 |
| `recipe/d3llm/compare_sglang_train_vs_val_path.py` | **自写** | 同 prompt 下 train/val 路径 diff。 |
| `recipe/d3llm/verify_repetition_token_provenance.py` | **自写** | 重复 token / 截断溯源。 |
| `recipe/d3llm/run_sglang_val_humaneval.sh` | **自写** | 单独跑 val HumanEval 管线。 |

---

## 6. Debug 与文档产出（问题驱动）

| 文档 | 对应问题 |
|------|----------|
| `docs/d3llm-dream-open-issues.md` | 双 nfe、HumanEval 基线偏低、推荐排查工具 |
| `docs/d3llm-dream-humaneval-quality.md` | HF ~58% vs SGLang val ~20% 交叉表、失败 taxonomy |
| `docs/d3llm-dream-sglang-plan.md` | 3A–3E 架构与 Hydra 映射 |

**已确认非训练栈 bug、但影响指标解读**：

- 日志中大量 `Error in code execution` / `SyntaxError`：**判分失败**，RL 正常。
- `generation flags are not valid: ['temperature']`：HF **加载** actor 时提示，SGLang 请求级 temperature 仍由 `dllm_algorithm_config` / `sampling_params` 控制。
- `Exception ignored in atexit`（`TemporaryDirectory`）：Ray 退出噪音。

**仍开放（质量，非 crash）**：SGLang `FullAttnMultiBlock` 与 HF `dream_multiblock` 文本级差异、未闭合 markdown 围栏、`max_response_length=512` 截断等（详见 humaneval-quality 文档）。

---

## 7. 来源归类总表

| 类别 | 代表内容 |
|------|----------|
| **自写（DARE）** | `dream_multiblock.py`、`sglang_dream_rollout.py`、`dream_rollout_debug.py`、`fsdp_rollout_inference_context`、`run_bgpo_dream_coder_d3llm.sh`、全部 `recipe/d3llm/verify_*` 与 benchmark、`schedule_policy` 双 nfe 补丁、规划/问题文档 |
| **从 d3LLM 官方 vendored** | `d3llm_dream_generate_util.py`（训练路径）；离线 `d3llm_multiblock.py` 绑定官方模块 |
| **参考/复用现有 DARE（SDAR 阶段）** | `FSDPSGLangSDARShardingManager`、`SGLangRollout` 基类、4 卡 Ray 与 `unset PYTORCH_CUDA_ALLOC_CONF`、`fast_dream_rollout` / `dllm_fsdp_workers` 路由模式、`run_bgpo_sdar_8b_chat.sh` 显存策略 |
| **从上游 pull（子模块）** | `third_party/sglang` PR #20615：`DreamModel`、`FullAttnMultiBlock`、Dream 相关 server/scheduler 支持 |
| **参考现有 Dream 代码** | `run_bgpo_dream_7b_instruct.sh`、`fast_dream_rollout` / `rollout_utils` entropy 路径 |
| **明确丢弃（旧分支）** | `model.name=d3llm_dream`、`D3LLM_ROOT` in `dllm_main_ppo`、`sglang_d3llm_dream_rollout.py`、`fast_d3llm_dream_rollout.py`、`fsdp_hf_rollout.py` 等 |

---

## 8. Hydra / 脚本关键映射（便于对照）

| 配置项 | Vanilla Dream 7B | d3LLM Dream-Coder（本阶段） |
|--------|------------------|----------------------------|
| `actor_rollout_ref.model.name` | `dream` | `dream` |
| `actor_rollout_ref.model.path` | `Dream-v0-Instruct-7B` | `models/finetune_d3LLM` |
| `actor_rollout_ref.rollout.dllm_decode` | `entropy` | **`multiblock`** |
| `actor_rollout_ref.rollout.name` | `hf` | **`hf`（阶段 1）或 `sglang`（阶段 3）** |
| `actor_rollout_ref.rollout.dllm_algorithm` | — | **`FullAttnMultiBlock`**（SGLang） |
| `actor_rollout_ref.rollout.block_length` | 32 | 32 |
| `actor_rollout_ref.rollout.d3llm_threshold` | — | 0.5（与 eval 脚本一致） |
| BGPO actor | `dream_dp_actor_bgpo` | **不变** |

---

## 9. 若只记住三件事

1. **命名与依赖**：只用 **`dream` + `dllm_decode=multiblock`**，生成核心 **vendored d3LLM**，不靠 `D3LLM_ROOT` 进训练。  
2. **两阶段推理**：HF 先解决 **FSDP 变长 NFE 死锁**；SGLang 再解决 **推理与训练抢卡 + 吞吐**，并 **对齐 per-sample 采样与 val 路径**。  
3. **指标解读**：HumanEval 低分优先查 **评测/解码链路**（SGLang val vs HF multiblock），不是先怀疑 1-step BGPO 训坏权重；双 nfe 是 **调度截断 bug**，已在 `schedule_policy` + rollout 参数修掉。

---

## 10. 相关命令（复现入口）

```bash
# 阶段 0
bash recipe/d3llm/setup_finetune_d3llm_model_code.sh
python recipe/d3llm/verify_finetune_d3llm.py --mode all --max-new-tokens 128

# 阶段 1 HF smoke
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine hf

# 阶段 3 SGLang smoke / 全量
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine sglang
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --engine sglang
```

更细的 open issue 与 HumanEval 分析见同目录 `d3llm-dream-*.md`。

---

## 11. 第一次全量 BGPO 训练复盘（效果不佳）

> 日志目录：`logs/DARE/1st_try_dream-code-d3llm-bgpo-sglang-bsz8-n4-prompt1024-response512-bl32-lr5e-7-temp0.2-gpu4-20260527_164839/`  
> 启动时间：2026-05-27 16:48；约训练至 step 380+（2 epoch 未完成即停止分析）。

### 11.1 运行配置（与当前脚本差异大）

| 项 | 第一次全量 | 当前推荐（2026-05-29 后） |
|----|-----------|---------------------------|
| 训练数据 | `lcbv5-K8_1` + `primeintellect-K8_1` + `taco-K8_1`（1816 行，**非 EvalPlus prompt 格式**） | `code_evalplus_mix_1.parquet`（811 行，EvalPlus 续写格式） |
| 验证数据 | `humaneval_1.parquet`（**非 EvalPlus+**） | `humaneval_evalplus_1.parquet` |
| `batch_size × n_rollout` | 8 × 8 = 64 | 8 × 8（全量）/ smoke 4 × 4 |
| `max_response_length` | 512 | 1024（全量/smoke 均已拉长） |
| `use_kl_loss` | **False** | 待试 True（coef 0.005–0.01） |
| TPF 效率奖励 | 无 | 有（见 `docs/d3llm-dream-tpf-reward与联合监控.md`） |
| 代码提取 | 旧逻辑（EvalPlus 续写易 `pred=""`） | 已修 `extract_code_from_model` |

引擎：SGLang + `FullAttnMultiBlock`；`train_temperature=0.2`，`val_temperature=0.2`，`val_do_sample=True`；`lr=5e-7`；`test_freq=20`。

### 11.2 HumanEval `val-core/acc/mean@1` 曲线

| Step | Acc | 备注 |
|------|-----|------|
| 0 | **59.1%** | 训练前基线 |
| 40 | 64.0% | 前期缓升 |
| 60 | 65.2% | |
| **120** | **67.1%** | **峰值** |
| 140–200 | 62–65% | 平台震荡 |
| 240 | 59.8% | 开始回落 |
| 280 | 60.4% | |
| 300–340 | **54.9%** | 明显下跌 |
| 360 | 57.3% | 略反弹 |
| **380** | **53.0%** | 低于 step 0 |

**结论**：acc 先升后跌，step 120 后长期低于峰值，step 300+ 跌破训练前基线；不能视为有效收敛。

### 11.3 问题归因（按优先级）

1. **训练/验证任务错配**  
   训练集为旧 taco/lcbv5/primeintellect 填空/补全风格，验证为 HumanEval；模型在训练分布上学到的格式与评测不一致，泛化上限受限。

2. **评测集与提取 bug**  
   使用 `humaneval_1` 而非 EvalPlus+；且当时 `extract_code_from_model` 对「prompt 已含开围栏、response 仅尾部闭围栏」的 EvalPlus 续写提取失败，部分样本 `pred=""` 被判 0 分（smoke 修复后 step0 25%→100% 即证）。

3. **无 KL 锚定 + BGPO 随机 mask**  
   `use_kl_loss=False`，ELBO 与 multiblock 块解码目标不对齐，易出现「acc 略升但解码置信度/TPF 下降」的退化（见 TPF 文档）。

4. **仅优化 pass/fail 二元奖励**  
   未约束 tokens-per-forward（TPF），模型可能用更多 NFE 换略高的 pass rate，实际推理效率变差。

5. **非 infra 问题**  
   双 nfe、val double rollout、FSDP 死锁等已在阶段 1–3 修掉；本次 run 能稳定跑完数百 step，瓶颈在 **数据 + 奖励 + 算法信号**，非 crash。

### 11.4 后续改进方向（已部分落地）

- [x] EvalPlus 混训数据 `code_evalplus_mix_1.parquet` + `humaneval_evalplus_*` 验证集  
- [x] 代码提取修复（`code_reward.py`）  
- [x] TPF 效率塑形 + W&B 联合监控 `pass_reward` / `reward` / `tpf`  
- [ ] 尝试 `use_kl_loss=True` + 更低 val/train temperature（0.0）  
- [ ] 全量重训并对比 acc **与** TPF 曲线  

详细设计与实现见：**[d3llm-dream-tpf-reward与联合监控.md](./d3llm-dream-tpf-reward与联合监控.md)**。
