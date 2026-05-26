# d3LLM Dream-Coder × BGPO 干净重做计划

> 基线分支：`feat/d3llm-dream-clean` @ `8984846`（*modifying gitignore (before adding d3llm)*）  
> 旧实现分支：`feat/sglang-from-cb8ce6b`（d3llm 相关 commit：`2738878` … `9f9ff07`）  
> 旧 WIP 已 stash：`wip d3llm before feat/d3llm-dream-clean branch`

---

## 1. Git 基线选择

| Commit | 说明 | 是否选用 |
|--------|------|----------|
| `cb8ce6b` | 仅 env scripts，**无** SDAR SGLang 4 卡适配 | 否 |
| `80053f9` … `9d401a0` | SDAR + SGLang 4×A6000 打通（LowConfidence 采样修复等） | 已包含在 8984846 祖先链中 |
| **`8984846`** | 明确标注 *before adding d3llm*；保留 SDAR SGLang 成果 | **选用** |
| `2738878` … `9f9ff07` | d3LLM 兼容（step 0 → sglang fix → verify） | **整段丢弃** |

当前已在 `feat/d3llm-dream-clean`；`third_party/sglang` 子模块已 reset 到 `bbe9c7e` 并删除未跟踪的 Dream/d3LLM 补丁文件。

恢复旧 WIP（仅供参考，不建议直接 merge）：

```bash
git stash list   # 找 wip d3llm before ...
git stash show -p stash@{0} | less
```

---

## 2. 旧实现：删除 / 合并清单

### 2.1 整文件删除（旧分支新增，干净基线已无）

| 路径 | 原因 |
|------|------|
| `verl/workers/rollout/fast_d3llm_dream_rollout.py` | 与 `fast_dream_rollout` 重复，仅解码不同 |
| `verl/workers/rollout/sglang_rollout/sglang_d3llm_dream_rollout.py` | 应合并为通用 Dream SGLang（阶段 3） |
| `verl/workers/rollout/d3llm_multiblock_bind.py` | 与 `recipe/d3llm/d3llm_multiblock.py` 双份实现 |
| `verl/workers/rollout/d3llm_rollout_debug.py` | debug 侵入 verl 核心；改放 recipe 脚本 |
| `verl/workers/sharding_manager/fsdp_hf_rollout.py` | 仅为 d3llm HF 路径新增；Dream 原用 `BaseShardingManager` |
| `recipe/d3llm/run_bgpo_d3llm_dream_coder.sh` | 过度 fork SDAR 脚本；改从 `recipe/dream` 派生 |
| `recipe/d3llm/run_humaneval_benchmark_parallel.sh` | 依赖 SGLang 验证栈，后置 |
| `recipe/d3llm/verify_sglang_dream_rollout.py` | 700+ 行，阶段 3 再写精简版 |
| `recipe/d3llm/diagnose_rank0_hang.py` | 临时诊断 |
| `third_party/sglang/.../dream_hf_kv.py` 等 5 个未跟踪文件 | SGLang Dream 补丁，阶段 3 再以最小 diff 引入 |

### 2.2 合并回 Dream 主路径（勿单独维护）

| 旧文件 | 合并目标 |
|--------|----------|
| `execute_d3llm_dream_generation()` in `rollout_utils.py` | → `execute_fastdream_generation()` 增加 `alg` 参数 |
| `d3llm_multiblock_bind.py` / `recipe/d3llm/d3llm_multiblock.py` | → 单一模块 `verl/workers/rollout/dream_multiblock.py`（或 vendored 进 `generation_utils_block.py`） |
| `dllm_fsdp_workers.py` 中 `model.name == "d3llm_dream"` 分支 | → `model.name == "dream"` + `rollout.dllm_decode=multiblock` |
| `dllm_main_ppo.py` 的 `D3LLM_ROOT` runtime_env | → 删除；multiblock 不依赖外部 repo |
| `fsdp_sglang_sdar` 用于 d3llm_dream | → 阶段 3 用独立 `fsdp_sglang_dream` 或复命名 |

### 2.3 旧分支中可 cherry-pick 的小修复（与 d3llm 无关）

```bash
# 在 feat/d3llm-dream-clean 上按需 cherry-pick（来自 feat/sglang-from-cb8ce6b）
git cherry-pick <commit>   # 或手动应用 diff

# 涉及文件：
# - verl/models/transformers/dream.py      DreamSdpaAttention 无 Flash 时 skip patch
# - verl/utils/dataset/rl_dataset.py       parquet List feature fallback
# - verl/utils/reward_score/code_reward.py   沙箱 stderr 截断
```

---

## 3. 最小文件清单（干净实现）

### 阶段 0 — 离线可运行（仅 recipe，不进 verl）

| 文件 | 作用 |
|------|------|
| `recipe/d3llm/setup_finetune_d3llm_model_code.sh` | 复制 Dream modeling 到权重目录 |
| `recipe/d3llm/verify_finetune_d3llm.py` | load / vanilla entropy / multiblock 冒烟 |
| `recipe/d3llm/README.md` | 阶段 0 说明 |

**约 3 个文件，~300 行。**

### 阶段 1 — Dream BGPO + HF multiblock（核心闭环）

| 文件 | 改动类型 | 作用 |
|------|----------|------|
| `verl/workers/rollout/rollout_utils.py` | **修改** | `execute_fastdream_generation(..., alg="entropy"\|"entropy_threshold")` |
| `verl/workers/rollout/dream_multiblock.py` | **新增** | multiblock 解码（从 d3LLM 官方逻辑 vendored，无 `D3LLM_ROOT`） |
| `verl/workers/rollout/fast_dream_rollout.py` | **修改** | 读 `config.dllm_decode` / `d3llm_*` 阈值，传给 rollout_utils |
| `verl/workers/dllm_fsdp_workers.py` | **修改** | 无需新 model name；Dream 路径读 rollout 配置即可 |
| `verl/models/transformers/dream.py` | **修改** | DreamSdpaAttention guard（cherry-pick） |
| `recipe/dream/run_bgpo_dream_coder_d3llm.sh` | **新增** | 从 `run_bgpo_dream_7b_instruct.sh` fork，`task=code`，path=finetune_d3LLM |

**约 1 新模块 + 3 处小改 + 1 脚本。Actor / BGPO / FSDP 零新增文件。**

### 阶段 2 — 验证与数据（按需）

| 文件 | 作用 |
|------|------|
| `verl/utils/dataset/rl_dataset.py` | parquet fallback（cherry-pick） |
| `verl/utils/reward_score/code_reward.py` | 日志截断（cherry-pick） |

### 阶段 3 — SGLang 加速（HF 闭环验证通过后）

| 文件 | 作用 |
|------|------|
| `verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py` | 通用 Dream SGLang（`dllm_algorithm` 配置） |
| `third_party/sglang/python/sglang/srt/dllm/algorithm/full_attn_multi_block.py` | 最小 upstream 补丁 |
| `recipe/d3llm/verify_sglang_dream_rollout.py` | HF vs SGLang 对齐（精简版，<200 行） |

---

## 4. 从 Dream 到 d3LLM Dream-Coder 的配置映射

**原则：不新增 `model.name=d3llm_dream`，沿用 `dream`。**

| 配置项 | Vanilla Dream 7B | d3LLM Dream-Coder |
|--------|------------------|-------------------|
| `actor_rollout_ref.model.name` | `dream` | `dream` |
| `actor_rollout_ref.model.path` | `models/Dream-v0-Instruct-7B` | `models/finetune_d3LLM` |
| `actor_rollout_ref.rollout.name` | `hf` | `hf`（阶段 1） |
| `actor_rollout_ref.rollout.dllm_decode` | `entropy`（默认） | `multiblock` |
| `actor_rollout_ref.rollout.block_length` | 32 | 32 |
| `actor_rollout_ref.rollout.mask_token_id` | 126336 | **151666**（从 config 读） |
| `actor_rollout_ref.rollout.d3llm_threshold` | — | 0.5 |
| `actor_rollout_ref.actor.*` | `dream_dp_actor_bgpo` | **相同** |
| FSDP wrap | `DreamDecoderLayer` | **相同** |

Hydra 示例（阶段 1 smoke）：

```bash
python -m verl.trainer.dllm_main_ppo \
  +actor_rollout_ref.model.name=dream \
  actor_rollout_ref.model.path=models/finetune_d3LLM \
  actor_rollout_ref.rollout.name=hf \
  +actor_rollout_ref.rollout.dllm_decode=multiblock \
  +actor_rollout_ref.rollout.block_length=32 \
  +actor_rollout_ref.rollout.mask_token_id=151666 \
  +actor_rollout_ref.rollout.d3llm_threshold=0.5 \
  +actor_rollout_ref.algorithm.name=bgpo \
  ...  # 其余与 recipe/dream/run_bgpo_dream_7b_instruct.sh 对齐，task 改 code
```

---

## 5. 分阶段实施顺序

```
阶段 0  recipe 离线 verify（load + multiblock generate）
   ↓
阶段 1a  rollout_utils 加 alg 分支 + dream_multiblock.py
   ↓
阶段 1b  单卡 / 2 卡 HF smoke：1 step BGPO，检查组内 reward std > 0
   ↓
阶段 1c  code 任务 smoke：HumanEval val_before_train
   ↓
阶段 2   cherry-pick 数据/reward 小修复
   ↓
阶段 3   SGLang FullAttnMultiBlock（仅当 HF 太慢且阶段 1 有学习信号）
```

### 阶段 0 完成状态（2026-05-26）

- [x] `recipe/d3llm/` 四文件就位（setup / verify / multiblock / README）
- [x] `docs/第一阶段SDAR+BGPO兼容与debug内容.md` 自旧分支复制
- [x] `verify_finetune_d3llm.py --mode load` 通过
- [x] `verify_finetune_d3llm.py --mode vanilla` 通过
- [x] `verify_finetune_d3llm.py --mode multiblock` 通过（NFE=21 @ max_new_tokens=64）

### 阶段 1 完成状态（2026-05-26）

- [x] `verl/workers/rollout/d3llm_dream_generate_util.py`（vendored 自 d3LLM 官方）
- [x] `verl/workers/rollout/dream_multiblock.py`（bind + rollout 执行）
- [x] `rollout_utils.execute_fastdream_generation` 增加 `dllm_decode=multiblock` 分支
- [x] `fast_dream_rollout.py` 读取 d3llm 超参 + `per_sample_seed`
- [x] `verl/models/transformers/dream.py` DreamSdpaAttention guard
- [x] `recipe/dream/run_bgpo_dream_coder_d3llm.sh`（HF smoke/full）
- [x] verl multiblock 路径单卡验证通过

### 阶段 1 成功标准（训练闭环）

- [ ] BGPO smoke：`bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke` 跑通 1 epoch
- [ ] 组内 reward 标准差 > 0（`n_rollout>=2`, `temperature>0`）
- [ ] 同一 prompt：`entropy` vs `multiblock` 输出不同且 multiblock 更短 NFE
- [ ] BGPO smoke：`n_rollout=4` 时组内 reward 标准差 > 0
- [ ] 1 个 epoch smoke 无 Ray hang；`Training Progress` 持续推进
- [ ] HumanEval pass@1 相对 base 有可见变化（或 reward 均值上升）

### 阶段 1 禁止事项

- 不引入 `d3llm_dream` model name
- 不默认 `engine=sglang`
- 不 fork `third_party/sglang`
- 不依赖 `D3LLM_ROOT` 环境变量
- 不复用 `fsdp_sglang_sdar` 给 Dream

---

## 6. `execute_fastdream_generation` 改造草图

当前（8984846）：

```python
outputs = module.diffusion_generate(..., alg="entropy", ...)
```

目标：

```python
alg = gen_kwargs.get("alg", "entropy")  # "entropy" | "entropy_threshold"
if alg == "entropy_threshold":
    from verl.workers.rollout.dream_multiblock import multiblock_generate
    outputs = multiblock_generate(module, idx_repeat, attention_mask_repeat, gen_kwargs)
else:
    outputs = module.diffusion_generate(..., alg="entropy", ...)
```

`fast_dream_rollout.py` 在 validate / train 分支根据 `config.dllm_decode` 设置 `gen_kwargs["alg"]` 及 d3llm 阈值参数。

---

## 7. 旧分支文件对照（便于 code review）

| 旧路径 | 处置 |
|--------|------|
| `recipe/d3llm/*`（6 文件 + diagnose） | 阶段 0 保留 3 个；其余删除或阶段 3 重写 |
| `verl/workers/rollout/fast_d3llm_dream_rollout.py` | 删除 → 合并进 `fast_dream_rollout.py` |
| `verl/workers/rollout/d3llm_multiblock_bind.py` | 删除 → `dream_multiblock.py` |
| `verl/workers/rollout/d3llm_rollout_debug.py` | 删除 |
| `verl/workers/rollout/sglang_rollout/sglang_d3llm_dream_rollout.py` | 删除 → 阶段 3 `sglang_dream_rollout.py` |
| `verl/workers/sharding_manager/fsdp_hf_rollout.py` | 删除（Dream HF 不需要） |
| `verl/trainer/dllm_main_ppo.py` D3LLM_ROOT 块 | 不引入 |
| `verl/workers/dllm_fsdp_workers.py` d3llm 分支 | 不引入；仅 rollout config |
| `third_party/sglang` Dream 补丁 | 子模块已清理；阶段 3 再加 |

---

## 8. 当前仓库状态

```bash
git branch
# * feat/d3llm-dream-clean  @ 8984846
#   feat/sglang-from-cb8ce6b  （旧 d3llm 实现，保留作参考）
#   main

git stash list
# stash@{0}: wip d3llm before feat/d3llm-dream-clean branch
```

下一步：在 `feat/d3llm-dream-clean` 上按 **§3 阶段 0** 恢复 `recipe/d3llm/` 三个文件（可从 `2738878` cherry-pick 或从 stash 提取），然后实施 **§5 阶段 1a**。
