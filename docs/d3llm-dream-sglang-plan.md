# d3LLM Dream-Coder × SGLang 独立推理 Pipeline 规划

> 前置：阶段 1 HF multiblock 已跑通（`20260526_231907` smoke：4 rank `batch_done`，FSDP NFE sync 修复生效）。  
> 分支：`feat/d3llm-dream-clean`  
> 关联文档：[d3llm-dream-clean-plan.md](./d3llm-dream-clean-plan.md)

---

## 1. 为什么要上 SGLang

| 问题（HF 路径） | SGLang 解法 |
|----------------|-------------|
| multiblock **变长 NFE** + 共享 FSDP → collective 死锁 | 推理与 FSDP **物理分离** |
| NFE sync + dummy forward → 所有 rank 被最慢样本拖住（~95s/sample） | 各 rank 独立 Engine，无 FSDP forward 同步 |
| rollout 占满 GPU，与 BGPO update 争抢 | `memory_saver` + `release_memory_occupation` 交替 |

**原则不变**：`model.name=dream`（不引入 `d3llm_dream`）；BGPO update 仍走 FSDP `dream_dp_actor_bgpo`。

---

## 2. 目标架构

```
┌─────────────────────────────────────────────────────────┐
│  Ray Worker (每 GPU 一个)                                │
│                                                         │
│  ┌──────────────────┐    state_dict sync   ┌─────────┐ │
│  │ FSDP2 Actor      │ ──────────────────► │ SGLang  │ │
│  │ (训练 + log_prob)│   FSDPSGLangSDAR    │ Engine  │ │
│  └──────────────────┘   ShardingManager    │ FullAttn│ │
│         ▲                                  │ MultiBlk│ │
│         │ BGPO update                      └────┬────┘ │
│         │                                       │ rollout │
└─────────┼───────────────────────────────────────┼────────┘
          │                                       ▼
     forward_process                         responses
```

参考实现：**SDAR**（`SGLangSDARRollout` + `FSDPSGLangSDARShardingManager` + `recipe/sdar/run_bgpo_sdar_8b_chat.sh`）。

---

## 3. 与 HF / 其他模型对比

| 路径 | 推理模块 | 解码步数 | FSDP 风险 |
|------|----------|----------|-----------|
| AR `HFRollout` | 同一 FSDP + `summon_full_params` | 固定上界 | 低（FSDP1） |
| vLLM / SGLang AR | 独立引擎 | 引擎内部 | 无 |
| Dream entropy HF | 同一 FSDP | 固定 256 steps | 低 |
| **Dream multiblock HF** | 同一 FSDP + unshard + NFE sync | **变长 NFE** | 高（已用 hack 修） |
| **Dream multiblock SGLang** | 独立 Engine | 变长 NFE（Engine 内） | **无** |

---

## 4. 分阶段实施

### 阶段 3A — SGLang 上游 + 离线 Engine（当前）

| 项 | 内容 |
|----|------|
| Submodule | `third_party/sglang` → PR [#20615](https://github.com/sgl-project/sglang/pull/20615)（`FullAttnMultiBlock` + `DreamModel`） |
| 脚本 | `recipe/d3llm/verify_sglang_engine_smoke.py` |
| 验收 | 单卡 Engine 启动 + 1 条 prompt 生成，无 mask 残留 |

```bash
bash recipe/d3llm/setup_finetune_d3llm_model_code.sh   # 若未做过
python recipe/d3llm/verify_sglang_engine_smoke.py --smoke
```

### 阶段 3B — HF vs SGLang 对齐

| 文件 | 作用 |
|------|------|
| `recipe/d3llm/verify_sglang_dream_rollout.py` | 精简版（<200 行），HF multiblock vs SGLang token 对齐 |

### 阶段 3C — VERL Rollout 集成

| 文件 | 动作 |
|------|------|
| `verl/workers/rollout/sglang_rollout/sglang_dream_rollout.py` | **新增**（自旧 `sglang_d3llm_dream_rollout.py` 迁移，去 `d3llm_dream` 命名） |
| `verl/workers/dllm_fsdp_workers.py` | `rollout.name=sglang` + `model.name=dream` + `dllm_decode=multiblock` 分支 |
| Sharding | 复用 `FSDPSGLangSDARShardingManager` |

**不要引入**：`FSDPHFRolloutShardingManager`、`D3LLM_ROOT`、`model.name=d3llm_dream`。

### 阶段 3D — 训练脚本 + 显存

| 文件 | 改动 |
|------|------|
| `recipe/dream/run_bgpo_dream_coder_d3llm.sh` | `--engine sglang`；`unset PYTORCH_CUDA_ALLOC_CONF`；SGLang mem 参数 |

4×A6000 48GB 建议（对齐 SDAR smoke）：

- `mem_fraction_static=0.35`
- `attention_backend=torch_native`（smoke）
- `disable_cuda_graph=True`
- `actor.fsdp_config.param_offload=True`
- `model.enable_activation_offload=True`

### 阶段 3E — 4 卡 Ray smoke

- rollout-only → 完整 BGPO 1 epoch
- 成功标志：4 rank 无 hang、显存不 OOM、`Training Progress` 推进

---

## 5. Hydra 配置映射（SGLang 路径）

| 配置项 | HF multiblock（阶段 1） | SGLang（阶段 3） |
|--------|------------------------|------------------|
| `rollout.name` | `hf` | `sglang` |
| `rollout.dllm_decode` | `multiblock` | `multiblock`（路由到 SGLang） |
| `rollout.dllm_algorithm` | — | `FullAttnMultiBlock` |
| `rollout.d3llm_threshold` | 0.5 | 0.5 |
| `rollout.block_length` | 32 | 32 |
| `rollout.d3llm_cache_delay_iter` | 32 | 32 |
| `rollout.mem_fraction_static` | — | 0.35（smoke） |
| `rollout.attention_backend` | — | `torch_native` / `flashinfer` |
| `actor.strategy` | `fsdp2` | `fsdp2`（不变） |

---

## 6. 风险与决策

| 风险 | 缓解 |
|------|------|
| PR #20615 未 merge main | submodule 固定 `pr-20615-d3llm-dream` 分支 |
| 4×48GB OOM | param_offload + activation_offload + 低 mem_fraction |
| 权重 sync 慢 | 首版接受；后续 layered_summon |
| SGLang ≠ HF 输出 | 3B verify 强制对齐后再开训练 |
| Ray 子进程 CUDA | 照搬 SDAR 的 `CUDA_HOME` / `LD_LIBRARY_PATH` |

---

## 7. 与 HF FSDP sync 修复的关系

| 路径 | 建议 |
|------|------|
| HF + NFE sync（当前未提交改动） | 保留作 fallback / debug |
| SGLang 上线后 | 生产默认 `engine=sglang` |
| `fsdp_rollout_inference_context` | SGLang 路径不再使用 |

---

## 8. 进度跟踪

- [x] **3A** Submodule 升级 + `verify_sglang_engine_smoke.py` PASS（2026-05-26，`c795ddb2e`，单卡 64 token ~11.5s，无 mask 残留）
- [ ] **3B** HF vs SGLang 对齐 verify
- [ ] **3C** `SGLangDreamRollout` + worker 路由
- [ ] **3D** 训练脚本 SGLang 模式
- [ ] **3E** 4 卡 BGPO smoke PASS

---

## 9. Submodule 版本记录

| 日期 | Commit | 说明 |
|------|--------|------|
| 2026-05-26 基线 | `bbe9c7e` | 无 Dream / FullAttnMultiBlock |
| 2026-05-26 3A | `c795ddb2e` | PR #20615 `pr-20615-d3llm-dream` |

更新 submodule：

```bash
cd third_party/sglang
git fetch origin pull/20615/head:pr-20615-d3llm-dream
git checkout pr-20615-d3llm-dream
```
