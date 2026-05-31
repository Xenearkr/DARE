# 第四阶段：d3LLM Certainty-Forcing Loss（CFL）接入 BGPO 优化目标

> 前置：[第三阶段文档](./第三阶段D3LLM-Dream-Coder-TPF效率奖励与联合监控.md)（TPF 效率奖励、EvalPlus 格式对齐、联合监控）  
> 动机：第三阶段在 **rollout 后** 用 `efficiency_reward` 塑形 TPF，信号 **稀疏、滞后**，且只作用于 **已通过判题** 的样本；模型在 actor 更新阶段仍可能学到「高熵、犹豫」的 token 分布，与 multiblock **entropy threshold** 解码目标不对齐。  
> 本阶段目标：将 d3LLM 蒸馏训练中的 **Certainty-Forcing Loss（CFL，代码中称 entropy loss on correctly predicted tokens）** 作为 **稠密辅助损失** 接入 BGPO actor 更新，在 **不牺牲 pass@1** 的前提下，从 **logit 层面** 压低已正确 token 的熵，提高块解码可并行接受的 token 数，从而提升 TPF。

---

## 1. 结论先行：是否可行？

**可行，且与第三阶段 TPF 效率奖励互补。**

| 维度 | 第三阶段 TPF 奖励 | 第四阶段 CFL |
|------|-------------------|--------------|
| 作用阶段 | Rollout 之后（reward manager） | Actor `update_policy`（梯度） |
| 信号密度 | 样本级标量 | Token 级（masked 且 pred==label） |
| 优化对象 | 整段 rollout 的 NFE / TPF 比 | 单步前向下各位置的预测熵 |
| 与 multiblock 关系 | 间接（鼓励少步完成） | 直接（低熵 → 更多 token 低于 `threshold` 被接受） |
| 正确性约束 | 仅 passed 样本有效 | 默认仅 **pred==ground_truth** 的位置 |

二者组合：**CFL 塑造「敢不敢果断写」**，**TPF 奖励塑造「写对之后快不快」**；CFL 提供训练内连续梯度，TPF 奖励提供与真实 SGLang 解码路径一致的 outcome 反馈。

**主要风险（可控）：** BGPO 现有 `entropy_coeff` 在 policy loss 中 **最大化** token 熵（`pg_loss - entropy_coeff * entropy`），与 CFL **最小化** 熵方向相反；实现时必须 **解耦** 两项，并建议默认 `entropy_coeff=0` 或显著小于 `cfl_coef`。

---

## 2. d3LLM 中 CFL 的精确定义

### 2.1 命名与出处

- 论文 / 实现社区常用名：**Certainty-Forcing Loss**（继承自 dParallel，d3LLM Appendix A.7 明确采用 *certainty-forcing loss with entropy regularization*）。
- 代码中 **无** `ForceCertaintyLoss` 类名；在 `d3llm_dream_train.py` 的 `CustomTrainer.compute_loss` 里以 **`entropy_loss`** 实现，注释为 *Apply entropy loss only to "correctly predicted" tokens*。

### 2.2 核心公式

对一次 forward 得到的 logits \(\ell_{b,t,v}\) 与标签 \(y_{b,t}\)：

1. 温度缩放概率：\(p_{b,t} = \mathrm{softmax}(\ell_{b,t} / \tau)\)
2. Token 熵：\(H_{b,t} = -\sum_v p_{b,t,v}\log p_{b,t,v}\)
3. 正确性门控：\(M_{b,t} = \mathbb{1}[\arg\max_v \ell_{b,t,v} = y_{b,t}] \cdot \mathbb{1}[\text{position } (b,t) \text{ is masked}]\)
4. CFL：\(\mathcal{L}_{\mathrm{CFL}} = \frac{\sum_{b,t} H_{b,t}\cdot M_{b,t}}{\sum_{b,t} M_{b,t} + \epsilon}\)

总蒸馏损失（Dream，无 complementary mask 分支）：

\[
\mathcal{L} = \frac{1}{4}\left(\mathcal{L}_{\mathrm{CE}} + w_{\mathrm{ent}}\cdot \mathcal{L}_{\mathrm{CFL}}\right)
\]

其中 \(w_{\mathrm{ent}}\) 即配置项 `entropy_weight`（Dream / Dream-Coder 默认 **1.0**，LLaDA 默认 **2.0**），\(\tau\) 即 `temperature`（默认 **0.5**）。

### 2.3 代码锚点（d3LLM 仓库）

| 文件 | 说明 |
|------|------|
| `d3llm/d3llm_DREAM/distill_2_coder_training_512/d3llm_dream_train.py` | Dream-Coder 512 蒸馏，`compute_loss` 中 CE + CFL |
| `d3llm/d3llm_DREAM/distill_2_training/d3llm_dream_train.py` | Dream 通用蒸馏，逻辑相同 |
| `d3llm/d3llm_DREAM/distill_2_coder_training_512/d3llm_train.yaml` | `temperature: 0.5`, `entropy_weight: 1.0` |

关键逻辑（节选）：

```python
probs = F.softmax(logits / self.temperature, dim=-1)
H_tok = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
pred_ids = logits.argmax(dim=-1)
correct_mask = (pred_ids == input_ids) & masked_indices
entropy_loss = (H_tok * correct_mask).sum() / num_correct.clamp_min(1)
total_loss = (ce_loss + self.entropy_weight * entropy_loss) / 4.0
```

### 2.4 为何 CFL 能提升 TPF

推理侧 multiblock 解码（SGLang `FullAttnMultiBlock`）在每一步接受满足 **\(H_{b,t} < \texttt{threshold}\)** 的 token（默认 threshold 0.4–0.5）。CFL 在训练侧 **压低已正确 token 的 \(H\)**，使更多位置在相同 threshold 下可被 **并行 unmask**，从而减少 NFE、提高 TPF，且因门控在 **pred==label** 上，不强迫模型对错误预测「盲目自信」。

这与 d3LLM README 中「Entropy-Based Multi-Block Decoding → ~30% TPF improvement」形成 **训练–推理闭环**；第三阶段 TPF 奖励则是在 RL 层面对该闭环做 **outcome 对齐**。

---

## 3. DARE BGPO 现状与接入点

### 3.1 当前优化目标（Actor）

BGPO actor 更新路径（Dream 继承 LLaDA 实现）：

| 组件 | 路径 | 作用 |
|------|------|------|
| 随机 mask 前向 | `verl/trainer/ppo/dllm_core_algos.py::_forward_process_bgpo` | 对 response 随机 mask，构造 `perturbed_seq` / `mask_indices` / `p_mask` |
| ELBO / log prob | `verl/workers/actor/llada_dp_actor_bgpo.py::_forward_micro_batch` | 在 mask 位置算 weighted CE → `loss_per_sample` |
| Policy gradient | `compute_policy_loss_bgpo` | 对 `l_theta` 与 `old_l_theta` 做 clipped ratio |
| PPO 熵 bonus | `update_policy`：`policy_loss = pg_loss - entropy_coeff * entropy_loss` | **鼓励高熵**（探索） |
| KL（可选） | `use_kl_loss` | 锚定参考策略 |

Dream 特化：`verl/workers/actor/dream_dp_actor_bgpo.py::_get_logits` 对 logits 做 **shift**（与 Dream 标签对齐），CFL 必须使用 **同一套 shift 后 logits** 与 `input_ids` 比较。

### 3.2 与 d3LLM 蒸馏的差异

| 项目 | d3LLM 蒸馏 | DARE BGPO |
|------|------------|------------|
| Mask 方式 | 伪轨迹 + 窗口 mask ratio 课程 | `_forward_process_bgpo` 随机 mask |
| 主损失 | CE + CFL | PG on masked ELBO |
| 标签来源 | 固定 ground-truth 序列 | Rollout 的 `responses`（RL 轨迹） |
| 探索 | 无（纯 SFT/蒸馏） | `entropy_coeff`、采样 rollout |

**接入 CFL 不改变 BGPO 的 PG 主目标**；仅在 **同一 forward 的 mask 子集** 上增加辅助项，与 d3LLM 在「masked 位置 + 正确预测」上施力一致。

### 3.3 推荐接入位置

**首选：** 在 `llada_dp_actor_bgpo.py::update_policy` 的内层 MC 循环中，复用 `_forward_micro_batch` 已算的 **per-position logits**（需小改：让 forward 返回 `token_entropy` 或在此处二次计算 CFL，避免重复 forward）。

**次选（更清晰）：** 新增 `verl/trainer/ppo/dllm_core_algos.py::compute_cfl_loss`，由 Dream actor 调用：

```python
cfl_loss = compute_cfl_loss(
    logits=logits_b,           # (seq_len, vocab), 已 shift
    labels=seq[b],             # (seq_len,)
    mask_indices=cur_mask_indices[b],
    temperature=cfl_temperature,
)
policy_loss = pg_loss - entropy_coeff * entropy_loss + cfl_coef * cfl_loss
```

**不要**把 CFL 放进 `reward_manager`：CFL 需要 **vocab 维 logits**，reward 阶段只有 rollout 文本与标量 TPF，无法高效计算。

---

## 4. 总体设计

### 4.1 损失组合

Actor 单步总损失（每个 MC 样本）：

\[
\mathcal{L}_{\mathrm{actor}} =
\underbrace{\mathcal{L}_{\mathrm{PG}}}_{\text{BGPO clipped loss}}
- \lambda_{\mathrm{ent}} \underbrace{\mathcal{L}_{\mathrm{PPO-ent}}}_{\text{可选探索}}
+ \lambda_{\mathrm{cfl}} \underbrace{\mathcal{L}_{\mathrm{CFL}}}_{\text{正确 mask 位置熵}}
+ \lambda_{\mathrm{kl}} \mathcal{L}_{\mathrm{KL}}
\]

默认建议：

- \(\lambda_{\mathrm{cfl}} = 1.0\)（对齐 d3LLM Dream-Coder `entropy_weight`）
- `cfl_temperature = 0.5`
- \(\lambda_{\mathrm{ent}} = 0\) 或 \(\ll \lambda_{\mathrm{cfl}}\)（避免与 CFL 对打）
- \(\lambda_{\mathrm{kl}}\)：沿用第三阶段全量实验计划（acc 漂移时开启）

CFL 在 MC 维上与 PG 相同：**对 `mc_num` 次 mask 采样取平均**，再除以 `gradient_accumulation`。

### 4.2 门控策略（RL 特化，建议分档实现）

d3LLM 使用 `correct_mask = (pred == label) & masked`。接入 BGPO 时 label 为 **rollout response token**（非数据集 gold code），在 RL 中更合理扩展为：

| 档位 | 门控 | 适用场景 |
|------|------|----------|
| **A（最小改动，对齐 d3LLM）** | `(pred == response_token) & mask` | 快速验证 CFL 实现与 TPF 相关性 |
| **B（推荐）** | 档位 A + 样本级 `pass_reward == 1` | 仅对判题通过轨迹施 CFL，避免强化错误代码的「自信错误」 |
| **C（可选）** | 档位 B + `advantage > 0` | 仅强化相对旧策略更优且正确的 token，与 PG 信号一致 |

**Phase 4 smoke 路径：** 先 **A** 验证 plumbing 与 metric；全量默认 **B**。

### 4.3 与第三阶段 TPF 效率奖励的分工

```
Rollout (SGLang multiblock)
    │
    ├─► pass_reward + efficiency_reward  ──► advantage ──► PG loss
    │
    └─► responses 作为 label ──► Actor forward (random mask)
              │
              └─► CFL on (pred==label) & mask  ──► 辅助梯度（稠密）
```

- **CFL**：训练步内、token 级、与 entropy threshold **机制对齐**。
- **efficiency_reward**：episode 级、仅 passed、与真实 **NFE/TPF 统计** 对齐。
- 若只开 TPF 奖励不开 CFL：可能出现 acc 升、TPF 降（第三阶段已观察到的「犹豫」）。
- 若只开 CFL 不开 TPF 奖励：logit 变 sharp，但 rollout 未必减少无效前向（如 fence/EOS 行为未优化）。

**第四阶段当前脚本默认（`run_bgpo_dream_coder_d3llm.sh`）：** `enable_tpf_efficiency=False`（仅 pass/fail 奖励）；训练集恢复为第二阶段三数据集混训（`lcbv5-K8_1` + `primeintellect-K8_1` + `taco-K8_1`），验证集仍为 EvalPlus 格式（full：`humaneval_evalplus_1`；smoke：`humaneval_evalplus_smoke_8`）。CFL 实现接入后，在 **不开启 TPF 效率奖励** 的前提下单独验收 CFL 对 TPF 的塑形效果；需要与第三阶段叠加时再显式打开 `enable_tpf_efficiency=True`。

**可选组合：** 第三阶段 TPF 奖励 + 第四阶段 CFL 以较小 \(\lambda_{\mathrm{cfl}}\) 叠加，联合监控 acc / TPF / CFL 值。

### 4.4 实现模块划分

| 模块 | 改动 | 优先级 |
|------|------|--------|
| `dllm_core_algos.py` | 新增 `compute_cfl_loss(logits, labels, mask, temperature)` | P0 |
| `llada_dp_actor_bgpo.py` | `_forward_micro_batch` 返回各样本 logits 或 token-level entropy；`update_policy` 叠加 CFL | P0 |
| `dream_dp_actor_bgpo.py` | 确认 shift 后 logits 与 label 对齐；必要时 override CFL 调用 | P0 |
| `dllm_fsdp_workers.py` | actor config 透传 `enable_cfl`, `cfl_coef`, `cfl_temperature`, `cfl_gate_passed_only` | P1 |
| `recipe/dream/run_bgpo_dream_coder_d3llm.sh` | Hydra / CLI 超参 | P1 |
| `verl/trainer/ppo/dllm_metric_utils.py` | W&B：`cfl_loss/mean`, `cfl_active_tokens/mean`, `token_entropy/mean`（mask & correct 子集） | P1 |

**实现注意（Dream FSDP packed forward）：** 当前 `_forward_micro_batch` 在 packed logits 还原为 `logits_b` 的循环内 **已有完整 vocab logits**，CFL 应在此内层计算，避免额外 forward。entropy 计算建议 **float32/float64**（对齐 SGLang `compute_entropy_float64` 与 d3LLM 数值习惯），与 bf16 forward 解耦。

### 4.5 配置项（建议）

| 超参 | 默认 | 作用 |
|------|------|------|
| `enable_cfl` | `False`（smoke 验收后开） | 总开关 |
| `cfl_coef` | `1.0` | CFL 强度，对应 d3LLM `entropy_weight` |
| `cfl_temperature` | `0.5` | softmax 温度，**仅用于 CFL 熵计算**，与 rollout `temperature` 无关 |
| `cfl_gate_passed_only` | `True` | 是否仅对 `pass_reward==1` 样本计算 CFL |
| `cfl_gate_positive_adv_only` | `False` | 是否额外要求 `advantage>0` |
| `entropy_coeff` | `0` | 与 CFL 冲突时建议置 0 |

脚本入口（规划）：`actor_rollout_ref.actor.cfl_*` 与 `run_bgpo_dream_coder_d3llm.sh` 中新增 export。

**当前脚本已固定：**

| 项 | smoke / full 默认 |
|----|-------------------|
| `enable_tpf_efficiency` | `False` |
| 训练集 | `lcbv5-K8_1` + `primeintellect-K8_1` + `taco-K8_1` |
| 验证集 | smoke：`humaneval_evalplus_smoke_8`（取自 `humaneval_evalplus_1`）；full：`humaneval_evalplus_1` |

---

## 5. 风险与缓解

| 风险 | 现象 | 缓解 |
|------|------|------|
| CFL vs PPO entropy | 策略振荡、acc 下降 | 默认 `entropy_coeff=0`；监控 `cfl_loss` 与 `pass_reward` |
| 错误 rollout 上的 CFL | 强化错误代码的高置信 | 使用 `cfl_gate_passed_only=True` |
| 与 KL 同时过强 | 模式坍缩、多样性丧失 | 先固定 `cfl_coef=0.5` smoke；KL 仅在 acc 漂移时开 |
| BGPO 随机 mask 与块解码不对齐 | CFL 提升有限 | 保留第三阶段 TPF 奖励；可选 Phase 4b：EBPO 块级 mask（`dllm_core_algos::_forward_process_ebpo`） |
| 算力 | 内层多算 entropy | 仅 mask 位置子集计算；与 CE 共用 logits |
| Rollout label 非 gold | 档位 A 会蒸馏 rollout 噪声 | 全量用档位 B；分析时对比 gold-code SFT 子集（可选 ablation） |

---

## 6. 验收标准与实验计划

### 6.1 Smoke（HumanEval 164，SGLang greedy fix 后）

1. 开启 `enable_cfl=true`, `cfl_coef=1.0`, `cfl_gate_passed_only=false`（档位 A）：确认 `cfl_loss` 有限、反传无 NaN、W&B 指标齐全。
2. 对比 **step 0 vs 若干 step**：
   - `val-core/humaneval/acc` 不显著低于仅 TPF 奖励基线
   - `val-aux/humaneval/tpf` **≥ 基线** 或同 acc 下 `rollout_nfe` 下降
   - `cfl_active_tokens/mean` 随训练下降（正确位置熵降低）

### 6.2 全量 BGPO

在 **第二阶段训练数据 + 仅 pass/fail 奖励**（当前脚本默认）上 **+CFL（档位 B）**；对照第三阶段时可额外打开 TPF 效率奖励与 EvalPlus 混训。

| 指标 | 期望 |
|------|------|
| HumanEval pass@1 | 不低于仅 pass/fail 基线（C0） |
| TPF / AUP 代理（passed 子集 mean TPF） | 相对 C0 **+5%～15%**（参考 d3LLM 蒸馏中 certainty-forcing 的量级） |
| acc–TPF 曲线 | acc 升时 TPF 不再系统性下跌 |

对照组：

- C0：仅 pass/fail + 三数据集混训（当前脚本默认）
- C1：C0 + TPF 效率奖励（第三阶段，需显式开启）
- C2：C0 + CFL（第四阶段主实验）
- C3（可选）：C2 + TPF 效率奖励，或 C2 + `use_kl_loss`

### 6.3 命令（规划）

```bash
# Smoke：CFL plumbing + 指标（TPF 效率奖励默认关闭）
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine sglang \
  actor_rollout_ref.actor.enable_cfl=true \
  actor_rollout_ref.actor.cfl_coef=1.0

# 全量：三数据集 + 仅 pass/fail + CFL
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --engine sglang \
  actor_rollout_ref.actor.enable_cfl=true \
  actor_rollout_ref.actor.cfl_gate_passed_only=true

# 可选：叠加第三阶段 TPF 效率奖励
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --engine sglang \
  reward_model.reward_kwargs.enable_tpf_efficiency=true \
  actor_rollout_ref.actor.enable_cfl=true \
  actor_rollout_ref.actor.cfl_gate_passed_only=true
```

---

## 7. 阶段边界与后续（Phase 4b，本文不实现）

以下 **不在第四阶段 P0 范围**，但作为自然延伸记录在案：

1. **块级 mask（EBPO 风格）**：用 `_forward_process_ebpo` 替代全 response 随机 mask，使 actor 更新与 block_size=32 更对齐。
2. **伪轨迹辅助数据**：离线 teacher trajectory + 在线 BGPO 混合（接近 d3LLM 完整蒸馏，成本高）。
3. **Threshold 联合调参**：训练后微调 `entropy_threshold` / `block_add_threshold`，与 CFL 后的 logit 分布匹配。
4. **Complementary masking loss**：d3LLM 中 `use_complementary_loss` 分支，Dream-Coder 若开启 complementary attention 再考虑。

---

## 8. 小结

d3LLM 的 **Certainty-Forcing Loss（CFL）** 本质是：在 **masked 且已预测正确** 的 token 上 **最小化预测熵**，从而与 **entropy-based multiblock 解码** 同向，提升 TPF。DARE BGPO 已在 `_forward_micro_batch` 具备所需 logits 与 mask 结构，**技术上可在 actor `update_policy` 以辅助损失接入**，无需改动 rollout 或 reward 主链路。

与第三阶段 **TPF 效率奖励** 可形成 **梯度塑形 + 结果塑形** 的双层结构；当前训练脚本默认 **关闭 TPF 效率奖励**，第四阶段先以 CFL 单独塑形，并以 **acc 与 TPF 联合监控** 验收，默认对 **passed rollout** 门控 CFL，以避免强化错误代码。

---

## 附录：关键路径速查

| 用途 | 路径 |
|------|------|
| d3LLM CFL 实现 | `~/Codes/d3LLM/d3llm/d3llm_DREAM/distill_2_coder_training_512/d3llm_dream_train.py` |
| d3LLM CFL 超参 | `~/Codes/d3LLM/d3llm/d3llm_DREAM/distill_2_coder_training_512/d3llm_train.yaml` |
| BGPO random mask | `verl/trainer/ppo/dllm_core_algos.py::_forward_process_bgpo` |
| BGPO actor 更新 | `verl/workers/actor/llada_dp_actor_bgpo.py::update_policy` |
| Dream logits shift | `verl/workers/actor/dream_dp_actor_bgpo.py::_get_logits` |
| SGLang entropy 解码 | `third_party/sglang/.../full_attn_multi_block.py` |
| 第三阶段 TPF 奖励 | `verl/utils/reward_score/code_efficiency.py`, `verl/workers/reward_manager/dllm.py` |
