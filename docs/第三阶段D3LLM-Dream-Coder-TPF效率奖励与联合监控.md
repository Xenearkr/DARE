# 第三阶段：d3LLM Dream-Coder TPF 效率奖励与联合监控

> 前置：[第二阶段文档](./第二阶段D3LLM-Dream-Coder+BGPO兼容与debug内容.md)（兼容、debug、第一次全量训练复盘）  
> 动机：第一次全量 BGPO 出现 **HumanEval acc 先升后跌**；离线分析还发现 **acc 略升时 TPF 反而下降**——模型更「犹豫」，块解码置信度退化，实际能力变差。  
> 本阶段目标：在 **不牺牲正确性** 的前提下，用 **TPF 效率项** 约束解码行为，并在 W&B 中 **联合监控 acc 与 TPF**。

---

## 1. 整体思路

此前奖励只有 **pass/fail（0/1）**，模型可以通过 **更多前向、更保守的 mask 策略** 换略高的通过率，但推理效率下降。这对 diffusion LLM 尤其危险：BGPO 的随机 mask ELBO 与 multiblock 块解码目标本身就不完全对齐，纯二元奖励更容易把策略推向「多算几步、多猜几次」。

第三阶段的补强是 **奖励分解 + 联合监控**：

| 分量 | 含义 | 训练 | 验证 |
|------|------|------|------|
| `pass_reward` | EvalPlus 判题是否通过 | 0 / 1 | 0 / 1 |
| `efficiency_reward` | 相对 baseline 的 TPF 塑形 | 仅 **passed** 样本 | **0**（只记录 TPF，不参与打分） |
| `reward` | 用于 RL 的总奖励 | `pass_reward + efficiency_reward` | `pass_reward` |

验证阶段不算 efficiency，避免「为了 TPF 刷分」影响评测；训练阶段则对 **已经答对** 的 rollout 再鼓励更高效的解码。

本阶段与第二阶段的其他修复 **配套使用**（EvalPlus 混训数据、代码提取修复等），否则 pass_reward 不可信，baseline 也会被污染。

---

## 2. 为什么用 TPF，而不是 NFE

- **NFE**（前向次数）是绝对量，受 `max_response_length`、block 完成度、早停等影响，**跨样本不可比**。
- **TPF**（tokens per forward）= 有效生成 token 数 / NFE，衡量 **每一步前向平均「固化」多少 token**，反映 multiblock 解码是否果断。
- 我们关心的是：**答对的前提下，能否用更少犹豫完成生成**——这是 TPF 要刻画的东西，而不是单纯压低 NFE。

---

## 3. 单样本 TPF 怎么算

对每个 rollout 样本：

1. **`rollout_nfe`**：该样本解码总前向次数（SGLang 多轮 `nfe` 列表时求和）。
2. **`rollout_gen_tokens`**：response 中有效 token 数（按 attention mask / EOS 计，非 pad）。
3. **`tpf`** = `rollout_gen_tokens / rollout_nfe`（任一为 0 则 TPF 记 0）。

Rollout 阶段把 `rollout_nfe`、`rollout_gen_tokens` 写入 batch，奖励管理器据此算 TPF。

---

## 4. Baseline TPF 怎么算

Baseline 表示 **「当前训练阶段，通过样本的典型 TPF」**，用作效率奖励的参照，而不是固定常数。

**更新规则（仅 passed 样本）：**

- 每个 batch 先算完所有样本的 `pass_reward`；
- 从 **通过测试** 且 `tpf > 0` 的样本收集 TPF；
- 用 EMA 更新 baseline：
  - 首个有效样本：`baseline ← tpf`
  - 之后：`baseline ← α · tpf + (1 − α) · baseline`

**未见过任何 passed 样本前**，使用冷启动值 `tpf_baseline_initial`（默认 2.0，接近离线 multiblock 典型量级）。

**为何 baseline 必须排除失败样本：** 失败样本往往 NFE 高、TPF 低（反复 mask 仍错）。若纳入 baseline，会把参照拉低，让「低效但偶尔蒙对」的样本获得虚高 efficiency bonus，干扰对「健康解码」的刻画。

---

## 5. `efficiency_reward` 的计算与设计理由

### 5.1 公式

对 **通过测试** 的样本（`passed=True` 且 `tpf > 0`）：

```
raw = coef × (tpf / baseline − 1)
efficiency_reward = clip(raw, −max_penalty, +max_bonus)
```

最终 **`reward = pass_reward + efficiency_reward`**。

### 5.2 设计理由

- **相对 baseline 的比率形式** `(tpf/baseline − 1)`：TPF 高于近期「答对样本的常态」则加分，低于则减分；随训练动态适应，不依赖绝对 NFE。
- **仅对 passed 样本生效**：正确性是硬约束；效率是 **在答对之后的二次优化**，避免模型用「更快但全错」换 efficiency 分。
- **线性 + 裁剪**：`coef` 控制塑形强度；`max_bonus` / `max_penalty` 防止 efficiency 项压过 0/1 的 `pass_reward`，保持 RL 信号主次分明。
- **验证不算 efficiency**：评测只看能力（acc / pass_reward），TPF 仅作 `val-aux` 监控，用于观察 acc 与效率是否同向变化。

直观理解：baseline 是「近期答对题目的平均解码节奏」；比它更果断（TPF 更高）给小幅奖励，更犹豫则小幅惩罚。

---

## 6. 超参作用

| 超参 | 默认 | 作用 |
|------|------|------|
| `enable_tpf_efficiency` | `True` | 总开关；关闭后等价于仅 pass/fail 奖励 |
| `tpf_efficiency_coef` | `0.1` | 效率项整体强度；越大 TPF 对 `reward` 影响越大 |
| `tpf_baseline_ema_alpha` | `0.1` | baseline 跟踪速度；越大越跟当前 batch，越小越平滑 |
| `tpf_baseline_initial` | `2.0` | 冷启动 baseline，在尚无 passed 样本时使用 |
| `tpf_efficiency_max_bonus` | `0.25` | efficiency 加分上限 |
| `tpf_efficiency_max_penalty` | `0.25` | efficiency 减分下限（绝对值） |

**调参原则：** smoke 上 `|efficiency_reward/mean|` 应明显小于 `pass_reward/mean`（例如 0.006 vs 0.5）。若效率项过大，降低 `coef` 或收紧裁剪；若几乎不起作用，可适当提高 `coef`。全量训练应同时看 **acc 与 TPF**——acc 升而 TPF 持续跌，说明塑形不足或需配合 KL 锚定。

脚本入口：`recipe/dream/run_bgpo_dream_coder_d3llm.sh` 中 `reward_model.reward_kwargs.*`。

---

## 7. 联合监控（W&B）

训练与验证分别记录，便于对照「能力」与「效率」：

**训练（每 step）：** `pass_reward/mean`、`reward/mean`、`efficiency_reward/mean`、`tpf/mean`、`tpf_passed/mean`（仅 passed）、`tpf_baseline/mean`、`rollout_nfe/mean`。

**验证：** `val-core/humaneval/acc` 为核心指标；`val-aux/humaneval/pass_reward`、`tpf`、`rollout_nfe` 等为辅助，用于联合观察 acc 与解码效率是否一致。

---

## 8. 阶段验收与后续

- **Smoke 已通过**（`tpf_efficiency_coef=0.1` 下 efficiency 量级合理，监控指标齐全）。
- **待全量实验：** 在 EvalPlus 混训 + 本奖励方案下重训，对比 acc 曲线与 TPF 曲线；必要时尝试 `use_kl_loss` 抑制策略漂移。

```bash
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine sglang   # 验收
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --engine sglang          # 全量
```
