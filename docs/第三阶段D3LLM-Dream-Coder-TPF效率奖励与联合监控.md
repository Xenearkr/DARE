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

## 2. EvalPlus 对齐的数据集格式改造

第一次全量 BGPO 用的是旧 parquet：训练侧是 LiveCodeBench / TACO / PrimeIntellect 的 **填空、补全** 风格 prompt，验证侧是 `humaneval_1` 的 **单轮补全** 格式，与 d3LLM 官方 evalplus 基准、Dream-Coder instruct 的 **续写式** 解码习惯都不一致。模型在训练分布上学到的「怎么写代码」，和 HumanEval 评测时期望的「怎么接着 assistant 前缀往下写」是两套任务；再叠加当时代码提取对 EvalPlus 续写格式不兼容（部分样本 `pred=""` 直接判 0），`pass_reward` 和 acc 曲线都不可信，TPF baseline 也会被噪声样本污染。

因此在引入 TPF 效率项之前，我们先把 **训练 prompt、验证 prompt、离线 evalplus 评测** 统一到同一套格式，让 RL 信号与最终 pass@1 对照落在同一条语义链上。

### 2.1 对齐后的 prompt 长什么样

格式与 d3LLM `run_code_eval.sh` / evalplus provider 一致，每条样本变为 **两轮 chat**：

1. **user**：固定 instruction（`Please provide a self-contained Python script...`）+ 用普通 \`\`\` 围栏包住的 **task body**（题目描述，MBPP 还附带 `Your code should pass these tests:` 与 assert 行）。
2. **assistant 前缀**（预填在 parquet 里）：`Below is a Python script...` + 开头的 \`\`\`python\n`，rollout 从该前缀 **续写** 代码块。

`reward_model.ground_truth`（EvalPlus 测试用例）**不改**，判题仍走现有沙箱逻辑；变的是 **模型看到的输入形态**，与 evalplus 离线 benchmark 一致。

### 2.2 实现入口与生成流程

| 文件 | 作用 |
|------|------|
| `recipe/d3llm/evalplus_prompt.py` | 格式常量、HumanEval/MBPP/TACO 解析、`convert_row_to_evalplus` 单行转换 |
| `recipe/d3llm/build_evalplus_code_mix.py` | 批量生成训练混集与验证集，校验 prompt 结构，导出人工抽检 JSONL |

运行 `python recipe/d3llm/build_evalplus_code_mix.py`（或由 `run_bgpo_dream_coder_d3llm.sh` 在 parquet 缺失时自动调用）会产出：

- **训练混集** `data/preprocessed/rl/train/code_evalplus_mix_1.parquet`（811 行）：MBPP completion + TACO 代码围栏 completion + LCBv5 / PrimeIntellect / TACO 竞赛题（后三者各 cap 100），shuffle 后写入 `mix_bucket` 便于抽检。
- **验证集** `data/preprocessed/rl/test/humaneval_evalplus_1.parquet`（164 行）：HumanEval 转 EvalPlus 续写格式，替代旧的 `humaneval_1.parquet`。
- **抽检样例** `code_evalplus_mix_samples.jsonl`：按 bucket 抽样，方便肉眼核对 user / assistant 前缀 / ground_truth 是否对齐。

脚本内还有 `validate_parquet`：检查 user/assistant 角色、instruction 前缀、围栏闭合、MBPP assert 是否写入等，避免「格式对了但缺测例」的 silent bug。

### 2.3 MBPP 解析专项修复

旧 `mbpp_1.parquet` 里常见 **多条 user turn**，题目与 assert 分散在不同轮次；若只读第一条 user，task body 会缺测试或 entry point 解析错误，训练时 MBPP 桶的 pass_reward 系统性偏低。

修复策略（见 `evalplus_prompt.py` 中 `extract_mbpp_task_and_tests`）：

- **优先读最后一条 user**，用 `here is your task: ... Your code should pass these tests:` 正则一次性抽出 task 与 assert 行；
- 若正文缺 tests，再回退到 `reward_model.ground_truth` 补全 assert；
- 格式化为 EvalPlus 风格的 task body（题目 + `Your code should pass these tests:` + assert 列表），并从首条 assert 推断 `entry_point` 写入 `extra_info`。

这样 MBPP 训练样本与 d3LLM evalplus 跑 MBPP 时的 prompt 语义一致，混训里 MBPP 桶的梯度才代表真实「写函数过测例」能力，而不是格式错位带来的假阴性。

### 2.4 与代码提取、TPF 奖励的衔接

格式统一后仍需 **`extract_code_from_model`** 能正确处理「prompt 已含开围栏、response 只补尾部闭围栏」的续写输出；该 bug 修好后，smoke 上 step0 HumanEval 曾从 ~25% 回到与离线 evalplus 同量级。只有 pass/fail 判题可靠，TPF 效率项才有意义：

- **passed 样本** 的 TPF 才进入 baseline 累计；
- **失败样本** 若因 `pred=""` 被误判，会同时污染 acc 与 baseline。

因此数据集格式改造不是独立的数据工程，而是第三阶段 **「pass_reward 可信 → baseline 可信 → efficiency_reward 可信」** 的前置条件；全量脚本 `run_bgpo_dream_coder_d3llm.sh` 已默认指向上述 parquet，并与 `max_response_length=1024` 等配置一并使用。

---

## 3. 为什么用 TPF，而不是 NFE

- **NFE**（前向次数）是绝对量，受 `max_response_length`、block 完成度、早停等影响，**跨样本不可比**。
- **TPF**（tokens per forward）= 有效生成 token 数 / NFE，衡量 **每一步前向平均「固化」多少 token**，反映 multiblock 解码是否果断。
- 我们关心的是：**答对的前提下，能否用更少犹豫完成生成**——这是 TPF 要刻画的东西，而不是单纯压低 NFE。

---

## 4. 单样本 TPF 怎么算

对每个 rollout 样本：

1. **`rollout_nfe`**：该样本解码总前向次数（SGLang 多轮 `nfe` 列表时求和）。
2. **`rollout_gen_tokens`**：response 中有效 token 数（按 attention mask / EOS 计，非 pad）。
3. **`tpf`** = `rollout_gen_tokens / rollout_nfe`（任一为 0 则 TPF 记 0）。

Rollout 阶段把 `rollout_nfe`、`rollout_gen_tokens` 写入 batch，奖励管理器据此算 TPF。

---

## 5. Baseline TPF 怎么算

Baseline 表示 **「训练至今，所有通过样本的 TPF 算术平均」**，用作效率奖励的参照。

**更新规则（仅 passed 样本）：**

- 每个 batch 先算完所有样本的 `pass_reward`；
- 用 **本 batch 之前** 已累计的 baseline 计算 `efficiency_reward`（避免同 batch 样本互相污染参照）；
- 再将本 batch 中 **通过测试** 且 `tpf > 0` 的样本纳入累计：
  - `baseline = (Σ tpf_passed) / N_passed`
- **尚无 passed 样本时**，使用冷启动值 `tpf_baseline_initial`（默认 2.0）。

**为何用算术平均而非 EMA：** 实现更简单、含义直观（全局平均解码节奏），且少一个超参。EMA 对近期 batch 权重更大、适应更快，但会引入 `α` 调节，且在本场景下与算术平均收益相近；passed 子集本身已在 batch 间混合，算术平均足够稳定。

**为何 baseline 必须排除失败样本：** 失败样本往往 NFE 高、TPF 低。若纳入，会把参照拉低，让「低效但偶尔蒙对」的样本获得虚高 efficiency bonus。

---

## 6. `efficiency_reward` 的计算与设计理由

### 6.1 公式

对 **通过测试** 的样本（`passed=True` 且 `tpf > 0`）：

```
raw = coef × (tpf / baseline − 1)
efficiency_reward = clip(raw, −max_penalty, +max_bonus)
```

最终 **`reward = pass_reward + efficiency_reward`**。

### 6.2 设计理由

- **相对 baseline 的比率形式** `(tpf/baseline − 1)`：TPF 高于近期「答对样本的常态」则加分，低于则减分；随训练动态适应，不依赖绝对 NFE。
- **仅对 passed 样本生效**：正确性是硬约束；效率是 **在答对之后的二次优化**，避免模型用「更快但全错」换 efficiency 分。
- **线性 + 裁剪**：`coef` 控制塑形强度；`max_bonus` / `max_penalty` 防止 efficiency 项压过 0/1 的 `pass_reward`，保持 RL 信号主次分明。
- **验证不算 efficiency**：评测只看能力（acc / pass_reward），TPF 仅作 `val-aux` 监控，用于观察 acc 与效率是否同向变化。

直观理解：baseline 是「近期答对题目的平均解码节奏」；比它更果断（TPF 更高）给小幅奖励，更犹豫则小幅惩罚。

---

## 7. 超参作用

| 超参 | 默认 | 作用 |
|------|------|------|
| `enable_tpf_efficiency` | `True` | 总开关；关闭后等价于仅 pass/fail 奖励 |
| `tpf_efficiency_coef` | `0.1` | 效率项整体强度；越大 TPF 对 `reward` 影响越大 |
| `tpf_baseline_initial` | `2.0` | 冷启动 baseline，在尚无 passed 样本时使用 |
| `tpf_efficiency_max_bonus` | `0.25` | efficiency 加分上限 |
| `tpf_efficiency_max_penalty` | `0.25` | efficiency 减分下限（绝对值） |

**调参原则：** smoke 上 `|efficiency_reward/mean|` 应明显小于 `pass_reward/mean`（例如 0.006 vs 0.5）。若效率项过大，降低 `coef` 或收紧裁剪；若几乎不起作用，可适当提高 `coef`。全量训练应同时看 **acc 与 TPF**——acc 升而 TPF 持续跌，说明塑形不足或需配合 KL 锚定。

脚本入口：`recipe/dream/run_bgpo_dream_coder_d3llm.sh` 中 `reward_model.reward_kwargs.*`。

---

## 8. 联合监控（W&B）

训练与验证分别记录，便于对照「能力」与「效率」：

**训练（每 step）：** `pass_reward/mean`、`reward/mean`、`efficiency_reward/mean`、`tpf/mean`、`tpf_passed/mean`（仅 passed）、`tpf_baseline/mean`、`rollout_nfe/mean`。

**验证：** `val-core/humaneval/acc` 为核心指标；`val-aux/humaneval/pass_reward`、`tpf`、`rollout_nfe` 等为辅助，用于联合观察 acc 与解码效率是否一致。

---

## 9. 阶段验收与后续

- **Smoke 已通过**（`tpf_efficiency_coef=0.1` 下 efficiency 量级合理，监控指标齐全）。
- **待全量实验：** 在 EvalPlus 混训 + 本奖励方案下重训，对比 acc 曲线与 TPF 曲线；必要时尝试 `use_kl_loss` 抑制策略漂移。

```bash
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --smoke --engine sglang   # 验收
bash recipe/dream/run_bgpo_dream_coder_d3llm.sh --engine sglang          # 全量
```
