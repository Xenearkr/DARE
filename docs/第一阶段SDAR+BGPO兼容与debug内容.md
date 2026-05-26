# 自 `cb8ce6b` 以来的主要改动思路

> 基线：`cb8ce6b`（2026-05-24，`add env scripts`）  
> 当前：`8984846`（2026-05-26）  
> 区间共 6 个 commit，核心工作集中在 **让 SDAR-8B 的 BGPO 在 4×A6000 上用 SGLang 跑通且训得动**。

---

## 一句话

从「**8 卡 + 默认 LMDeploy 外挂推理**」转向「**4 卡整机协同 + SGLang 内嵌 rollout 为默认**」，并围绕 **采样正确性（GRPO 要有优势方差）** 和 **显存极限下的可训练性** 做了一轮端到端打通。

---

## 1. 部署假设变了：先适配 4×A6000，再谈加速

**基线**：Ray 按 8 GPU 起；`engine` 默认 `lmdeploy`（GPU0 起 API，其余卡训练）。

**现在**：

- 固定 `NUM_GPUS=4`，Ray 的 `--num-gpus` 与 `CUDA_VISIBLE_DEVICES` 在 **起 Ray 之前** 按 engine 分好。
- **SGLang**：4 卡都给训练+内嵌推理（无单独 lmdeploy 进程）。
- **LMDeploy**：仍保留「GPU0 服务 + 1–3 训练」三分法，但规模从 8 卡缩到 4 卡。
- 默认 engine 改为 **`sglang`**，LMDeploy 仍可通过 `--engine lmdeploy` 切换。

**思路**：不是换了个 backend 名字，而是把目标机器从「实验室 8 卡」收敛到「手头 4×A6000」，先保证拓扑和资源划分自洽，再堆功能。

---

## 2. 主线：把 SGLang 做成 SDAR rollout 的默认通路

### 2.1 运行时环境要对齐 SGLang，而不是照搬 PyTorch 训练习惯

- SGLang 的 `memory_saver` 与 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` **冲突** → 走 SGLang 时 **unset** 该变量，由 `run_bgpo_sdar_8b_chat.sh` 按 engine 分支设置；`dare_env.sh` 只保留注释说明。
- 为 Ray worker 补 `CUDA_HOME` / `LD_LIBRARY_PATH`，降低子进程里 Triton/CUDA 链不到库的概率。
- 引入 `third_party/sglang` 子模块，便于固定版本、本地改 dLLM 解码逻辑。

### 2.2 推理侧显存策略：宁可慢一点，也要在 4 卡上起得来

全量 SGLang 训练路径采用偏保守的推理配置（与 smoke 类似）：

- 降低 `mem_fraction_static` / `gpu_memory_utilization`
- `attention_backend=torch_native`，`disable_cuda_graph=True`（避免 fa3/flashinfer + cuda graph 在 Ray 里踩 Triton 链接等问题）
- 训练侧打开 **`enable_activation_offload`**，用激活 offload 换 actor 显存

**思路**：4 卡同时要扛 **FSDP actor + 内嵌 SGLang Engine**，不能沿用 8 卡 + 外挂 lmdeploy 时的显存预算；先 **降推理峰值、offload 训练激活**，再调 batch。

### 2.3 可重复的「最小闭环」：smoke 模式

- `--smoke`：缩短数据/序列、减小 batch、更激进的 SGLang 内存参数，用于 **验证环境能起、能 rollout、能反传**，而不是跑完整 code 任务。

**思路**：大脚本拆不出问题时，用 smoke 把「引擎 / Ray / 权重同步」和「完整 RL 超参」解耦。

---

## 3. 正确性：RL 需要「会随机」的 rollout，不能和 LMDeploy 各说各话

这是区间里 **最有算法含义** 的改动，不是小修补。

### 3.1 问题

SGLang 自带的 `LowConfidence` dLLM 解码偏 **argmax / 确定性**，同一 prompt 多次 rollout 几乎相同 → **GRPO 组内 advantage 方差为 0**，训练名义上在跑、实际上没有对比信号。

LMDeploy 侧用的是 `low_confidence_dynamic` + **temperature 采样**，行为不一致。

### 3.2 对策（两层）

1. **`sglang_sdar_dllm_patch.py`**：在 verl 侧 monkey-patch `LowConfidence`，实现与 LMDeploy 对齐的 **动态阈值 + temperature/top-k/top-p 采样**；用 `contextvars` 传递每次 generate 的采样参数。
2. **`SGLangSDARRollout` 行为对齐 LMDeploy**：
   - 通过 `dllm_algorithm_config`（YAML）把 `threshold`、`denoising_steps`、`temperature`、`top_p` 传给引擎子进程；
   - 训练时走 **`_lmdeploy_style_batch_level_generate_sequences`**：每个样本 **独立 `n=1` 请求 + 随机 `sampling_seed`**，而不是一次 batch `n>1` 绑死随机性。

3. **`third_party/sglang` 内 `low_confidence.py`**：同步落地 temperature / top-k / top-p 与 per-request seed，与上游 patch 思路一致。

**思路**：加速 backend 可以换，但 **「rollout 分布」必须与 BGPO/GRPO 假设一致**；先对齐 LMDeploy 的随机解码语义，再谈谁更快。

### 3.3 训练算法侧一个实质 bugfix

`core_algos.py` 里 GRPO 组内标准差：`torch.std(torch.tensor([id2score[idx]]))` → `torch.std(torch.tensor(id2score[idx]))`。  
前者几乎恒为 0，会进一步 **压扁 advantage**；与 rollout 随机性问题是同一类「训练信号被抹平」问题。

---

## 4. 可训练性：actor 在 4 卡上扛不住时，改更新粒度而不是只调 batch

**`sdar_dp_actor_bgpo.py`（`update_policy`）**：

- BGPO 的 MC 维度上，从「整 micro-batch 一次 forward/backward」改为 **按样本 × MC 逐条 backward**；
- 步骤间 `empty_cache()`，metrics 按 `mc_num * num_samples` 归一化。

**思路**：8 卡或 LMDeploy 外挂时显存更宽裕；SGLang 内嵌后 actor 与 rollout **抢同机显存**，用 **更细的更新粒度换峰值显存**，保证梯度仍覆盖所有 MC 样本。

---

## 5. 工程化与后续方向（次要但有意图）

| 改动 | 意图 |
|------|------|
| 实验名加入 `gpu{N}`、`smoke` 标签、batch 整除校验 | 日志可区分 4 卡 / smoke / 全量，减少 silent misconfig |
| `wandb` 改 online、调 `test_freq` | 运维偏好，不改变训练架构 |
| `.gitignore` + opencompass `__init__.py` 小改 | 为接入 **d3LLM** 清路径（commit 信息：`before adding d3llm`） |
| 误提交的 `wandb/` 运行目录 | 应视为噪音，不宜当作设计方向 |

---

## 6. 未改动的共识（便于对照）

- **LMDeploy 路径仍保留**，三分 GPU + `api_server` 流程未删；只是默认与优化重心转到 SGLang。
- **算法仍是 BGPO**；动的是 rollout 引擎、采样语义、显存与更新策略，不是换 RL 目标。
- **LLaDA / Dream 等其它模型族** 不在此区间的重点里；改动几乎全落在 **SDAR + `run_bgpo_sdar_8b_chat.sh` + SGLang rollout/actor**。

---

## 时间线（按 commit 意图）

```
cb8ce6b  环境脚本基线（8 卡、默认 lmdeploy）
    ↓
80053f9  4×A6000 + 默认 sglang + Ray/显存环境对齐 + smoke
    ↓
cc6a9e4  采样语义对齐 LMDeploy（patch + rollout + actor 显存）
    ↓
9d401a0  SGLang rollout 调试收尾（generate 路径整理）
    ↓
8984846  d3LLM 接入前的仓库卫生（gitignore 等）
```

---

## 若只记住三件事

1. **机器与默认引擎**：8 卡 lmdeploy 外挂 → **4 卡 sglang 内嵌**。  
2. **算法信号**：rollout 必须 **随机且与 LMDeploy 同分布**，否则 GRPO/BGPO 没有有效优势。  
3. **资源策略**：推理降配 + activation offload + actor 逐样本反传，是在 **同一套 4 卡** 上同时放下训练和 SGLang Engine 的代价。
