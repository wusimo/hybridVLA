# HippoVLA 实验设计与改动清单

> 本文档基于实际代码（`transformer/src/transformers/modeling_memory.py`、`modeling_qwen3_vl.py` 的 `Qwen3VLModel`、`starVLA/model/framework/RynnBrainOFT.py`、`starVLA/dataloader/gr00t_lerobot/datasets_2.py`）撰写，是论文 §Method 与 §Experiments 的工程对照。
> 旧版 `memory_experiment_design.md`（MicroMem 阶段）保留作为历史参照；本文件为面向投稿的 **HippoVLA** 版本。

---

## 0. 整体定位

HippoVLA 在工程上不重写 VLA pipeline，而是在三个层次上对现有 RynnBrain-OFT 框架做最小侵入式扩展：

1. **数据层**：以稀疏间隔回溯历史帧（theta 节律采样），把"过去"作为独立字段而非拼到 prompt。
2. **VLM 内部**：在 Qwen3-VL 的 DeepStack 视觉栈里挂一个固定容量的记忆库 `ShortTermMemoryBank`（即 sTMB），**只改视觉特征，不改 LLM 上下文**。
3. **方法创新**：扩展 sTMB 的写门为 **空间显著性感知门控**（Spatial-Saliency-Aware Gate, SSAG），由 RynnBrain 的空间通路反向决定哪些 token 值得入库。这是论文中可发表的核心架构贡献。

整体上 HippoVLA = `RynnBrain-8B 空间主干（𝒩）` + `sTMB 时间通路（ℋ）` + `SSAG 显著性门控（𝒩→ℋ 的耦合）`。

---

## 1. Framework 改动总览（与基线 VLA 的差异）

| 层 | 文件 | 基线行为 | HippoVLA 行为 | 状态 |
|---|---|---|---|---|
| dataloader | `gr00t_lerobot/datasets_2.py:1535,1693-1766` | 单帧观测 `image=[view0,view1]` | 额外回溯 `[t-5I, …, t-I]` 共 5 帧；不足时 clamp 到 frame 0；输出新增 `sample["memory"][5][V]`、`sample["step"]` | ✅ 已实现 |
| framework (训练) | `framework/RynnBrainOFT.py:forward` | 调 `build_rynnbrain_inputs(images, instructions)` | `if memory_mode:` 改调 `build_rynnbrain_inputs_with_memorys(images, instructions, memorys, steps)` | ✅ |
| framework (推理) | `RynnBrainOFT.predict_action` | 同上 | 同训练；额外用 `to_pil_preserve` 把 msgpack 传回的 numpy 还原为 PIL | ✅（待跑 `[MEM PROBE]` 验证） |
| vlm_interface | `modules/vlm/RynnBrain.py:build_rynnbrain_inputs_with_memorys` | 不存在 | flatten 历史图 → `image_processor` → 写 `pixel_values_mem / memorys_length / steps` 入 `batch_inputs`；返回普通 LM 输入 | ✅ |
| modeling | `transformers/models/qwen3_vl/modeling_qwen3_vl.py:Qwen3VLModel` | DeepStack 三层视觉特征直接送 LLM | `memory_mode=True` 时，先用 `get_memory_features` 编码历史帧 → reshape 为 `memory_tensor[B,3,T,2,64,D]` 与 `visual_tensor[B,3,2,64,D]` → 调 `self.memory(...)` 得到更新后的 `visual_tensor` → 写回 DeepStack | ✅ |
| modeling | `transformers/modeling_memory.py:ShortTermMemoryBank` | 不存在 | 3 level × 2 view × 64 slot × D 的记忆库；正弦绝对时间编码；逐 level 独立 read-attn + sigmoid gate | ✅（**待加 SSAG**） |
| eval | `examples/LIBERO/eval_files/eval_libero.py` | 单帧 | 镜像训练采样窗 `[t-5I, …, t-I]`；当前帧不入 memory | ✅ |

**与 PAM / MemoryVLA 的关键差异**：

- 历史不以 token 形式拼到 LLM 上下文（不增加 `seq_len`），仅介入视觉栈 → 时间复杂度不随历史长度二次膨胀。
- 写入受门控控制（默认 sigmoid，**HippoVLA 升级为空间显著性条件化**）→ 全局替换 vs 选择性替换的关键区别。

---

## 2. sTMB 模块设计（核心）

### 2.1 数据张量约定

输入：

| 张量 | 形状 | 来源 |
|---|---|---|
| `memory` | `[B, 3, T, V, S, D]` | DeepStack 三个层级、T=5 步历史、V=2 视角、S=64 slot、D=2560 |
| `visual` | `[B, 3, V, S, D]` | DeepStack 三个层级的当前帧特征 |
| `timestep` | `[B]` int 或 scalar | 数据集中的 `base_index`，作为绝对时间 |

输出：

| 张量 | 形状 | 写回位置 |
|---|---|---|
| `new_memory` | `[B, 3, V, S, D]` | 替换 `Qwen3VLModel` 中 DeepStack 的 visual_tensor，再走原 LLM forward |

> 这意味着 sTMB 的代价 = **每个 forward 多算一次跨层注意力 + 一次 64-slot 门控**，不改变后续 LLM 的序列长度。

### 2.2 当前实现（已在代码中）

`modeling_memory.py:ShortTermMemoryBank.forward`：

```
# 对每个 DeepStack level ∈ {0,1,2}，独立处理：
level_visual_t  = level_visual + sinusoidal_PE(timestep)            # 时间锚定
last_memory     = level_memory[:, -1]                                # [B,V,S,D]
memory_per_view = level_memory.permute(0,2,1,3,4).reshape(B,V,T*S,D) # 拍平 T 步
history_agg     = write_attn(last_memory, memory_per_view)           # 跨注意力聚合
gate            = sigmoid( W_g [history_agg ; level_visual_t] )      # ← 仅数据驱动门
level_new       = gate * level_visual_t + (1-gate) * history_agg
level_new       = RMSNorm(level_new)
```

**已实现的脑科学对应**：
- `num_slots=64`、`S << H·W`：齿状回 **模式分离**（pattern separation）
- `interval=10` 的稀疏采样（在 dataloader 端实现）：海马 **theta--gamma 离散编码节律**
- 正弦时间编码 `sinusoidal_PE(timestep)`：海马 **时间细胞**（time cells）

**论文需要的、目前缺失的核心创新**：当前 gate 是单纯数据驱动的 GRU 风格门，没有任何来自空间通路的信号 —— 这意味着"复用 RynnBrain"和"sTMB"是松耦合的。要让论文站住，必须把 SSAG 实装。

### 2.3 提议改动：空间显著性感知门控（SSAG）

把当前公式改为：

```
g_t  = sigmoid( W_g [history_agg ; level_visual_t] )    ⊙   φ(s_t)
M_t  = g_t * level_visual_t + (1 - g_t) * history_agg
```

其中 `s_t ∈ [0,1]^S` 是 **逐 slot 的空间显著性**，三个候选定义（按从易到难排序）：

| 方案 | `s_t` 的来源 | 工程改动量 | 论文卖点 |
|---|---|---|---|
| **S1**（推荐 v1） | RynnBrain DeepStack level 2 的特征范数：`s_t = ||V_t^{(2)}|| / max(||V_t^{(2)}||)` | 最小：在 `Qwen3VLModel.forward` 里多算一行 | 可解释性弱，但能正面对比 sigmoid-only |
| **S2**（推荐 v2） | RynnBrain `[PT]` token（ChainOfPoint 的空间 token）对每个 slot 的 attention rollout | 中等：需要 hook ChainOfPoint head | 强：直接把 RynnBrain 的"空间注意"翻译成"记忆开关" |
| **S3**（远期） | 一个轻量级 saliency head 与动作 loss 联合训练 | 大：新增可学习头 | 学习式，效果上限最高，但消融变多 |

**论文里把 SSAG 写成 `g ⊙ φ(s_t)` 的统一形式，附录里给三种 `s_t` 实例**——既是方法贡献，又是天然 ablation。

### 2.4 sTMB 的 5 条设计原则（论文 §3.2 / Discussion 用）

| ID | 原则 | 现实代码体现 | 违反它的反例（用作 ablation） |
|---|---|---|---|
| **P1** | 介入感知（视觉特征级），不进语言上下文 | `Qwen3VLModel` 在视觉栈替换 visual_tensor | A-late-injection：把记忆挂到 LLM 输出 |
| **P2** | 容量隔离：固定 `K=64` slot，与序列长度解耦 | `num_slots=64` | A-token-inject：把 memory 拼成 token 进 input_ids（MemoryVLA 风格） |
| **P3** | 稀疏时间采样（theta 节律） | dataloader `interval=10`、`max_step=5` | A-dense-history：`interval=1, max_step=30` |
| **P4** | 选择性写入（不全局替换） | 门控融合（sigmoid + SSAG） | A-no-gate：直接 `visual = visual + history` |
| **P5** | 显式时间锚定 | 正弦绝对时间编码 | A-no-time-emb：去掉 PE_time |

---

## 3. 实验设计

### 3.1 主表 — LIBERO 4-in-1 + CALVIN ABC→D

| Config | memory_mode | max_step | interval | gate | 说明 | 状态 |
|---|---|---|---|---|---|---|
| **Baseline** | False | — | — | — | 当前 RynnBrain-8B-OFT | ✅ 已跑（CALVIN 3.70 / LIBERO 见 tab2） |
| **HippoVLA-base** (M5/I10 + sigmoid) | True | 5 | 10 | sigmoid | 现有 0420 权重 | ✅ 已跑 |
| **HippoVLA-base** (M5/I5) | True | 5 | 5 | sigmoid | 间隔扫描点 | ✅ 已跑（LIBERO） |
| **HippoVLA-S1** | True | 5 | 10 | sigmoid + ‖V‖ | SSAG-S1 | ⏳ 待跑（核心新结果） |
| **HippoVLA-S2** | True | 5 | 10 | sigmoid + [PT]-rollout | SSAG-S2 | ⏳ 待跑（论文最强卖点） |

### 3.2 P1–P5 消融（Method 章节配套）

| Ablation | 改动 | 违反 | 期望 |
|---|---|---|---|
| A-token-inject | memory 拼到 input_ids | P2 | 显存涨、SR 下降 |
| A-no-gate | 直接相加 | P4 | SR 显著下降，长程任务尤甚 |
| A-no-time-emb | 关闭 sinusoidal PE | P5 | LIBERO-Long / CALVIN 下降 |
| A-dense-history | `interval=1, max_step=30` | P3 | 因果混淆，SR 不升或下降 |
| A-late-injection | memory 挂在 LLM 输出 | P1 | 全面下降 |
| **A-shuffled-saliency** | SSAG 的 `s_t` 跨帧打乱 | SSAG content-check | 退化为 sigmoid-only，证明 `s_t` 承载真实信号 |

### 3.3 间隔与槽数曲线

| 扫描 | 维度 | 点数 |
|---|---|---|
| τ-curve | `interval ∈ {1, 3, 5, 10, 20}`，固定 `max_step=5` | 5 |
| K-curve | `num_slots ∈ {8, 16, 32, 64, 128}`，固定 `interval=10` | 5 |
| T-curve | `max_step ∈ {1, 3, 5, 10}`，固定 `interval=10` | 4 |

每条曲线只跑 LIBERO-Long + CALVIN-Avg-Len 这两个最敏感的指标即可，避免训练量爆炸。

### 3.4 显著性可视化（论文图）

- **图 1**：架构图（双通路 + SSAG 流向，论文 §3.1）
- **图 2**：对一段 LIBERO-Long rollout，沿时间轴可视化每帧的 `mean(s_t)` 与门控开关状态 → 证明 SSAG 在物体出现 / 接触瞬间打开、在转场闲置时关闭
- **图 3**：对 64 个 slot 做 PCA 或 probing —— 探针出"物体身份"slot 与"夹爪—物体相对位姿"slot 的分工（pattern separation 的可视证据）
- **图 4**：τ-curve / K-curve / T-curve 三合一

### 3.5 静默失败诊断（继承 MicroMem 文档的 §4.4）

仅作为附录健全性检查，确认评测代码没有把 memory 用错：

| 失败模式 | 注入 | 期望 |
|---|---|---|
| F1 训练-评测窗错位 | 评测 memory 偏移 1·interval | SR 显著下降，说明窗口对齐有用 |
| F2 零图像填充 | 开局用全 0 而非 clamp 到 frame 0 | 开局几步异常 |
| F3 写-查询自指 | 当前帧入 memory 末位 | 退化为 baseline |
| F4 因果泄漏代理 | memory 中泄漏前一帧动作 | 训练 loss 急降但评测 SR 下降 |

---

## 4. 必须先修的 3 个 Bug（在跑论文实验前）

| ID | 位置 | 问题 | 修复 |
|---|---|---|---|
| **B1** | `modeling_qwen3_vl.py:904` | `ShortTermMemoryBank(num_timesteps=5)` 写死，`config.framework.qwenvl.max_memory_step` 未生效 | 把 `max_memory_step` 经 `_RynnBrain_Interface.from_pretrained → Qwen3VLModel.__init__` 传入 |
| **B2** | `modeling_qwen3_vl.py:1268-1269` + `modeling_memory.py:138` | `num_views=2` 硬编码 | 单视角任务再抽象（低优） |
| **B3** | `RynnBrain.py:build_rynnbrain_inputs_with_memorys` | memory 的 `image_grid_thw` 被丢弃，下游复用主图 grid | 写入 `batch_inputs['memorys_grid_thw']`，模型侧优先读它 |

B1 / B3 必修；B2 等出现单视角需求再说。

---

## 5. 参数 / 时延开销（论文 §3.3 的 `\todo{}` 用）

需要在单卡 A100 上量化：

```bash
# 1) 可训练参数差值 = sTMB + SSAG 额外参数
# 2) per-step 推理 wall-clock (bs=1, chunk_len=8)
# 3) 显存峰值 (bs=16, T=5)
```

预期数量级：

- sTMB 额外参数：≈ 3 × (3·D² + D·1)·M_levels ≈ 60–80M（D=2560 时），约占 RynnBrain-8B 的 < 1%
- SSAG 额外参数：< 1M（小 MLP）
- 推理时延：相对 baseline +5–10%（每 forward 多一次跨注意力）

---

## 6. 实验产物（与论文表 / 图对应）

| 论文位置 | 实验产物 |
|---|---|
| Tab. 1 (memory ablation) | §3.1 的 5 行 + SSAG 行 |
| Tab. 2 (cross-backbone) | 已存在，补 HippoVLA-S2 一行 |
| Tab. 3 (CALVIN leaderboard) | 已存在，可补 HippoVLA-S2 行 |
| Tab. 4 (P1–P5 ablation) | §3.2 |
| Fig. 2 (saliency timeline) | §3.4 图 2 |
| Fig. 3 (slot probing) | §3.4 图 3 |
| Fig. 4 (τ/K/T 曲线) | §3.3 |
| Appendix A (silent failure) | §3.5 |
| §3.3 的开销 `\todo{}` | §5 |

---

## 7. 推荐执行顺序

1. **修 B1 / B3**（半天）。
2. **实装 SSAG-S1**（特征范数版，半天）— 跑 LIBERO 4-in-1，先验证方向正确。
3. **实装 SSAG-S2**（[PT]-rollout 版，1–2 天）— 这是论文核心。
4. 跑 §3.2 五条 P1–P5 ablation（每条 ≈ 20k step；可并行 5 卡 / 5 任务）。
5. 跑 §3.3 三条曲线（共 14 个点，可裁剪到 LIBERO-Long 单一指标降负担）。
6. 出 §3.4 的可视化（基于已训练好的 HippoVLA-S2 权重，只是分析）。
7. CALVIN 上跑 HippoVLA-S2，更新 leaderboard 那一行。
8. 测开销，填论文 §3.3 的 `\todo{}`。

---

## 8. 与旧 `memory_experiment_design.md` 的差异

| 维度 | 旧（MicroMem） | 新（HippoVLA） |
|---|---|---|
| 命名 | MicroMem | **HippoVLA** / **sTMB** / **SSAG** |
| 核心叙事 | "极简记忆适配器对抗视觉混淆" | "类海马体的时空解耦双通路 + 空间显著性门控" |
| 方法贡献 | sTMB 本体 | sTMB + **SSAG**（这是真正的可发表创新） |
| 主基线对照 | MicroMem vs Baseline | HippoVLA(SSAG) vs HippoVLA(sigmoid) vs Baseline |
| 论文挂靠 | LIBERO 主导 | LIBERO + CALVIN 双主导 |
| 是否保留 | 作为历史与 silent failure 检查参考 | 本文档为投稿版主清单 |
