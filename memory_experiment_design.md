# MicroMem 实验设计与改动清单

## 0. 目标

验证"单帧 VLA 策略在视觉观测混淆(Visual Observation Aliasing)下会出现模式塌缩"这一假设，并定量证明极简特征级短期记忆适配器 **MicroMem** 能够以极小的参数/计算代价恢复被混淆吃掉的性能。

---

## 1. 代码改动总览

改动范围分 4 层：**dataloader → framework → vlm_interface → modeling**。

| 层 | 文件 | 改动内容 |
|---|---|---|
| dataloader | `starVLA/dataloader/gr00t_lerobot/datasets.py` | 按 `(max_step=5, interval=10)` 回溯历史帧；叶子为 PIL 224×224；输出 `sample["memory"][5][num_views]` + `sample["step"]` |
| framework (训练) | `starVLA/model/framework/RynnBrainOFT.py` `forward` | 根据 `memory_mode` 调用 `build_rynnbrain_inputs_with_memorys`；正确透传 `memorys/steps` |
| framework (推理) | `RynnBrainOFT.predict_action` | 同上；额外 `to_pil_preserve` 将 msgpack 回传的 numpy 恢复成 PIL；已插入 `[MEM PROBE]` 日志 |
| vlm_interface | `starVLA/model/modules/vlm/RynnBrain.py` `build_rynnbrain_inputs_with_memorys` | flatten memory 图像 → `image_processor` → 将 `pixel_values_mem` / `memorys_length` / `steps` 写入 batch_inputs |
| modeling | `transformer/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py` `Qwen3VLModel` | 构造 `memory_tensor [B,3,T,2,64,D]` 与 `visual_tensor [B,3,2,64,D]`；调用 `ShortTermMemoryBank` 更新后写回 DeepStack |
| modeling | `transformer/src/transformers/modeling_memory.py` `ShortTermMemoryBank` | 3 level × per-level read-attn + 门控融合 + 正弦绝对时间编码 |
| eval | `examples/LIBERO/eval_files/eval_libero.py` | 镜像训练的 `[t-5I, …, t-I]` 采样窗；numpy over msgpack |

---

## 2. 已验证通过的数据流节点 (Step A + B + 静态 Step C)

| 节点 | 形状/值 | 状态 |
|---|---|---|
| dataloader 输出 `memory` | `[max_step=5][num_views=2]` PIL 224×224 | ✅ 实跑验证 |
| dataloader 输出 `step` | `base_index` (整数) | ✅ |
| processor packed patches | `[B·5·2·256, 1536]`，每图 `grid_thw=(1,16,16)` | ✅ 实跑验证 |
| 主图像 grid vs memory grid | 均为 `(1,16,16)` | ✅ 实跑验证 |
| 模型侧接收 `memorys/memorys_length/steps` | `kwargs[...]` 正确接收 | ✅ 静态验证 |
| `ShortTermMemoryBank` 更新 `visual_tensor` 后写回 DeepStack | `deepstack_visual_embeds` 被替换 | ✅ 静态验证 |
| 评测侧 numpy→PIL 保持嵌套结构 | `to_pil_preserve` 递归处理 | ✅ 静态验证 |
| 训练 vs 评测的 memory 窗口对齐 | 都是 `[t-5I, …, t-I]`，当前帧不入 memory | ✅ 代码对照 |

---

## 3. 发现的 Bug / 隐性耦合

| 编号 | 位置 | 描述 | 当前影响 | 建议动作 |
|---|---|---|---|---|
| B1 | `modeling_qwen3_vl.py:904` | `ShortTermMemoryBank(num_timesteps=5)` 硬编码，忽略 `config.framework.qwenvl.max_memory_step` | 仅当改动配置值才会炸；当前 `max_memory_step=5` 所以无问题 | **修**：把 `max_memory_step` 从 `_RynnBrain_Interface` 传进 `Qwen3VLModel` |
| B2 | `modeling_qwen3_vl.py:1268-1269` + `modeling_memory.py:138` | `num_views=2` 硬编码 | 单视角任务必炸 | 低优：单视角任务出现时再抽象 |
| B3 | `RynnBrain.py:build_rynnbrain_inputs_with_memorys` | memory 的 `image_grid_thw` 被丢弃；模型侧复用主图 grid | 只要 memory 图分辨率 ≠ 主图就错位 | **修**：把 memory 的 `image_grid_thw` 也写入 `batch_inputs` |
| C1 | 评测 `step` 分布 | 评测开局 `step=0`，但训练数据集中 `base_index=0` 占比极低 | 开局前几步时间编码 OOD | 诊断 + 可选随机 offset |
| C2 | 评测开局 memory 内容 | 前 5 步 memory 全部退化为当前帧重复 | 训练数据中也有类似退化(clamp 到 frame 0) 所以覆盖 | 仅记录，不改 |

---

## 4. 实验设计

### 4.1 主实验：LIBERO 五套件 + MicroMem 开关消融

| Config | 说明 | 期望指标 |
|---|---|---|
| **Baseline** | `framework.qwenvl.memory=False`，纯单帧 RynnBrain-OFT | 现有论文基线 |
| **MicroMem (ours)** | `memory=True`, `max_step=5`, `interval=10` | 相对 Baseline 有一致提升 |
| **Dense-300** | `max_step=30, interval=1`（PAM 风格密集历史） | 预期不升甚至下降（因果混淆） |
| **Sparse-3** | `max_step=3, interval=10` | 预期略低于 M=5 |
| **Sparse-10** | `max_step=10, interval=10` | 预期与 M=5 接近，增量递减 |

均在 `libero_spatial / object / goal / 10 / 90` 上评估 ≥50 ep/task。

### 4.2 P1–P5 设计约束的消融

针对论文 §3.2 的五条约束各做一组 ablation：

| Ablation | 改动 | 对应违反约束 | 期望结果 |
|---|---|---|---|
| **A-token-inject** | 把 memory 以 token 形式拼到 input_ids（MemoryVLA 风格） | 违反 P2 | 下降或持平；序列长度显著增加 |
| **A-no-gate** | 去掉 sigmoid 门控，直接 `visual = visual + history` | 违反 P4 | 下降；全局替换破坏未受混淆的 token |
| **A-no-time-emb** | 去掉 `PE_time` | 违反 P5 | 在有视觉循环的任务(如 libero_10)下降明显 |
| **A-dense-history** | `interval=1, max_step=30` | 违反 P3 | 因果混淆显现（见 Dense-300） |
| **A-late-injection** | 把 memory 改挂在语言模型输出而非视觉端 | 违反 P1 | 下降，证明感知层介入的必要性 |

### 4.3 混淆分数 (Aliasing Score) 与性能相关性

计算 LIBERO 各 suite 的 `AS(𝒟)`（Def 1.1），将 MicroMem 相对 Baseline 的提升 $\Delta$ 对 `AS` 做 Pearson/Spearman 相关分析。

- **假设**：`Corr(ΔSR, AS) > 0.6`，即混淆严重的 suite 提升更大。
- **产物**：一张 AS vs ΔSR 的散点图。

### 4.4 Silent 失败模式验证

为论文 §4 的 F1–F4 各做一个最小对照实验，证明"静默失败"会让 memory 看起来无用：

| 失败模式 | 注入方式 | 期望验证 |
|---|---|---|
| **F1 训练-评测帧错位** | 人为让评测 memory 窗口偏移 1 个 interval | SR 显著下降 |
| **F2 零图像填充** | 评测开局用全 0 图像填充 memory 而非 clamp 到 frame 0 | 开局几步动作异常 |
| **F3 写-查询自指** | 将当前帧也放入 memory bank 末位再做 read | SR 接近 Baseline（memory 退化为恒等） |
| **F4 因果混淆代理** | 在 memory 里泄露前一时刻的动作(例如上一帧 overlay 动作向量) | 训练 loss 下降更快但评测 SR 下降 |

---

## 5. 参数 & 计算开销测量

在单卡 A100 上跑以下探针：

```bash
# 记录两项数值
python - <<'PY'
from starVLA.model.framework.RynnBrainOFT import RynnBrain_OFT
# ... 构建带/不带 memory 的模型
# 1) 可训练参数差值 = MicroMem 额外参数
# 2) forward wall-clock: per-step 毫秒数 (bs=1, chunk_len=8)
PY
```

预期报告三个数：
- 额外可训练参数量（M）
- 额外参数占主干比例（%）
- 额外每步推理延迟（ms / %）

---

## 6. 预研诊断脚本（已就绪，保留复用）

| 脚本 | 用途 | 当前状态 |
|---|---|---|
| `examples/LIBERO/train_files/test_memory_AB.py` | 验证 dataloader 输出 + processor 拼接 + grid 一致性 | ✅ 已跑通 |
| `RynnBrainOFT.predict_action` 内 `[MEM PROBE]` | 打印评测真实进入模型的 memory shape/steps | ⏳ 未实跑 |

---

## 7. 待办 (顺序执行)

1. **修 B1**：让 `max_memory_step` 配置真正生效（1 行改动 `from_pretrained(memory_mode=..., max_memory_step=...)` + `Qwen3VLModel.__init__` 接收）
2. **修 B3**：在 `build_rynnbrain_inputs_with_memorys` 里额外写 `batch_inputs['memorys_grid_thw']`，模型侧优先使用该字段
3. 跑一次 libero_spatial 评测，收集 `[MEM PROBE]` 日志，确认推理侧形状完全对齐训练
4. 实现 4.1 主实验五组 config，训练 + 评测
5. 实现 4.2 五组 ablation
6. 计算 4.3 的 AS 并做相关性分析
7. 完成 4.4 的四组 silent failure 对照
8. 参数/延迟测量填入论文 §3.3 的 `\todo{}` 占位

---

## 8. 实验产物(论文所需)

- **表 1**：LIBERO 主表 (五 suite × 五 config 的 SR)
- **表 2**：P1–P5 消融表
- **表 3**：参数/延迟/序列长度对比 (vs MemoryVLA / PAM / Baseline)
- **图 1**：AS vs ΔSR 散点 + 回归线
- **图 2**：F1–F4 silent failure 对照柱状图
- **附录**：`test_memory_AB.py` 的实测输出 + `[MEM PROBE]` 日志
