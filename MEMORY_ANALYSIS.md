# starVLA Memory Module: Compatibility Analysis & Debug Guide

## 1. The Showstopper: Memory Is NOT Used During Evaluation

**This is almost certainly why you see no improvement.**

During LIBERO evaluation, `model2libero_interface.py` creates a `ModelClient` with `horizon=0` (default). The `step()` method sends each timestep as a standalone example with **no `memory` key** in the dict. When `predict_action()` runs:

```python
if self.memory_mode:
    memorys = [example.get('memory', []) for example in examples]  # gets [] every time
```

So `memorys = [[]]`. Then in `build_qwenvl_inputs_with_memorys`:

```python
if memorys and any(memorys):  # any([[]]) = False!
    ...
else:
    batch_inputs['memorys'] = [torch.empty(0, 2, 64, D, ...) for _ in range(B)]
```

The memory bank receives `T=0` tensors, and its forward does:

```python
if T == 0:
    return visual  # passthrough, memory does nothing
```

**Result:** The model trains with memory but evaluates without it. The memory module is a no-op at test time.

---

## 2. Memory Length vs LIBERO Task Length: Incompatible

### What the memory covers

| Parameter | Value |
|-----------|-------|
| max_memory_steps | 5 frames |
| interval | 5 timesteps |
| Total lookback | 25 timesteps |

### What LIBERO tasks need

| Task Suite | Max Episode Steps | Typical completion |
|------------|-------------------|-------------------|
| libero_spatial | 220 | ~100-150 |
| libero_object | 280 | ~120-180 |
| libero_goal | 300 | ~150-250 |
| libero_10 | 520 | ~200-400 |
| libero_90 | 400 | ~150-350 |

**25 steps of lookback covers only ~10-15% of a typical LIBERO episode.** For long-horizon tasks (libero_10, libero_90), the robot needs to remember what happened 100-300 steps ago (e.g., "I already picked up the mug and placed it on the shelf, now I need to close the drawer"). A 25-step window cannot capture this.

### Why this matters for long-horizon tasks specifically

Long-horizon LIBERO tasks involve sequential sub-goals (e.g., "pick up the bowl, put it in the microwave, close the microwave"). The robot needs to remember which sub-goals are complete. With 25 steps of lookback, the memory only sees the current sub-goal's recent actions — it has no information about earlier completed sub-goals.

---

## 3. Bugs in the Memory Integration

### Bug 1: Hardcoded `timestep=5`

```python
# modeling_qwen3_vl.py:1315
deepstack_visual_embeds[i][j] = self.memory(visual_memorys[j], deepstack_visual_embeds[i][j], timestep=5)
```

Every frame gets `timestep=5`, making the sinusoidal temporal encoding constant. The memory cannot distinguish "this happened 5 steps ago" from "this happened 25 steps ago." The temporal embedding is useless.

### Bug 2: Same memory applied to every deepstack layer

The loop iterates `i` over deepstack layers (typically 4-8 layers). Each layer's visual embeddings go through the **same single `self.memory` module** with the **same `visual_memorys[j]`** input. This means:
- Every deepstack layer gets the same memory-augmented features
- The gate learns a single blend ratio applied identically across all depth levels
- Earlier layers (low-level features) and later layers (high-level features) get no differentiation

### Bug 3: `torch.no_grad()` on memory image encoding

In `QWen3.py:333`:
```python
with torch.no_grad():
    vision_output = self.model.visual(pixel_values_mem, grid_thw=image_grid_thw_mem)
```

The memory features are frozen. The vision encoder never learns to produce better representations for historical frames. Only the memory bank's attention/gate parameters get gradients.

### Bug 4: Hardcoded 64 tokens per view assumption

```python
feat = memory_features_flat[idx : idx + X_i * 2 * 64]  # QWen3.py:349
feat = feat.view(X_i, 2, 64, D)
```

With 224x224 images and patch_size=14, you get 16x16=256 patches. After Qwen3-VL's 2x2 spatial merge, that's 64 tokens. But this hardcoding breaks if image resolution changes.

### Bug 5: In-place tensor modification in autograd graph

```python
deepstack_visual_embeds[i][j] = self.memory(...)  # slice assignment
```

This modifies a tensor view in-place, which can cause autograd issues. The gradient may not flow correctly through this operation.

---

## 4. Why You Don't See Improvement: Root Cause Summary

| Cause | Impact | Severity |
|-------|--------|----------|
| Memory not used at eval time | Memory module is a no-op during testing | **Critical** |
| 25-step lookback on 220-520 step episodes | Cannot capture long-range dependencies | **Critical** |
| Hardcoded timestep=5 | Temporal encoding is constant/useless | **High** |
| Same memory for all deepstack layers | No depth-wise specialization | **Medium** |
| Frozen memory image features | Limited representation quality | **Medium** |

---

## 5. How to Debug and Validate

### Step 1: Verify memory is doing something during training

Add logging inside `ShortTermMemoryBank.forward()`:

```python
def forward(self, memory, visual, timestep):
    ...
    gate = self.write_gate(gate_input)
    # DEBUG: log gate statistics
    print(f"Gate mean: {gate.mean().item():.4f}, std: {gate.std().item():.4f}")
    print(f"Memory T={T}, timestep={timestep}")
    ...
```

If gate is always ~0.5 or always ~1.0, the memory isn't learning useful blending.

### Step 2: Check gradient flow

```python
# In QwenOFT.py forward(), after loss computation:
for name, param in self.qwen_vl_interface.model.memory.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
    else:
        print(f"{name}: NO GRADIENT")
```

### Step 3: Visualize what memory sees

Save the memory images alongside current images during training to verify the data pipeline is correct:

```python
# In datasets.py, after constructing memory_images:
if len(memory_images) > 0:
    for idx, frame in enumerate(memory_images):
        frame[0].save(f"debug/memory_t-{(idx+1)*5}_view0.png")
        frame[1].save(f"debug/memory_t-{(idx+1)*5}_view1.png")
```

### Step 4: Ablation experiments

Run training with `memory: False` and compare loss curves. If the loss curves are identical, the memory module isn't contributing during training either.

---

## 6. Recommendations for Showing Memory's Strength

### Recommendation A: Fix the eval pipeline first (minimum viable fix)

Modify `model2libero_interface.py` to maintain memory state across timesteps:

```python
class ModelClient:
    def __init__(self, ...):
        ...
        self.memory_buffer = []  # store past observations
        self.memory_interval = 5
        self.max_memory_frames = 5

    def step(self, example, step, **kwargs):
        # Collect memory frames
        if step % self.memory_interval == 0 and step > 0:
            self.memory_buffer.append(example["image"])
            if len(self.memory_buffer) > self.max_memory_frames:
                self.memory_buffer = self.memory_buffer[-self.max_memory_frames:]

        # Construct memory for the model
        memory_images = []
        for past_images in self.memory_buffer:
            resized = [Image.fromarray(img).resize((224, 224)) for img in past_images]
            memory_images.append(resized)

        example["memory"] = memory_images
        # ... rest of inference
```

### Recommendation B: Increase memory span for long-horizon tasks

The current 25-step lookback is too short. Options:

| Strategy | Lookback | Memory frames | Pros | Cons |
|----------|----------|---------------|------|------|
| Current | 25 steps | 5 @ interval 5 | Low compute | Too short |
| Wider | 100 steps | 5 @ interval 20 | 4x range | Loses fine detail |
| Deeper | 125 steps | 10 @ interval 12-13 | Good range | 2x compute |
| Hierarchical | 250 steps | 5 near + 5 far | Best range | More complex |

**Recommended: Hierarchical sampling** — keep 2-3 recent frames (interval=5) for fine-grained motion, plus 2-3 far frames (interval=50-100) for sub-goal memory:

```python
# Near memory: recent context
near_steps = [step - 5, step - 10, step - 15]
# Far memory: sub-goal context
far_steps = [step - 50, step - 100, step - 200]
memory_steps = [s for s in near_steps + far_steps if s >= 0]
```

### Recommendation C: Better benchmark for memory

LIBERO-spatial/goal/object may not be the right benchmarks to show memory strength. These tasks are relatively short and often solvable from the current observation alone.

**Better options to demonstrate memory:**

1. **LIBERO-10 (long-horizon)**: Multi-step tasks that require tracking sub-goal completion. Episode length 520 steps. Focus on tasks with >3 sequential sub-goals.

2. **LIBERO-90**: Larger task set, more diversity, tasks that require remembering object states.

3. **Custom "memory-critical" tasks**: Design tasks where the correct action depends on something that happened earlier and is no longer visible:
   - "Pick up the red block, then pick up the blue block" (need to remember red block is done)
   - "Open drawer, take out object, close drawer" (need to remember drawer was opened)
   - Partial observability: object goes behind an occluder

4. **Ablation on LIBERO-10**: Compare:
   - No memory (baseline)
   - Memory with 25-step span (current)
   - Memory with 100-step span
   - Memory with 250-step span

   Plot success rate vs memory span to show the sweet spot.

### Recommendation D: Fix timestep encoding

Pass the actual relative timestep from the data:

```python
# In datasets.py, when building memory_images:
memory_timesteps = []
for i in range(1, max_memory_steps + 1):
    hist_step = step - i * interval
    if hist_step >= 0:
        memory_timesteps.append(i * interval)  # relative time offset

# Return timesteps alongside images
return dict(..., memory=memory_images, memory_timesteps=memory_timesteps)
```

Then propagate this through to the model forward.

### Recommendation E: Consider recurrent memory instead of retrieval

The current design re-encodes past frames from scratch each step (even with caching, the memory bank processes `[T, 2, 64, D]` each forward pass). For deployment:

- **Recurrent approach**: Maintain a compressed memory state `[2, 64, D]` that persists across timesteps. Update it online as new frames arrive. No need to store/replay past images.
- **Advantage**: O(1) memory per step instead of O(T). Works naturally during evaluation.
- **Implementation**: Change `ShortTermMemoryBank.forward()` to accept `prev_memory_state` instead of `memory_tensor`.

---

## 7. Suggested Fix Priority

1. **[P0]** Fix eval pipeline to actually pass memory (Section 6A) — without this, all training effort is wasted
2. **[P0]** Fix hardcoded `timestep=5` — temporal encoding is useless without this
3. **[P1]** Increase memory span with hierarchical sampling (Section 6B)
4. **[P1]** Evaluate on LIBERO-10 with proper memory ablations (Section 6C)
5. **[P2]** Fix in-place tensor modification for correct gradient flow
6. **[P2]** Consider per-layer or shared-but-different memory for deepstack layers
7. **[P3]** Switch to recurrent memory for deployment efficiency (Section 6E)
