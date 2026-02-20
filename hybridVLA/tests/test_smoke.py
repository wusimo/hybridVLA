"""
Smoke tests for HybridVLA.

These tests validate that:
1. All modules instantiate without errors
2. Forward passes produce correct output shapes
3. Gradients flow through the quantized layers
4. Memory module accumulates state across timesteps
5. Chain-of-Point produces coordinate predictions
6. Action chunking produces the right trajectory shape
7. The full end-to-end pipeline works

Run: python -m hybrid_vla.tests.test_smoke
  or: pytest hybrid_vla/tests/test_smoke.py -v
"""

import sys
import traceback

import torch
import torch.nn as nn

# Force CPU for tests (no GPU required)
DEVICE = "cpu"
DTYPE = torch.float32


def test_quantization():
    """Test BitNet 1.58-bit quantization utilities."""
    from hybrid_vla.model.quantization import (
        BitLinear158,
        weight_quant_ternary,
        activation_quant_int8,
        DistillationLoss,
        replace_linear_with_bitlinear,
        compute_model_size_bits,
    )

    # Test ternary weight quantization
    w = torch.randn(64, 32)
    w_q = weight_quant_ternary(w)
    scale = w.abs().mean()
    unique_vals = (w_q / scale).round().unique()
    assert all(v in [-1, 0, 1] for v in unique_vals.tolist()), \
        f"Expected ternary values, got {unique_vals}"

    # Test INT8 activation quantization
    x = torch.randn(4, 16, 32)
    x_q = activation_quant_int8(x)
    assert x_q.shape == x.shape

    # Test BitLinear158 forward + backward
    layer = BitLinear158(32, 64)
    x = torch.randn(2, 8, 32, requires_grad=True)
    out = layer(x)
    assert out.shape == (2, 8, 64), f"Expected (2,8,64), got {out.shape}"
    out.sum().backward()
    assert x.grad is not None, "Gradients did not flow through BitLinear158"

    # Test replace utility
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 16),
    )
    replace_linear_with_bitlinear(model)
    assert isinstance(model[0], BitLinear158)
    assert isinstance(model[2], BitLinear158)

    # Test distillation loss
    dl = DistillationLoss(gamma=0.1)
    task_loss = torch.tensor(1.0)
    student = torch.randn(2, 8, 32)
    teacher = torch.randn(2, 8, 32)
    total = dl(task_loss, student, teacher)
    assert total.item() > 1.0, "Distillation loss should add to task loss"

    # Test model size computation
    stats = compute_model_size_bits(model)
    assert stats["ternary_params"] > 0

    print("  [PASS] quantization")


def test_mrope():
    """Test Multimodal Rotary Position Embedding."""
    from hybrid_vla.model.mrope import (
        MultimodalRoPE,
        apply_rotary_emb,
        build_multimodal_position_ids,
    )

    head_dim = 48  # divisible by 6
    mrope = MultimodalRoPE(head_dim, interleave=True)

    # Text position IDs
    text_ids = mrope._build_position_ids_text(10)
    assert text_ids.shape == (10, 3)
    assert (text_ids[:, 0] == text_ids[:, 1]).all(), "Text t/h/w should be identical"

    # Image position IDs
    img_ids = mrope._build_position_ids_image(4, 4)
    assert img_ids.shape == (16, 3)
    assert img_ids[:, 0].unique().numel() == 1, "Image temporal should be constant"

    # Video position IDs
    vid_ids = mrope._build_position_ids_video(3, 4, 4)
    assert vid_ids.shape == (48, 3)  # 3 frames x 16 patches

    # Forward: compute cos/sin
    cos, sin = mrope(img_ids, dtype=DTYPE)
    assert cos.shape == (16, head_dim)
    assert sin.shape == (16, head_dim)

    # Apply to attention tensor
    x = torch.randn(1, 4, 16, head_dim)  # [B, H, N, D]
    out = apply_rotary_emb(x, cos, sin)
    assert out.shape == x.shape

    # Multimodal position ID builder
    pos_ids = build_multimodal_position_ids(
        text_len=5,
        image_grids=[(4, 4)],
    )
    assert pos_ids.shape[1] == 3
    assert pos_ids.shape[0] == 16 + 5  # 4x4 image + 5 text tokens

    print("  [PASS] mrope")


def test_vision_encoder():
    """Test quantized ViT with window attention."""
    from hybrid_vla.model.vision_encoder import (
        PatchEmbed,
        WindowAttention,
        VisionBlock,
        TokenMerger,
        QuantizedVisionEncoder,
    )

    # PatchEmbed
    pe = PatchEmbed(patch_size=14, embed_dim=192)
    img = torch.randn(2, 3, 224, 224)
    patches, gh, gw = pe(img)
    assert patches.shape == (2, 16 * 16, 192), f"Expected (2, 256, 192), got {patches.shape}"
    assert gh == 16 and gw == 16

    # TokenMerger
    merger = TokenMerger(192, 256)
    merged, mh, mw = merger(patches, gh, gw)
    assert merged.shape == (2, 8 * 8, 256), f"Expected (2, 64, 256), got {merged.shape}"
    assert mh == 8 and mw == 8

    # Full QuantizedVisionEncoder (small config, no quantization for speed)
    # head_dim must be divisible by 6 for M-RoPE: 192/4=48, 48%6=0
    enc = QuantizedVisionEncoder(
        img_size=112, patch_size=14, embed_dim=192, llm_dim=256,
        depth=4, num_heads=4, window_size=4, quantize=False,
    )
    img = torch.randn(2, 3, 112, 112)
    vis_tokens, mh, mw = enc(img)
    expected_patches = (112 // 14)  # 8
    expected_merged = expected_patches // 2  # 4
    assert vis_tokens.shape == (2, expected_merged * expected_merged, 256), \
        f"Expected (2, {expected_merged**2}, 256), got {vis_tokens.shape}"

    # Intermediate features for DeepStack
    features = enc.get_intermediate_features(img, layer_indices=[1, 3])
    assert len(features) == 2
    assert features[0].shape[0] == 2

    # Gradient flow
    loss = vis_tokens.sum()
    loss.backward()
    assert enc.blocks[0].mlp.w1.weight.grad is not None

    print("  [PASS] vision_encoder")


def test_language_backbone():
    """Test quantized LLM backbone."""
    from hybrid_vla.model.language_backbone import QuantizedLLMBackbone

    # head_dim must be divisible by 6 for M-RoPE: 192/4=48, 48%6=0
    llm = QuantizedLLMBackbone(
        vocab_size=1000, dim=192, depth=4, num_heads=4,
        num_kv_heads=2, quantize=False,
        deepstack_layers=[1, 3],
    )

    # Forward with token IDs
    ids = torch.randint(0, 1000, (2, 16))
    logits, hidden = llm(input_ids=ids, causal=True)
    assert logits.shape == (2, 16, 1000)
    assert hidden.shape == (2, 16, 192)

    # Forward with embeddings (for multimodal input)
    embeds = torch.randn(2, 20, 192)
    logits, hidden = llm(inputs_embeds=embeds, causal=False)
    assert logits.shape == (2, 20, 1000)

    # Forward with DeepStack injection
    vis_inj = [torch.randn(2, 20, 192) for _ in range(2)]
    vis_mask = torch.ones(2, 20)
    logits, hidden = llm(
        inputs_embeds=embeds,
        visual_injections=vis_inj,
        visual_mask=vis_mask,
    )
    assert logits.shape == (2, 20, 1000)

    # Gradient flow through deepstack gates
    loss = logits.sum()
    loss.backward()
    gate_param = llm.layers[1].vis_gate
    assert gate_param.grad is not None, "Gradient should flow through DeepStack gate"

    print("  [PASS] language_backbone")


def test_spatiotemporal_memory():
    """Test spatiotemporal memory module."""
    from hybrid_vla.model.spatiotemporal_memory import (
        SpatiotemporalMemory,
        ChainOfPointReasoner,
    )

    dim = 192
    mem = SpatiotemporalMemory(dim=dim, num_slots=16, num_heads=4)

    # Initialize memory
    state = mem.init_memory(2, torch.device(DEVICE))
    assert state.shape == (2, 16, dim)

    # Write visual features
    vis_features = torch.randn(2, 32, dim)
    state_t1 = mem.write(state, vis_features, timestep=0)
    assert state_t1.shape == (2, 16, dim)
    # Memory should change after write
    assert not torch.allclose(state, state_t1, atol=1e-5), \
        "Memory should update after write"

    # Write again at next timestep
    vis_features_2 = torch.randn(2, 32, dim)
    state_t2 = mem.write(state_t1, vis_features_2, timestep=1)
    assert not torch.allclose(state_t1, state_t2, atol=1e-5), \
        "Memory should continue updating"

    # Read from memory
    query = torch.randn(2, 10, dim)
    context = mem.read(query, state_t2)
    assert context.shape == (2, 10, dim)
    # Context should differ from raw query (memory adds information)
    assert not torch.allclose(query, context, atol=1e-5), \
        "Memory read should augment the query"

    # Gradient flow
    loss = context.sum()
    loss.backward()
    assert mem.slot_init.grad is not None
    assert mem.write_gate[0].weight.grad is not None

    # Chain-of-Point
    cop = ChainOfPointReasoner(dim=dim, num_points=8)
    hidden = torch.randn(2, 20, dim)
    mask = torch.zeros(2, 20)
    mask[:, [3, 7, 11, 15]] = 1.0  # predict points at these positions

    cop_out = cop(hidden, mask)
    assert cop_out["points"].shape == (2, 20, 3)
    assert cop_out["confidence"].shape == (2, 20, 1)
    assert cop_out["point_embeddings"].shape == (2, 20, dim)

    # Masked positions should have non-zero predictions, others should be zero
    assert (cop_out["points"][:, 3, :].abs().sum() > 0), "Point at mask position should be non-zero"
    assert (cop_out["points"][:, 0, :].abs().sum() == 0), "Point outside mask should be zero"

    print("  [PASS] spatiotemporal_memory")


def test_action_head():
    """Test action chunk head with parallel decoding."""
    from hybrid_vla.model.action_head import ActionChunkHead

    dim = 192
    action_dim = 7
    chunk_size = 5
    head = ActionChunkHead(dim=dim, action_dim=action_dim, chunk_size=chunk_size, num_heads=4)

    # Forward
    hidden = torch.randn(2, 32, dim)
    actions = head(hidden)
    assert actions.shape == (2, chunk_size, action_dim), \
        f"Expected (2, {chunk_size}, {action_dim}), got {actions.shape}"

    # Action normalization
    mean = torch.zeros(action_dim)
    std = torch.ones(action_dim) * 2.0
    head.set_action_stats(mean, std)
    denorm = head.denormalize_actions(actions)
    assert denorm.shape == actions.shape
    # With std=2, denormalized should be 2x the normalized
    assert torch.allclose(denorm, actions * 2.0, atol=1e-5)

    # Loss computation
    targets = torch.randn(2, chunk_size, action_dim)
    loss = head.compute_loss(actions, targets)
    assert loss.ndim == 0, "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"

    # Gradient flow
    loss.backward()
    assert head.action_queries.grad is not None
    assert head.action_proj[1].weight.grad is not None

    print("  [PASS] action_head")


def test_hybrid_vla_full():
    """Test the full HybridVLA model end-to-end."""
    from hybrid_vla.model.config import HybridVLAConfig, PretrainedSources
    from hybrid_vla.model.hybrid_vla import HybridVLA

    # Minimal config for testing (small dims, no quantization, no pretrained loading)
    # head_dim must be divisible by 6 for M-RoPE: 192/4=48, 48%6=0
    config = HybridVLAConfig(
        model_name="test",
        pretrained=PretrainedSources(
            vision_model_id="", llm_model_id="", vlm_model_id=None,
            teacher_vision_model_id="",
        ),
        img_size=112, patch_size=14, vis_embed_dim=192,
        vis_depth=2, vis_num_heads=4, vis_window_size=4,
        vis_quantize=False,
        vocab_size=500, llm_dim=192, llm_depth=4,
        llm_num_heads=4, llm_num_kv_heads=2, llm_quantize=False,
        action_dim=7, action_chunk_size=5, action_num_heads=4,
        memory_num_slots=16, memory_num_heads=4,
        use_memory=True, use_chain_of_point=True,
        cop_num_points=4,
    )

    model = HybridVLA(config)
    model.to(DEVICE)

    # Inputs
    B = 2
    pixel_values = torch.randn(B, 3, 112, 112)
    input_ids = torch.randint(0, 500, (B, 10))
    action_targets = torch.randn(B, 5, 7)

    # --- Test action mode ---
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        action_targets=action_targets,
        mode="action",
        use_deepstack=False,  # skip deepstack for simple test
    )

    assert "actions" in outputs, "Should produce actions in action mode"
    assert outputs["actions"].shape == (B, 5, 7), \
        f"Action shape wrong: {outputs['actions'].shape}"
    assert "action_loss" in outputs, "Should have action loss when targets provided"
    assert "loss" in outputs
    assert outputs["loss"].item() > 0

    # Gradient flow through full model
    outputs["loss"].backward()
    # Check critical gradient paths
    assert model.action_head.action_queries.grad is not None, \
        "Gradients should reach action head"
    assert model.vision_encoder.blocks[0].mlp.w1.weight.grad is not None, \
        "Gradients should reach vision encoder"

    model.zero_grad()

    # --- Test VLM mode ---
    outputs_vlm = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        mode="vlm",
    )
    assert "logits" in outputs_vlm
    assert outputs_vlm["logits"].shape[-1] == 500  # vocab size

    # --- Test planning mode ---
    point_mask = torch.zeros(B, outputs_vlm["hidden_states"].shape[1])
    point_mask[:, 5] = 1.0
    outputs_plan = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        point_mask=point_mask,
        mode="planning",
    )
    assert "points" in outputs_plan, "Planning mode should produce CoP points"
    assert outputs_plan["points"].shape[-1] == 3

    # --- Test memory persistence ---
    mem_state = None
    for t in range(3):
        out = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            memory_state=mem_state,
            timestep=t,
            mode="action",
            use_deepstack=False,
        )
        new_mem = out["memory_state"]
        if mem_state is not None:
            assert not torch.allclose(mem_state, new_mem, atol=1e-5), \
                f"Memory should evolve at timestep {t}"
        mem_state = new_mem.detach()

    # --- Test inference helper ---
    with torch.no_grad():
        actions, mem = model.predict_action(pixel_values, input_ids)
    assert actions.shape == (B, 5, 7)

    # --- Model stats ---
    stats = model.get_model_stats()
    assert stats["total_params"] > 0

    # --- Freeze/unfreeze helpers ---
    model.freeze_all_except_connector()
    trainable = sum(1 for p in model.parameters() if p.requires_grad)
    total = sum(1 for p in model.parameters())
    assert trainable < total, "freeze_all_except_connector should freeze most params"
    assert trainable > 0, "Connector should remain trainable"

    print("  [PASS] hybrid_vla_full (end-to-end)")


def test_data_utilities():
    """Test data preparation utilities."""
    import json
    import tempfile
    import os

    from hybrid_vla.data.prepare_data import (
        compute_action_statistics,
        _extract_heuristic_keypoints,
        RoboticsDataset,
    )

    # Create a temporary manifest
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        for i in range(10):
            sample = {
                "image": "/tmp/nonexistent.png",  # won't actually load in stats
                "instruction": f"test instruction {i}",
                "actions": [[0.1 * i, 0.2, 0.3, 0.0, 0.0, 0.0, float(i % 2)] for _ in range(5)],
                "episode_id": f"ep_{i}",
                "timestep": 0,
            }
            f.write(json.dumps(sample) + "\n")
        tmpfile = f.name

    try:
        # Action statistics
        mean, std = compute_action_statistics(tmpfile)
        assert mean.shape == (7,), f"Expected (7,), got {mean.shape}"
        assert std.shape == (7,), f"Expected (7,), got {std.shape}"
        assert (std > 0).all(), "All stds should be positive"

        # Heuristic keypoints
        actions = [[float(i), float(i), float(i), 0, 0, 0, 0] for i in range(20)]
        kps = _extract_heuristic_keypoints(actions, num_points=4)
        assert len(kps) == 4
        assert len(kps[0]) == 3  # (x, y, z)
        # First keypoint should be near the start
        assert kps[0][0] < kps[-1][0], "Keypoints should span the trajectory"

    finally:
        os.unlink(tmpfile)

    print("  [PASS] data_utilities")


# =============================================================================
# Main runner
# =============================================================================

def main():
    """Run all smoke tests."""
    tests = [
        ("Quantization", test_quantization),
        ("M-RoPE", test_mrope),
        ("Vision Encoder", test_vision_encoder),
        ("Language Backbone", test_language_backbone),
        ("Spatiotemporal Memory", test_spatiotemporal_memory),
        ("Action Head", test_action_head),
        ("Full HybridVLA (E2E)", test_hybrid_vla_full),
        ("Data Utilities", test_data_utilities),
    ]

    print("=" * 60)
    print("HybridVLA Smoke Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f"  [FAIL] {name}: {e}")
            traceback.print_exc()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")

    if errors:
        print("\nFailures:")
        for name, e in errors:
            print(f"  - {name}: {e}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
