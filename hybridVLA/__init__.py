"""
HybridVLA - A Vision-Language-Action model combining:
- BitVLA: 1.58-bit ternary quantization, distillation-aware QAT, action chunking
- RynnBrain: Hierarchical planning, Chain-of-Point reasoning, spatiotemporal memory
- Qwen VLM: M-RoPE position encoding, dynamic resolution, window attention, SwiGLU

v2 additions (LingBot-VLA / π0 / GR00T N1 inspired):
- MoT Action Expert: dedicated transformer branch for action prediction
- Flow Matching: continuous action generation via learned velocity fields
- Multi-view: 1-4 camera views through shared ViT
- Depth perception: distillation from Depth Anything V2
- Knowledge insulation: action gradients don't corrupt VLM weights
- Proprioception: robot joint state as input to action expert
"""

__version__ = "0.2.0"
