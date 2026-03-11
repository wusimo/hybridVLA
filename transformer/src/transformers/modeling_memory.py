import math
import torch
import torch.nn as nn


class MemorySlotAttention(nn.Module):
    """Multi-head attention, handles [V, N, D] or [N, D] inputs."""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.norm_q = nn.RMSNorm(dim)
        self.norm_kv = nn.RMSNorm(dim)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: [V, Nq, D] or [Nq, D]
            key_value: [V, Nkv, D] or [Nkv, D]
        Returns:
            output: same shape as query
        """
        if query.dim() == 2:
            query = query.unsqueeze(0)
            key_value = key_value.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
            
        V, Nq, D = query.shape
        Nkv = key_value.shape[1]
        
        q = self.q_proj(self.norm_q(query.reshape(V * Nq, D))).reshape(V, Nq, D)
        k = self.k_proj(self.norm_kv(key_value.reshape(V * Nkv, D))).reshape(V, Nkv, D)
        v = self.v_proj(self.norm_kv(key_value.reshape(V * Nkv, D))).reshape(V, Nkv, D)
        
        q = q.view(V, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(V * self.num_heads, Nq, self.head_dim)
        k = k.view(V, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(V * self.num_heads, Nkv, self.head_dim)
        v = v.view(V, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3).reshape(V * self.num_heads, Nkv, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v)
        out = out.view(V, self.num_heads, Nq, self.head_dim).permute(0, 2, 1, 3).reshape(V, Nq, D)
        out = self.o_proj(out.reshape(V * Nq, D)).reshape(V, Nq, D)
        
        if squeeze_back:
            out = out.squeeze(0)
            
        return out


class ShortTermMemoryBank(nn.Module):
    """
    Short-term memory for dual-view features, no batch dimension.
    只返回最新的记忆状态 [2, 64, D]，不维护历史序列。
    
    Input:
        - memory: [T, 2, 64, D], T in [1, 5]，历史记忆（外部管理）
        - visual: [2, 64, D]，当前帧
        - timestep: int，绝对时间步用于编码
        
    Output:
        - new_memory: [2, 64, D]，更新后的最新记忆状态
    """
    def __init__(
        self,
        dim: int,
        num_slots: int = 64,
        num_heads: int = 8,
        max_timesteps: int = 5,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.max_timesteps = max_timesteps
        
        self.write_attn = MemorySlotAttention(dim, num_heads)
        self.read_attn = MemorySlotAttention(dim, num_heads)
        
        self.write_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        
        self.register_buffer(
            "_temporal_div",
            torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        )
        
        self.norm = nn.RMSNorm(dim)
        
    def get_temporal_embedding(self, timestep: int, device: torch.device) -> torch.Tensor:
        """Get sine-cosine temporal embedding."""
        t = torch.tensor(timestep, device=device, dtype=torch.float32)
        angles = t * self._temporal_div.to(device)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return emb  # [D]
    
    def forward(
        self,
        memory: torch.Tensor,   # [T, 2, 64, D], T in [1,5]
        visual: torch.Tensor,   # [2, 64, D]
        timestep: int,
    ) -> torch.Tensor:
        """
        更新记忆，只返回最新的记忆状态 [2, 64, D]。
        外部需要自己用列表/队列维护历史。
        """
        assert memory.dim() == 4, f"memory must be [T,2,64,D], got {memory.shape}"
        assert visual.dim() == 3, f"visual must be [2,64,D], got {visual.shape}"
        
        T, V, S, D = memory.shape
        assert V == 2, f"Need 2 views, got {V}"
        assert S == self.num_slots, f"Need {self.num_slots} slots, got {S}"
        assert visual.shape == (V, S, D), f"visual shape {visual.shape} != {(V,S,D)}"
        assert 0 <= T <= self.max_timesteps
        if T == 0:
            return visual
        device = memory.device
        
        # 时间编码
        t_emb = self.get_temporal_embedding(timestep, device)  # [D]
        visual_t = visual + t_emb.view(1, 1, D)  # [2, 64, D]
        # 聚合历史记忆: [T, 2, 64, D] -> [2, T*64, D]
        memory_per_view = memory.permute(1, 0, 2, 3).reshape(V, T * S, D)  # [2, T*64, D]
        
        # 用最后一帧作为query去attend历史
        last_memory = memory[-1]  # [2, 64, D]
        history_agg = self.write_attn(last_memory, memory_per_view)  # [2, 64, D]
        
        # 门控融合
        gate_input = torch.cat([history_agg, visual_t], dim=-1)  # [2, 64, 2D]
        gate = self.write_gate(gate_input)  # [2, 64, D]
        
        new_memory = gate * visual_t + (1 - gate) * history_agg
        new_memory = self.norm(new_memory)  # [2, 64, D]
        
        return new_memory  # ← 只返回最新状态，不是序列！
    
    def read(
        self,
        queries: torch.Tensor,  # [2, Nq, D] or [Nq, D]
        memory: torch.Tensor,   # [T, 2, 64, D] or [2, 64, D]
    ) -> torch.Tensor:
        """从记忆中读取，支持历史序列或单帧记忆。"""
        # 统一处理：如果是历史序列，取最后一帧
        if memory.dim() == 4:
            mem = memory[-1]  # [2, 64, D]
        else:
            mem = memory  # [2, 64, D]
            assert memory.dim() == 3, f"memory must be 3D or 4D, got {memory.dim()}D"
        
        # 单视角查询
        if queries.dim() == 2:
            mem_avg = mem.mean(dim=0, keepdim=True)  # [1, 64, D]
            context = self.read_attn(queries.unsqueeze(0), mem_avg).squeeze(0)
            return queries + context
        
        # 双视角查询
        context = self.read_attn(queries, mem)
        return queries + context


# ============== 使用示例 ==============

def demo():
    D = 2560  # 你的维度
    num_slots = 64
    device = torch.device('cpu')
    
    memory_bank = ShortTermMemoryBank(dim=D, num_slots=num_slots).to(device)
    
    # 外部管理历史：用一个列表
    history = []  # 存储 [2, 64, D] 的列表
    
    # 第一帧
    visual = torch.randn(2, num_slots, D).to(device)
    # 第一帧没有历史，用自身作为历史
    history.append(visual)  # T=1
    memory_tensor = torch.stack(history, dim=0)  # [1, 2, 64, D]
    
    # 后续帧
    for t in range(1, 10):
        visual = torch.randn(2, num_slots, D).to(device)
        
        # 准备历史（最多5帧）
        if len(history) >= 5:
            history = history[-4:]  # 保留最近4帧
        memory_tensor = torch.stack(history, dim=0)  # [T, 2, 64, D], T in [1,5]
        
        # 更新记忆，返回新的状态
        new_memory = memory_bank(memory_tensor, visual, timestep=t)  # [2, 64, D]
        print(f"Step {t}: input memory {memory_tensor.shape}, output {new_memory.shape}")
        
        # 添加到历史
        history.append(new_memory)
    
    # 读取
    queries = torch.randn(2, 100, D).to(device)
    # 可以用历史序列或单帧
    enhanced = memory_bank.read(queries, torch.stack(history[-5:], dim=0))
    print(f"\nEnhanced: {enhanced.shape}")


if __name__ == "__main__":
    demo()