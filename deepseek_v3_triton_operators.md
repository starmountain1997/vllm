以下是 DeepSeek-V3 (DeepSeek 3.2) 在 vLLM 中的 Triton 算子整理。主要涉及 **MLA (Multi-Head Latent Attention)** 和 **MoE (Mixture of Experts)** 模块。其他模块如 RMSNorm 和 Rotary Embedding 主要使用 CUDA C++ 算子或 PyTorch 原生实现。

### DeepSeek-V3 Triton 算子调用树

*   **Model**: `DeepseekV2ForCausalLM` / `DeepseekV3ForCausalLM` (in `vllm/model_executor/models/deepseek_v2.py`)
    *   **Module**: `self.layers` (`DeepseekV2DecoderLayer`)
        *   **Component**: **Attention (MLA)**
            *   **Layer Class**: `DeepseekV2MLAAttention`
            *   **Wrapper**: `MultiHeadLatentAttentionWrapper` (in `vllm/model_executor/layers/mla.py`)
            *   **Core Layer**: `MLAAttention` (in `vllm/attention/layer.py`)
            *   **Backend**: `TritonMLAImpl` (in `vllm/v1/attention/backends/mla/triton_mla.py`)
            *   **Triton Kernels (Decode Phase)**:
                *   **Definition File**: `vllm/attention/ops/triton_decode_attention.py`
                *   `_fwd_kernel_stage1`
                *   `_fwd_grouped_kernel_stage1` (用于 GQA/MQA/MLA 分组情况)
                *   `_fwd_kernel_stage2`
        *   **Component**: **MoE (Mixture of Experts)**
            *   **Layer Class**: `DeepseekV2MoE` (in `vllm/model_executor/models/deepseek_v2.py`)
            *   **Core Layer**: `SharedFusedMoE` (in `vllm/model_executor/layers/fused_moe/shared_fused_moe.py`)
                *   Inherits from `FusedMoE` (in `vllm/model_executor/layers/fused_moe/layer.py`)
            *   **Method**: `FusedMoE.forward` -> `self.quant_method.apply` -> `fused_experts`
            *   **Triton Kernels**:
                *   **Definition File**: `vllm/model_executor/layers/fused_moe/fused_moe.py`
                *   `fused_moe_kernel`: 核心 MoE 计算算子。
                *   `fused_moe_kernel_gptq_awq`: 量化场景下的 MoE 计算算子。
                *   `compute_identity_kernel`: 用于处理 Zero Experts 的特殊情况。

---

### 详细调用栈与定义定位

#### 1. Attention (MLA - Decode Phase)

DeepSeek-V3 使用 MLA 架构，其 Decode 阶段的 Attention 计算由 Triton 算子实现。

*   **调用入口**:
    `vllm/model_executor/models/deepseek_v2.py` -> `DeepseekV2MLAAttention.forward()`
    `->` `vllm/model_executor/layers/mla.py` -> `MultiHeadLatentAttentionWrapper.forward_native()`
    `->` `vllm/attention/layer.py` -> `MLAAttention.forward()`
    `->` `vllm/v1/attention/backends/mla/triton_mla.py` -> `TritonMLAImpl._forward_decode()`
    `->` `vllm/attention/ops/triton_decode_attention.py` -> `decode_attention_fwd()`

*   **Triton 算子定义 (`vllm/attention/ops/triton_decode_attention.py`)**:
    *   **`_fwd_kernel_stage1`**: 第一阶段 Attention 计算 kernel。
    *   **`_fwd_grouped_kernel_stage1`**: 针对 MLA/GQA 优化的分组 Attention 计算 kernel。
    *   **`_fwd_kernel_stage2`**: 第二阶段 Reduction kernel (ReduceV)。

#### 2. Mixture of Experts (MoE)

DeepSeek-V3 使用 Shared Expert + Routed Expert 的架构，底层由 `FusedMoE` 逻辑支撑。

*   **调用入口**:
    `vllm/model_executor/models/deepseek_v2.py` -> `DeepseekV2MoE.forward()`
    `->` `vllm/model_executor/layers/fused_moe/shared_fused_moe.py` -> `SharedFusedMoE.forward()` (处理 Shared Expert 部分)
    `->` `vllm/model_executor/layers/fused_moe/layer.py` -> `FusedMoE.forward()` (处理 Routed Expert 部分)
    `->` `vllm/model_executor/layers/fused_moe/layer.py` -> `FusedMoE.forward_native()` -> `self.forward_impl()`
    `->` `vllm/model_executor/layers/fused_moe/layer.py` -> `FusedMoE.quant_method.apply()` (通常是 `UnquantizedFusedMoEMethod`)
    `->` `vllm/model_executor/layers/fused_moe/fused_moe.py` -> `fused_experts()` -> `invoke_fused_moe_kernel()`

*   **Triton 算子定义 (`vllm/model_executor/layers/fused_moe/fused_moe.py`)**:
    *   **`fused_moe_kernel`**: 标准 Fused MoE 计算的核心 Triton kernel。
    *   **`fused_moe_kernel_gptq_awq`**: 如果模型经过 GPTQ/AWQ 量化，则调用此 kernel。
    *   **`compute_identity_kernel`**: `zero_experts_compute_triton` 函数中使用，用于某些特殊 Expert配置。

#### 3. 其他模块 (Non-Triton)

*   **RMSNorm (`vllm/model_executor/layers/layernorm.py`)**:
    *   使用的是 `torch.ops.vllm.rms_norm`，这是一个 **CUDA C++** 扩展算子，而非 Triton 算子。
*   **Rotary Embedding (`vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py`)**:
    *   `DeepseekScalingRotaryEmbedding` 类调用 `forward_native`，使用的是 **PyTorch 原生操作** (`torch.cat`, `sin`, `cos` 等) 进行计算，未使用 Triton 算子。