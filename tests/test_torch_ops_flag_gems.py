"""
Test torch.ops.flag_gems interface correctness.

This module tests operators registered via TORCH_LIBRARY to the flag_gems namespace,
directly calling C++ registered operators through PyTorch dispatcher.

Note: Overloaded operators use dot notation (e.g., torch.ops.flag_gems.div.Tensor)
"""
import pytest
import torch

import flag_gems

from .accuracy_utils import (
    FLOAT_DTYPES,
    REDUCTION_SHAPES,
    REDUCTION_SMALL_SHAPES,
    SCALARS,
    gems_assert_close,
    gems_assert_equal,
    to_reference,
)
from .conftest import QUICK_MODE

# Device selection
device = flag_gems.device

# Test shapes
BASIC_SHAPES = [(128, 256)] if QUICK_MODE else [(128, 256), (1024, 1024), (32, 64, 128)]
MNK_SHAPES = [(64, 128, 256)] if QUICK_MODE else [(64, 128, 256), (128, 256, 64)]
FLOAT_DTYPES_TEST = [torch.float32] if QUICK_MODE else FLOAT_DTYPES
DIM_LIST = [0, 1] if not QUICK_MODE else [1]
KEEPDIM_LIST = [True, False] if not QUICK_MODE else [False]
TOPK_K_LIST = [1, 5] if not QUICK_MODE else [5]


# ==================== Basic Ops Tests ====================
class TestBasicOps:
    """Test basic arithmetic and tensor operations."""

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", BASIC_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_add_tensor(self, shape, dtype):
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn(shape, dtype=dtype, device=device)

        out = torch.ops.flag_gems.add_tensor(a, b)
        ref = a + b

        gems_assert_close(out, ref, dtype)


# ==================== BLAS Ops Tests ====================
class TestBLASOps:
    """Test BLAS operations: mm, bmm, addmm."""

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("M, N, K", MNK_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_mm(self, M, N, K, dtype):
        mat1 = torch.randn((M, K), dtype=dtype, device=device)
        mat2 = torch.randn((K, N), dtype=dtype, device=device)

        out = torch.ops.flag_gems.mm(mat1, mat2)
        # Compute reference on same device/dtype (no upcast - muDNN doesn't support float64)
        ref = torch.mm(mat1, mat2)

        gems_assert_close(out, ref, dtype, reduce_dim=K)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("M, N, K", MNK_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_bmm(self, M, N, K, dtype):
        batch = 4
        mat1 = torch.randn((batch, M, K), dtype=dtype, device=device)
        mat2 = torch.randn((batch, K, N), dtype=dtype, device=device)

        out = torch.ops.flag_gems.bmm(mat1, mat2)
        # Compute reference on same device/dtype (no upcast - muDNN doesn't support float64)
        ref = torch.bmm(mat1, mat2)

        gems_assert_close(out, ref, dtype, reduce_dim=K)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("M, N, K", MNK_SHAPES)
    @pytest.mark.parametrize("scalar", SCALARS[:2] if QUICK_MODE else SCALARS)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_addmm(self, M, N, K, scalar, dtype):
        mat1 = torch.randn((M, K), dtype=dtype, device=device)
        mat2 = torch.randn((K, N), dtype=dtype, device=device)
        bias = torch.randn((N,), dtype=dtype, device=device)

        alpha = beta = scalar

        out = torch.ops.flag_gems.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)
        # Compute reference on same device/dtype (no upcast - muDNN doesn't support float64)
        ref = torch.addmm(bias, mat1, mat2, beta=beta, alpha=alpha)

        gems_assert_close(out, ref, dtype, reduce_dim=K)


# ==================== Reduction Ops Tests ====================
class TestReductionOps:
    """Test reduction operations: sum, max, argmax."""

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES[:2] if QUICK_MODE else REDUCTION_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_sum(self, shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp, True)

        out = torch.ops.flag_gems.sum(inp)
        ref = torch.sum(ref_inp)

        gems_assert_close(out, ref, dtype)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES[:2] if QUICK_MODE else REDUCTION_SHAPES)
    @pytest.mark.parametrize("dim", DIM_LIST)
    @pytest.mark.parametrize("keepdim", KEEPDIM_LIST)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_sum_dim(self, shape, dim, keepdim, dtype):
        if dim >= len(shape):
            pytest.skip(f"dim {dim} out of range for shape {shape}")

        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp, True)

        # Use dot notation for overloaded operator: sum.dim_IntList
        out = torch.ops.flag_gems.sum.dim_IntList(inp, [dim], keepdim=keepdim)
        ref = torch.sum(ref_inp, dim=[dim], keepdim=keepdim)

        # Use reduce_dim for proper tolerance scaling
        gems_assert_close(out, ref, dtype, reduce_dim=shape[dim])

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES[:2] if QUICK_MODE else REDUCTION_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_max(self, shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp)

        out = torch.ops.flag_gems.max(inp)
        ref = torch.max(ref_inp)

        gems_assert_equal(out, ref)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", REDUCTION_SHAPES[:2] if QUICK_MODE else REDUCTION_SHAPES)
    @pytest.mark.parametrize("dim", DIM_LIST)
    @pytest.mark.parametrize("keepdim", KEEPDIM_LIST)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_max_dim(self, shape, dim, keepdim, dtype):
        if dim >= len(shape):
            pytest.skip(f"dim {dim} out of range for shape {shape}")

        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp)

        # Use dot notation for overloaded operator: max.dim
        values, indices = torch.ops.flag_gems.max.dim(inp, dim, keepdim=keepdim)
        ref_values, ref_indices = torch.max(ref_inp, dim=dim, keepdim=keepdim)

        gems_assert_equal(values, ref_values)
        gems_assert_equal(indices, ref_indices)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", REDUCTION_SMALL_SHAPES)
    @pytest.mark.parametrize("dim", DIM_LIST + [None])
    @pytest.mark.parametrize("keepdim", KEEPDIM_LIST)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_argmax(self, shape, dim, keepdim, dtype):
        if dim is not None and dim >= len(shape):
            pytest.skip(f"dim {dim} out of range for shape {shape}")

        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp)

        out = torch.ops.flag_gems.argmax(inp, dim=dim, keepdim=keepdim)
        ref = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)

        gems_assert_equal(out, ref)


# ==================== Division Ops Tests ====================
class TestDivisionOps:
    """Test division operations: div, floor_divide, true_divide, remainder."""

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", BASIC_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_div_tensor(self, shape, dtype):
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn(shape, dtype=dtype, device=device)
        b = b + 0.1 * torch.sign(b)  # Avoid division by zero
        ref_a = to_reference(a, True)
        ref_b = to_reference(b, True)

        # Use dot notation: div.Tensor
        out = torch.ops.flag_gems.div.Tensor(a, b)
        ref = torch.div(ref_a, ref_b)

        gems_assert_close(out, ref, dtype)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", BASIC_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_true_divide_tensor(self, shape, dtype):
        a = torch.randn(shape, dtype=dtype, device=device)
        b = torch.randn(shape, dtype=dtype, device=device)
        b = b + 0.1 * torch.sign(b)
        ref_a = to_reference(a, True)
        ref_b = to_reference(b, True)

        # Use dot notation: true_divide.Tensor
        out = torch.ops.flag_gems.true_divide.Tensor(a, b)
        ref = torch.true_divide(ref_a, ref_b)

        gems_assert_close(out, ref, dtype)


# ==================== Fill Ops Tests ====================
# Note: fill ops have stability issues with bfloat16 and inplace operations
# Only testing float16/float32 with non-inplace version
FILL_DTYPES = [torch.float16, torch.float32]


class TestFillOps:
    """Test fill operations: fill.Scalar.

    Note: fill_.Scalar (inplace) causes core dump and is skipped.
    bfloat16 causes triton compiler crash and is skipped.
    """

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", BASIC_SHAPES[:2])  # Skip 3D shape for stability
    @pytest.mark.parametrize("value", [0.0, 1.0, -3.14])
    @pytest.mark.parametrize("dtype", FILL_DTYPES)
    def test_fill_scalar(self, shape, value, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)

        # Use dot notation: fill.Scalar
        out = torch.ops.flag_gems.fill.Scalar(inp, value)
        ref = torch.full(shape, value, dtype=dtype, device=device)

        gems_assert_equal(out, ref)


# ==================== Sort Ops Tests ====================
class TestSortOps:
    """Test sorting operations: topk, sort."""

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", [(32, 64)] if QUICK_MODE else [(32, 64), (128, 256)])
    @pytest.mark.parametrize("k", TOPK_K_LIST)
    @pytest.mark.parametrize("largest", [True, False])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_topk(self, shape, k, largest, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp)

        # Only test last dimension (dim=-1) as that's what's currently supported
        dim = -1
        values, indices = torch.ops.flag_gems.topk(inp, k, dim, largest, True)
        ref_values, ref_indices = torch.topk(ref_inp, k, dim=dim, largest=largest, sorted=True)

        gems_assert_close(values, ref_values, dtype)
        # Indices comparison may differ for equal values, so just check shape
        assert indices.shape == ref_indices.shape

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", [(32, 64)] if QUICK_MODE else [(32, 64), (128, 256)])
    @pytest.mark.parametrize("dim", [-1, 0])
    @pytest.mark.parametrize("descending", [True, False])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_sort(self, shape, dim, descending, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        ref_inp = to_reference(inp)

        values, indices = torch.ops.flag_gems.sort(inp, dim, descending)
        ref_values, ref_indices = torch.sort(ref_inp, dim=dim, descending=descending)

        gems_assert_close(values, ref_values, dtype)
        assert indices.shape == ref_indices.shape


# ==================== Normalization Ops Tests ====================
class TestNormalizationOps:
    """Test normalization operations: rms_norm.

    Note: softmax via torch.ops.flag_gems.softmax has a known bug (incorrect normalization).
    Use flag_gems.use_gems() context for softmax instead.
    """

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", [(128, 256)] if QUICK_MODE else [(128, 256), (32, 64, 128)])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_rms_norm(self, shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(shape[-1], dtype=dtype, device=device)
        epsilon = 1e-5

        out = torch.ops.flag_gems.rms_norm(inp, weight, epsilon)

        # Reference implementation (on same device, same dtype to avoid precision issues)
        variance = inp.pow(2).mean(-1, keepdim=True)
        ref = inp * torch.rsqrt(variance + epsilon) * weight

        # Use reduce_dim for tolerance scaling (normalization reduces over last dim)
        gems_assert_close(out, ref, dtype, reduce_dim=shape[-1])


# ==================== Tensor Ops Tests ====================
class TestTensorOps:
    """Test tensor manipulation operations: cat, contiguous, nonzero, embedding."""

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("num_tensors", [2, 3])
    @pytest.mark.parametrize("dim", [0, 1])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_cat(self, num_tensors, dim, dtype):
        shape = (16, 32)
        tensors = [torch.randn(shape, dtype=dtype, device=device) for _ in range(num_tensors)]
        ref_tensors = [to_reference(t) for t in tensors]

        out = torch.ops.flag_gems.cat(tensors, dim)
        ref = torch.cat(ref_tensors, dim=dim)

        gems_assert_close(out, ref, dtype)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", BASIC_SHAPES)
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_contiguous(self, shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        # Make non-contiguous by transposing
        if len(shape) == 2:
            inp = inp.t()
        elif len(shape) >= 3:
            # For 3D+ tensors, swap last two dimensions
            inp = inp.transpose(-1, -2)

        out = torch.ops.flag_gems.contiguous(inp)
        ref = inp.contiguous()

        gems_assert_close(out, ref, dtype)
        assert out.is_contiguous()

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("shape", [(32, 64)] if QUICK_MODE else [(32, 64), (16, 32, 16)])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_nonzero(self, shape, dtype):
        inp = torch.randn(shape, dtype=dtype, device=device)
        inp = inp > 0  # Create boolean tensor
        inp = inp.to(dtype)
        ref_inp = to_reference(inp)

        out = torch.ops.flag_gems.nonzero(inp)
        ref = torch.nonzero(ref_inp)

        gems_assert_equal(out, ref)

    @pytest.mark.torch_ops_flag_gems
    @pytest.mark.parametrize("num_embeddings", [100, 1000])
    @pytest.mark.parametrize("embedding_dim", [64, 128])
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES_TEST)
    def test_embedding(self, num_embeddings, embedding_dim, dtype):
        weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
        indices = torch.randint(0, num_embeddings, (8, 16), device=device)
        ref_weight = to_reference(weight)
        ref_indices = to_reference(indices)

        out = torch.ops.flag_gems.embedding(weight, indices)
        ref = torch.embedding(ref_weight, ref_indices)

        gems_assert_close(out, ref, dtype)
