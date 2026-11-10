import ctypes
import math
import pytest
import torch

import hxgemm_ext  # noqa: F401  registers torch.ops.hxgemm

torch.backends.cuda.matmul.allow_tf32 = False

KERNELS = [1, 2, 3, 4, 5]
SIZES = [1, 7, 33, 128, 1000, 2047, 4096]


def ref(a, b, c, alpha, beta):
    out = alpha * (a.double() @ b.double())
    if beta != 0:
        out += beta * c.double()
    return out


def close(got, want, k):
    rtol, atol = 1e-5 * math.sqrt(k), 1e-6 * k
    return torch.all((got.double() - want).abs() <= atol + rtol * want.abs())


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("n", SIZES)
def test_square(kernel, n):
    if kernel <= 2 and n > 2047:
        pytest.skip("naive kernels are too slow at 4096")
    g = torch.Generator(device="cuda").manual_seed(0)
    a = torch.rand(n, n, device="cuda", generator=g) * 2 - 1
    b = torch.rand(n, n, device="cuda", generator=g) * 2 - 1
    got = torch.ops.hxgemm.sgemm(a, b, kernel, [])
    assert close(got, ref(a, b, None, 1.0, 0.0), n)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.5, 2.0), (1.0, 1.0)])
def test_alpha_beta_and_offset(kernel, alpha, beta):
    m, n, k = 129, 67, 45
    big_a = torch.rand(m, k + 1, device="cuda") * 2 - 1
    big_b = torch.rand(k, n + 1, device="cuda") * 2 - 1
    a, b = big_a[:, 1:], big_b[:, 1:]   # pointer offset 1 float, ld = cols + 1
    c = torch.rand(m, n, device="cuda")
    want = ref(a, b, c, alpha, beta)
    torch.ops.hxgemm.sgemm_out(c, a, b, kernel, [], alpha, beta)
    assert close(c, want, k)


@pytest.mark.parametrize("kernel", KERNELS)
def test_beta_zero_ignores_nan(kernel):
    a = torch.rand(40, 50, device="cuda")
    b = torch.rand(50, 60, device="cuda")
    c = torch.full((40, 60), float("nan"), device="cuda")
    torch.ops.hxgemm.sgemm_out(c, a, b, kernel, [], 1.0, 0.0)
    assert not torch.isnan(c).any()


class _Tile(ctypes.Structure):
    _fields_ = [(n, ctypes.c_int) for n in ("bm", "bn", "bk", "tm", "tn", "stages")]


def _default_tile(kernel, m, n, k):
    lib = ctypes.CDLL(hxgemm_ext.__file__)
    f = lib.hxgemm_default_tile
    f.restype, f.argtypes = _Tile, [ctypes.c_int] * 4
    t = f(kernel, m, n, k)
    return (t.bm, t.bn, t.bk, t.tm, t.tn, t.stages)


BIG, SMALL = (128, 256, 16, 16, 8, 3), (128, 128, 32, 8, 4, 2)


@pytest.mark.parametrize("m,n,want", [
    (4096, 4096, BIG),    # 32 x 16 = 512 blocks of 128x256
    (2048, 4096, BIG),    # 16 x 16 = 256: boundary, big
    (1920, 4096, SMALL),  # 15 x 16 = 240: just below
    (4096, 1920, BIG),    # 32 x ceil(1920/256) = 32 x 8 = 256: boundary from the N side
    (4096, 1792, SMALL),  # 32 x 7 = 224
    (2048, 2048, SMALL),  # 16 x 8 = 128
    (16, 4096, SMALL),    # 1 x 16 = 16 (decode)
])
def test_default_tile_rule(m, n, want):
    assert _default_tile(5, m, n, 4096) == want
    assert _default_tile(4, m, n, 4096) == (64, 128, 16, 8, 4, 0)
    assert _default_tile(3, m, n, 4096) == (0, 0, 0, 0, 0, 0)


@pytest.mark.parametrize("m,n", [(2048, 4096), (1920, 4096)])
def test_default_tile_matches_explicit(m, n):
    # The all-zero tile must give bitwise the same result as passing the rule's tile explicitly.
    g = torch.Generator(device="cuda").manual_seed(1)
    a = torch.rand(m, 512, device="cuda", generator=g) * 2 - 1
    b = torch.rand(512, n, device="cuda", generator=g) * 2 - 1
    got_default = torch.ops.hxgemm.sgemm(a, b, 5, [])
    got_explicit = torch.ops.hxgemm.sgemm(a, b, 5, list(_default_tile(5, m, n, 512)))
    assert torch.equal(got_default, got_explicit)


def test_rejects_bad_input():
    a = torch.rand(4, 5, device="cuda")
    with pytest.raises(ValueError):
        torch.ops.hxgemm.sgemm(a, torch.rand(6, 4, device="cuda"), 1, [])
    with pytest.raises(ValueError):
        torch.ops.hxgemm.sgemm(a.double(), torch.rand(5, 4, device="cuda").double(), 1, [])
    with pytest.raises(ValueError):
        torch.ops.hxgemm.sgemm(a.t(), torch.rand(4, 4, device="cuda"), 1, [])  # column-major
