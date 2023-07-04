import numpy as np
import numpy.typing as npt
import itertools


SYMBOLS = "abcdefghijklmnopqrstuvwxyz"


def build_my_design_tensor(n: int, d_F: int, d_G: int) -> npt.NDArray[np.bool_]:
    assert d_F < d_G, "Convolution mask must be smaller than the convoluted tensor."
    d_O = d_G - d_F + 1
    p_shape = tuple(n * [d_O] + n * [d_F] + n * [d_G])
    p = np.full(p_shape, False)
    for idx in np.ndindex(p_shape):
        x = np.array(idx[:n])
        y = np.array(idx[n:2 * n])
        z = np.array(idx[2 * n:3 * n])
        # z + 1 because we use indices that start at 1 in the definition
        p[idx] = z + 1 == x + d_F - y
    return p


def my_convolve(f: npt.NDArray[np.int64], g: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    # assert F and G work with our definition
    assert len(f.shape) == len(g.shape)
    n = len(f.shape)
    d_F = f.shape[0]
    d_G = g.shape[0]
    assert d_F < d_G
    for axis in range(n):
        assert f.shape[axis] == d_F
        assert g.shape[axis] == d_G
    # build the design tensor and compute the tensor contraction
    p = build_my_design_tensor(n, d_F, d_G)
    s_x = SYMBOLS[:n]
    s_y = SYMBOLS[n:2 * n]
    s_z = SYMBOLS[2 * n:3 * n]
    return np.einsum(f"{s_x}{s_y}{s_z},{s_y},{s_z} -> {s_x}", p, f, g, dtype=np.int64)


def build_julien_design_tensor(d_F: int, d_G: int) -> npt.NDArray[np.bool_]:
    pass


def main():
    f = np.array([1, 2])
    g = np.array([4, 5, 6])
    np_result = np.convolve(g, f, mode="valid")
    my_result = my_convolve(f, g)
    assert np.all(my_result == np_result)


if __name__ == "__main__":
    main()
