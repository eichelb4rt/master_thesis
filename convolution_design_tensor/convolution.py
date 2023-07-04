import numpy as np
import numpy.typing as npt
from scipy import signal


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
    """Builds the design tensor only for convolution on 2d pictures. Views the 2d picture as a vector."""

    assert d_F < d_G, "Convolution mask must be smaller than the convoluted tensor."
    d_O = d_G - d_F + 1
    p = np.full((d_O**2, d_G**2), False)
    # for all output pixels
    for y_O in range(d_O):
        for x_O in range(d_O):
            # figure out which row in the design tensor the pixel represents
            row_index = d_O * y_O + x_O
            # and figure out how this row is filled
            for y_F in range(d_F):
                for x_F in range(d_F):
                    x_G = x_O + x_F
                    y_G = y_O + y_F
                    column_index = d_G * x_G + y_G
                    p[row_index, column_index] = True
    return p


def pic_to_vec(pic: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return np.reshape(pic, (pic.shape[0]**2))


def vec_to_pic(vec: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    n = int(np.sqrt(vec.shape[0]))
    return np.reshape(vec, (n, n))


def julien_convolve(g_2d: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    p = build_julien_design_tensor(2, 3)
    # need to transpose because of the way we reshape things around here (i think)
    return vec_to_pic(p @ pic_to_vec(g_2d)).T


def main():
    f = np.array([1, 2])
    g = np.array([4, 5, 6])
    np_result = np.convolve(g, f, mode="valid")
    my_result = my_convolve(f, g)
    assert np.all(my_result == np_result)

    f_2d = np.array([
        [1, 1],
        [1, 1]
    ])
    g_2d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    np_result_2d = signal.convolve2d(g_2d, f_2d, mode="valid")
    julien_result = julien_convolve(g_2d)
    print(julien_result)
    print(np_result_2d)
    assert np.all(np_result_2d == julien_result)


if __name__ == "__main__":
    main()
