import numpy as np
import numpy.typing as npt
from scipy import signal


# pool of symbols used for einsum
SYMBOLS = "abcdefghijklmnopqrstuvwxyz"


def pic_to_vec(pic: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return np.reshape(pic, (pic.shape[0]**2))


def vec_to_pic(vec: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    n = int(np.sqrt(vec.shape[0]))
    return np.reshape(vec, (n, n))


def build_my_design_tensor(n: int, d_G: int, d_F: int) -> npt.NDArray[np.bool_]:
    assert d_F < d_G, "Convolution mask must be smaller than the convoluted tensor."
    d_O = d_G - d_F + 1
    p_shape = tuple(n * [d_O] + n * [d_F] + n * [d_G])
    p = np.full(p_shape, False)
    for idx in np.ndindex(p_shape):
        x = np.array(idx[:n])
        y = np.array(idx[n:2 * n])
        z = np.array(idx[2 * n:3 * n])
        # z + 1 because we use indices that start at 1 in the definition
        p[idx] = np.all(z + 1 == x + d_F - y)
    return p


def build_my_convolution_tensor(d_G: int, f_2d: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    d_F = f_2d.shape[0]
    assert len(f_2d.shape) == 2, "This tensor is only defined for 2d masks."
    assert f_2d.shape[0] == f_2d.shape[1], "This convolution is only defined for square masks."
    assert d_F < d_G, "Convolution mask must be smaller than the convoluted tensor."
    n = 2
    d_O = d_G - d_F + 1
    p_2d = build_my_design_tensor(n, d_G, d_F)
    s_x = SYMBOLS[:n]
    s_y = SYMBOLS[n:2 * n]
    s_z = SYMBOLS[2 * n:3 * n]
    convolution_tensor_matrix_form = np.einsum(f"{s_x}{s_y}{s_z},{s_y} -> {s_x}{s_z}", p_2d, f_2d, dtype=np.int64)
    return np.reshape(convolution_tensor_matrix_form, (d_O**2, d_G**2))


def my_convolve(g: npt.NDArray[np.int64], f: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
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
    p = build_my_design_tensor(n, d_G, d_F)
    s_x = SYMBOLS[:n]
    s_y = SYMBOLS[n:2 * n]
    s_z = SYMBOLS[2 * n:3 * n]
    return np.einsum(f"{s_x}{s_y}{s_z},{s_y},{s_z} -> {s_x}", p, f, g, dtype=np.int64)


def build_juliens_convolution_tensor(d_G: int, f_2d: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    """Builds the design tensor only for convolution on 2d pictures. Views the 2d picture as a vector."""

    d_F = f_2d.shape[0]
    assert len(f_2d.shape) == 2, "This tensor is only defined for 2d masks."
    assert f_2d.shape[0] == f_2d.shape[1], "This convolution is only defined for square masks."
    assert d_F < d_G, "Convolution mask must be smaller than the convoluted tensor."
    d_O = d_G - d_F + 1
    p = np.zeros((d_O**2, d_G**2), dtype=np.int64)
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
                    column_index = d_G * y_G + x_G
                    # we have to mirror the mask
                    p[row_index, column_index] = f_2d[d_F - y_F - 1, d_F - x_F - 1]
    return p


def julien_convolve(g_2d: npt.NDArray[np.int64], f_2d: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    d_G = g_2d.shape[0]
    d_F = f_2d.shape[0]
    assert len(g_2d.shape) == 2, "This convolution is only defined for 2d masks."
    assert len(f_2d.shape) == 2, "This convolution is only defined for 2d pictures."
    assert g_2d.shape[0] == g_2d.shape[1], "This convolution is only defined for square masks."
    assert f_2d.shape[0] == f_2d.shape[1], "This convolution is only defined for square pictures."
    assert d_F < d_G, "Convolution mask must be smaller than the convoluted tensor."
    p = build_juliens_convolution_tensor(d_G, f_2d)
    return vec_to_pic(p @ pic_to_vec(g_2d))


def main():
    # does my convolution work?
    f = np.array([1, 2])
    g = np.array([4, 5, 6])
    np_result = np.convolve(g, f, mode="valid")
    my_result = my_convolve(g, f)
    assert np.all(my_result == np_result)

    # does julien's convolution work?
    f_2d = np.array([
        [1, 2],
        [3, 4]
    ])
    g_2d = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])
    np_result_2d = signal.convolve2d(g_2d, f_2d, mode="valid")
    julien_result = julien_convolve(g_2d, f_2d)
    assert np.all(np_result_2d == julien_result)

    # is my convolution tensor the same as julien's convolution tensor?
    d_G = g_2d.shape[0]
    my_convolution_tensor = build_my_convolution_tensor(d_G, f_2d)
    juliens_convolution_tensor = build_juliens_convolution_tensor(d_G, f_2d)
    print("my convolution tensor:")
    print(my_convolution_tensor)
    print("juliens convolution tensor:")
    print(juliens_convolution_tensor)


if __name__ == "__main__":
    main()
