import torch
import torch_semiring_einsum

from semiring_einsum import Operations, build_semiring_einsum


# test our stuff with matrix multiplication
m = 4
r = 5
n = 6
matrix_1 = torch.rand(m, r)
matrix_2 = torch.rand(r, n)
einsum_string = "ik,kj->ij"
equation = torch_semiring_einsum.compile_equation(einsum_string)


standard_einsum = build_semiring_einsum(Operations.ADDITION, Operations.MULTPLICATION)
tropical_einsum = build_semiring_einsum(Operations.MAX, Operations.ADDITION)


def manual_tropical_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert len(a.shape) == 2, "Matrix multiplication is only defined for matrices."
    assert len(b.shape) == 2, "Matrix multiplication is only defined for matrices."
    assert a.shape[1] == b.shape[0], "Number of columns of left matrix has to equal number of rows of right matrix."
    m = a.shape[0]
    r = a.shape[1]
    n = b.shape[1]
    c = torch.empty((m, n))
    for i in range(m):
        for j in range(n):
            c[i, j] = -torch.inf
            for k in range(r):
                c[i, j] = max(c[i, j], a[i, k] + b[k, j])
    return c


expected_result = torch.einsum(einsum_string, matrix_1, matrix_2)
semiring_result = standard_einsum(equation, matrix_1, matrix_2, block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE)
assert torch.allclose(semiring_result, expected_result), "Unexpected result in the standard semiring."

expected_result = tropical_einsum(equation, matrix_1, matrix_2, block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE)
semiring_result = manual_tropical_matmul(matrix_1, matrix_2)
assert torch.allclose(expected_result, semiring_result), "Unexpected result in the tropical semiring."
print("all tests passed.")
