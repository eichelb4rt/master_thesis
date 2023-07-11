import torch
import torch_semiring_einsum

equation = torch_semiring_einsum.compile_equation("ik,kj->ij")
m = 4
r = 5
n = 6
matrix_1 = torch.rand(m, r)
matrix_2 = torch.rand(r, n)


def standard_semiring(compute_sum):
    def add_in_place(a, b):
        a += b

    def sum_block(a, dims):
        if dims:
            return torch.sum(a, dim=dims)
        else:
            # This is an edge case that `torch.sum` does not
            # handle correctly.
            return a

    def multiply_in_place(a, b):
        a *= b
    return compute_sum(add_in_place, sum_block, multiply_in_place)


def tropical_semiring(compute_sum):
    def add_in_place(a, b):
        a = torch.maximum(a, b)

    def sum_block(a, dims):
        if dims:
            return torch.amax(a, dim=dims)
        else:
            # This is an edge case that `torch.sum` does not
            # handle correctly.
            return a

    def multiply_in_place(a, b):
        a += b
    return compute_sum(add_in_place, sum_block, multiply_in_place)


def standard_einsum(equation, *args, block_size):
    return torch_semiring_einsum.semiring_einsum_forward(equation, args=args, block_size=block_size, func=standard_semiring)


def tropical_einsum(equation, *args, block_size):
    return torch_semiring_einsum.semiring_einsum_forward(equation, args=args, block_size=block_size, func=tropical_semiring)


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


normal_result = torch.einsum("ik,kj->ij", matrix_1, matrix_2)
semiring_result = standard_einsum(equation, matrix_1, matrix_2, block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE)
assert torch.allclose(semiring_result, normal_result)
tropical_torch_result = tropical_einsum(equation, matrix_1, matrix_2, block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE)
tropical_manual_result = manual_tropical_matmul(matrix_1, matrix_2)
assert torch.allclose(tropical_torch_result, tropical_manual_result)
print("all tests passed.")
