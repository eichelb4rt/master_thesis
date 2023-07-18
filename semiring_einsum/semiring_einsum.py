import torch
import torch_semiring_einsum
from enum import Enum
from typing import Callable


class Operations(Enum):
    MIN = "min"
    MAX = "max"
    ADDITION = "plus"
    MULTPLICATION = "mul"


################################################################
# build functions to evaluate einsum on a semiring
################################################################


def build_in_place_function(operation: Operations) -> Callable:
    if operation == Operations.MIN:
        def in_place_function(a, b):
            a = min(a, b)
    elif operation == Operations.MAX:
        def in_place_function(a, b):
            a = max(a, b)
    elif operation == Operations.ADDITION:
        def in_place_function(a, b):
            a += b
    elif operation == Operations.MULTPLICATION:
        def in_place_function(a, b):
            a *= b
    else:
        raise ValueError(f"Operation {operation} not supported for in place functions.")
    return in_place_function


def build_block_function(operation: Operations) -> Callable:
    if operation == Operations.MIN:
        def block_function(a, dims):
            if not dims:
                return a
            return torch.amin(a, dim=dims)
    elif operation == Operations.MAX:
        def block_function(a, dims):
            if not dims:
                return a
            return torch.amax(a, dim=dims)
    elif operation == Operations.ADDITION:
        def block_function(a, dims):
            if not dims:
                return a
            return torch.sum(a, dim=dims)
    else:
        raise ValueError(f"Operation {operation} not supported for block functions.")
    return block_function


def build_semiring_einsum(plus_operation: Operations, times_operation: Operations) -> Callable:
    plus_in_place = build_in_place_function(plus_operation)
    plus_block = build_block_function(plus_operation)
    times_in_place = build_in_place_function(times_operation)
    semiring_func = lambda compute_sum: compute_sum(plus_in_place, plus_block, times_in_place)
    return lambda equation, *args, block_size=torch_semiring_einsum.AUTOMATIC_BLOCK_SIZE: torch_semiring_einsum.semiring_einsum_forward(equation, args=args, block_size=block_size, func=semiring_func)


################################################################
# Testing
################################################################


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


def manual_tropical_norm(matrix: torch.Tensor, vector: torch.Tensor) -> float:
    assert len(matrix.shape) == 2, "First argument has to be a matrix."
    assert len(vector.shape) == 1, "Second argument has to be a vector."
    assert matrix.shape[1] == vector.shape[0], "Number of columns of matrix has to equal number of entries in the vector."
    m, n = matrix.shape
    resulting_vector = torch.empty(n)
    for i in range(m):
        resulting_vector[i] = -torch.inf
        for j in range(n):
            resulting_vector[i] = max(resulting_vector[i], matrix[i, j] + vector[j])
    # compute the "tropical norm" ("tropical squared")
    return 2 * torch.max(resulting_vector)


def main():
    my_standard_einsum = build_semiring_einsum(Operations.ADDITION, Operations.MULTPLICATION)
    my_tropical_einsum = build_semiring_einsum(Operations.MAX, Operations.ADDITION)

    # test matrix multiplication
    m = 4
    r = 5
    n = 6
    matrix_1 = torch.rand(m, r)
    matrix_2 = torch.rand(r, n)
    einsum_string = "ik,kj->ij"
    equation = torch_semiring_einsum.compile_equation(einsum_string)
    expected_result = torch.einsum(einsum_string, matrix_1, matrix_2)
    my_result = my_standard_einsum(equation, matrix_1, matrix_2)
    assert torch.allclose(my_result, expected_result), "Unexpected result during matmul: standard semiring."
    print("matmul: standard semiring passed.")
    expected_result = manual_tropical_matmul(matrix_1, matrix_2)
    my_result = my_tropical_einsum(equation, matrix_1, matrix_2)
    assert torch.allclose(expected_result, my_result), "Unexpected result during matmul: tropical semiring."
    print("matmul: tropical semiring test passed.")

    # test element-wise product
    m = 4
    n = 5
    matrix_1 = torch.rand(m, n)
    matrix_2 = torch.rand(m, n)
    einsum_string = "ij,ij->ij"
    equation = torch_semiring_einsum.compile_equation(einsum_string)
    expected_result = torch.einsum(einsum_string, matrix_1, matrix_2)
    my_result = my_standard_einsum(equation, matrix_1, matrix_2)
    assert torch.allclose(my_result, expected_result), "Unexpected result during element-wise: standard semiring."
    print("element-wise: standard semiring test passed.")
    expected_result = matrix_1 + matrix_2
    my_result = my_tropical_einsum(equation, matrix_1, matrix_2)
    assert torch.allclose(my_result, expected_result), "Unexpected result during element-wise: tropical semiring."
    print("element-wise: tropical semiring test passed.")

    # test norm of matrix-vector product
    m = 5
    n = 6
    matrix_1 = torch.rand(m, n)
    vector_1 = torch.rand(n)
    einsum_string = "ij,j,ij,j->"
    equation = torch_semiring_einsum.compile_equation(einsum_string)
    expected_result = torch.einsum(einsum_string, matrix_1, vector_1, matrix_1, vector_1)
    my_result = my_standard_einsum(equation, matrix_1, vector_1, matrix_1, vector_1)
    assert torch.isclose(my_result, expected_result), "Unexpected result during norm: standard semiring."
    print("norm: standard semiring test passed.")
    expected_result = manual_tropical_norm(matrix_1, vector_1)
    my_result = my_tropical_einsum(equation, matrix_1, vector_1, matrix_1, vector_1)
    assert torch.isclose(my_result, expected_result), "Unexpected result during norm: tropical semiring."
    print("norm: tropical semiring test passed.")

    print("all tests passed.")


if __name__ == "__main__":
    main()
