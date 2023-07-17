import torch
import torch_semiring_einsum
from enum import Enum
from typing import Callable


class Operations(Enum):
    MIN = "min"
    MAX = "max"
    ADDITION = "plus"
    MULTPLICATION = "mul"


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
            if dims is None:
                return a
            return torch.amin(a, dim=dims)
    elif operation == Operations.MAX:
        def block_function(a, dims):
            if dims is None:
                return a
            return torch.amax(a, dim=dims)
    elif operation == Operations.ADDITION:
        def block_function(a, dims):
            if dims is None:
                return a
            return torch.sum(a, dim=dims)
    else:
        raise ValueError(f"Operation {operation} not supported for block functions.")
    return block_function


def build_semiring(plus_operation: Operations, times_operation: Operations) -> Callable:
    plus_in_place = build_in_place_function(plus_operation)
    plus_block = build_block_function(plus_operation)
    times_in_place = build_in_place_function(times_operation)
    return lambda compute_sum: compute_sum(plus_in_place, plus_block, times_in_place)


def build_semiring_einsum(plus_operation: Operations, times_operation: Operations) -> Callable:
    semiring = build_semiring(plus_operation, times_operation)
    return lambda equation, *args, block_size: torch_semiring_einsum.semiring_einsum_forward(equation, args=args, block_size=block_size, func=semiring)