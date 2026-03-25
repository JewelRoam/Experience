def apply_twice(f, x: int) -> int:
    first = f(x)
    result = f(first)
    return result
