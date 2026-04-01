def repeat(text: str, n: int = 3) -> str:
    result = ''
    for i in range(n):
        result = result + text
    return result
