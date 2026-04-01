def make_adder(x: int):
    def adder(y: int) -> int:
        return x + y
    return adder
