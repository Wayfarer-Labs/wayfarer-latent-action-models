from multimethod import multimethod, parametric


def is_even(x):
    return isinstance(x, int) and x % 2 == 0

def is_odd(x):
    return isinstance(x, int) and x % 2 == 1


IsEven = parametric(int, is_even)
IsOdd  = parametric(int, is_odd)

@multimethod
def f(x: IsEven):
    print("even", x)

@multimethod
def f(x: IsOdd):
    print("odd", x)


if __name__ == "__main__":
    f(1)
    f(2)

