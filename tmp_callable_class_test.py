class test:

    def __init__(self, a, b):

        self.a = a
        self.b = b

        self.out = a + b

    def __call__(self, *args, **kwargs):

        return self.out


def f2(a, b):

    return a+b

def f1(a, **kwargs):

    return f2(a, **kwargs)

print(f1(a=1))