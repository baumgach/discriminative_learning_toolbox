class test:

    def __init__(self, a, b):

        self.a = a
        self.b = b

        self.out = a + b

    def __call__(self, *args, **kwargs):

        return self.out