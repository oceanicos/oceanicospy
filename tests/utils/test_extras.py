from oceanicospy.utils.extras import timing_decorator


def test_preserves_function_name():
    @timing_decorator
    def my_func():
        return 42

    assert my_func.__name__ == "my_func"


def test_returns_correct_result():
    @timing_decorator
    def add(a, b):
        return a + b

    assert add(3, 4) == 7


def test_works_with_string_return():
    @timing_decorator
    def greet(name):
        return f"hello {name}"

    assert greet("world") == "hello world"


def test_works_with_no_args():
    @timing_decorator
    def constant():
        return 99

    assert constant() == 99


def test_works_with_kwargs():
    @timing_decorator
    def power(base, exp=2):
        return base**exp

    assert power(3, exp=3) == 27
