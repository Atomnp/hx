from utils import last_n

def test_last_two():
    assert last_n([1, 2, 3, 4], 2) == [3, 4]

def test_last_one():
    assert last_n(["a", "b"], 1) == ["b"]
