from discount import apply

def test_save10():
    assert apply(100, "SAVE10") == 90

def test_unknown():
    assert apply(100, "NOPE") == 100
