from invoice import total

def test_total():
    assert total("3,4,5") == 12
