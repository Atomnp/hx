from pkg.orders import order_total

def test_no_discount():
    assert order_total([10, 20]) == 30

def test_ten_percent():
    assert order_total([10, 20], pct=10) == 27
