def apply(price, code):
    """SAVE10 gives 10% off; unknown codes give nothing off."""
    if code == "SAVE10":
        return price * 0.1
    return price
