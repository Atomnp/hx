from pkg.pricing import discounted

def order_total(prices, pct=0):
    return round(sum(discounted(p, pct) for p in prices), 2)
