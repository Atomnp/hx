from parser import parse_amounts

def total(line):
    return sum(parse_amounts(line))
