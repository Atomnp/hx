import ast, inspect, mathx
fn = mathx.multiply
assert fn(3, 4) == 12
sig = inspect.signature(fn)
assert all(p.annotation is not inspect.Parameter.empty for p in sig.parameters.values()), "missing param annotations"
assert sig.return_annotation is not inspect.Signature.empty, "missing return annotation"
assert fn.__doc__ and fn.__doc__.strip(), "missing docstring"
print("ok")
