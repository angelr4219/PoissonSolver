import sys, types, inspect
sys.path.append('/app')
import gate_benchmark as M

print("== Top-level functions ==")
for n,o in vars(M).items():
    if isinstance(o, types.FunctionType):
        try: sig = str(inspect.signature(o))
        except Exception: sig="(...)"
        print(f"  {n}{sig}")

print("\n== Top-level classes (with run/solve/main/call) ==")
for n,cls in [(n,o) for n,o in vars(M).items() if isinstance(o,type)]:
    meths = [m for m in ("run","solve","main","__call__") if hasattr(cls,m)]
    if meths:
        print(f"  {n}  methods={meths}")
