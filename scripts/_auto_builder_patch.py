import importlib, inspect, sys
m = importlib.import_module("exact_rect_gates")
cands = []
for name, fn in inspect.getmembers(m, inspect.isfunction):
    nl = name.lower()
    if any(k in nl for k in ("phi","rect","gate")):
        try:
            ps = inspect.signature(fn).parameters
        except Exception:
            continue
        # needs x,y,z and likely a,vp1,vx,vp2 (allow extras)
        if all(k in ps for k in ("x","y","z")) or len(ps)>=3:
            cands.append(name)
print("CANDIDATES:", cands)
# Pick the first by default
if cands:
    print("PICK:", cands[0])
else:
    print("NO_CANDIDATE")
