import re, sys, pathlib

p = pathlib.Path("rect_gate_benchmark.py")
src = p.read_text()

# -------- 1) Replace the whole solve_laplace(...) with a clean version ----------
new_solve = r'''
# ---------- Solve (IMPORTANT: use the SAME V the BCs were built on) ----------
def solve_laplace(V, bcs, eps_r=11.7, eps0=8.8541878128e-12):
    domain = V.mesh
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    lam = PETSc.ScalarType(eps_r * eps0)
    a = lam * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, PETSc.ScalarType(0.0)) * v * ufl.dx

    petsc_opts = {
        "ksp_type": "cg",
        "pc_type": "hypre",
        "ksp_rtol": 1e-10,
        "ksp_error_if_not_converged": True
    }
    problem = LinearProblem(a, L, bcs=bcs, petsc_options=petsc_opts)
    uh = problem.solve()
    uh.name = "phi"

    # quick sanity report
    uarr = uh.x.array
    n_bad = int(np.sum(~np.isfinite(uarr)))
    gmin = float(np.nanmin(uarr)) if uarr.size else np.nan
    gmax = float(np.nanmax(uarr)) if uarr.size else np.nan
    if domain.comm.rank == 0:
        print(f"[VOLTAGE] nonfinite={n_bad}, min={gmin:.6e}, max={gmax:.6e}")
    return uh
'''.lstrip("\n")

solve_pattern = re.compile(
    r'(?s)^def\s+solve_laplace\([\s\S]*?\)\s*:\s*[\s\S]*?^\s*return\s+uh\s*$',
    re.MULTILINE
)
if not solve_pattern.search(src):
    print("[-] Could not locate solve_laplace(...) to replace.", file=sys.stderr)
else:
    src = solve_pattern.sub(new_solve.rstrip(), src)
    print("[+] Replaced solve_laplace(...)")

# -------- 2) Insert error block right after 'uh = solve_laplace(V, bcs)' ----------
insert_after = re.search(r'^(?P<indent>\s*)uh\s*=\s*solve_laplace\(V,\s*bcs\)\s*$', src, re.MULTILINE)
if not insert_after:
    print("[-] Could not find the line 'uh = solve_laplace(V, bcs)' to insert after.", file=sys.stderr)
else:
    indent = insert_after.group('indent')
    err_block = f'''
{indent}# --- Global error metrics vs analytic rectangular-gate potential ---
{indent}from errors import report_errors
{indent}from exact_rect_gates import phi0_rect_three_gates_factory
{indent}os.makedirs("results", exist_ok=True)
{indent}
{indent}# Use the same geometry/voltages you just ran
{indent}exact_fun = phi0_rect_three_gates_factory(a, zbar, xs_gates, Vs_gates)
{indent}
{indent}metrics = report_errors(domain, uh, exact_fun, out_prefix=f"{{outprefix}}_err", qdeg=4)
{indent}if domain.comm.rank == 0:
{indent}    print(f"[GLOBAL] ||e||_L2      = {{metrics['L2']:.6e}}")
{indent}    print(f"[GLOBAL] |e|_H1        = {{metrics['H1_seminorm']:.6e}}")
{indent}    print(f"[GLOBAL] ||e||_Linf(d) = {{metrics['Linf_dof']:.6e}}")
{indent}    print("[HOTSPOT] local cell id =", metrics["max_cell"]["local_id"])
{indent}    print("[HOTSPOT] centroid xyz  =", metrics["max_cell"]["centroid"])
{indent}    print("[HOTSPOT] L2(cell)      =", metrics["max_cell"]["L2_cell_norm"])
{indent}    # also log to CSV for Leah
{indent}    import csv
{indent}    cx, cy, cz = (metrics["max_cell"]["centroid"] or [None, None, None])
{indent}    with open("results/error_summary.csv", "a", newline="") as f:
{indent}        csv.writer(f).writerow([
{indent}            outprefix, metrics["L2"], metrics["H1_seminorm"], metrics["Linf_dof"],
{indent}            cx, cy, cz, metrics["max_cell"]["L2_cell_norm"],
{indent}        ])
'''.lstrip("\n")
    pos = insert_after.end()
    src = src[:pos] + "\n" + err_block + src[pos:]
    print("[+] Inserted error-analysis block after solve_laplace call in run_once(...)")

# -------- 3) Remove the old auto-detect footer (the thing printing 'Skipped ...') ----------
footer_pattern = re.compile(r'(?s)#\s*={10,}\s*\n#\s*Error analysis \(append\)[\s\S]*\Z')
if footer_pattern.search(src):
    src = footer_pattern.sub("", src)
    print("[+] Removed legacy error-analysis footer")
else:
    # also try other footer variants you might still have
    footer2 = re.compile(r'(?s)#\s*============================================[\s\S]*\Z')
    if footer2.search(src):
        src = footer2.sub("", src)
        print("[+] Removed legacy error-analysis footer (variant)")

p.write_text(src)
print("[✓] Done. Saved to rect_gate_benchmark.py")
