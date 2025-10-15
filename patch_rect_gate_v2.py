import re, sys, pathlib
p = pathlib.Path("rect_gate_benchmark.py")
src = p.read_text()

# --- robust pattern for calling solve_laplace in run_once ---
pat1 = re.compile(r'^(?P<indent>\s*)uh\s*=\s*solve_laplace\([^)]*\)\s*(?:#.*)?$', re.MULTILINE)

m = pat1.search(src)
insert_pos = None
indent = None
if m:
    insert_pos = m.end()
    indent = m.group('indent')

# Fallback: insert before the XDMF write in run_once
if insert_pos is None:
    pat2 = re.compile(r'^(?P<indent>\s*)with\s+io\.XDMFFile\(domain\.comm,\s*f"\{outprefix\}\.xdmf",\s*"w"\)\s+as\s+xdmf:', re.MULTILINE)
    m2 = pat2.search(src)
    if m2:
        insert_pos = m2.start()
        indent = m2.group('indent')

if insert_pos is None:
    print("[-] Could not locate insertion point (solve_laplace call or XDMF write).", file=sys.stderr)
    sys.exit(1)

err_block = f'''
{indent}# --- Global error metrics vs analytic rectangular-gate potential ---
{indent}from errors import report_errors
{indent}from exact_rect_gates import phi0_rect_three_gates_factory
{indent}import os
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
'''

src = src[:insert_pos] + "\n" + err_block + src[insert_pos:]
p.write_text(src)
print("[+] Inserted error-analysis block.")
