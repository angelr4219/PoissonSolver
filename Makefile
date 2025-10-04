# ===== Makefile for fenicsx_poisson =====

# Use your container wrapper if available; else run locally
RUN := ./run_dolfinx.sh
ifeq ("$(wildcard $(RUN))","")
  RUN := python3
endif

# Module entrypoint and common paths
ENTRY   := -m src.main
OUTDIR  := results
DATE    := $(shell date +%Y%m%d-%H%M%S)

# macOS ParaView helper (tries to find a versioned app)
PARA := $(shell /usr/bin/find /Applications -maxdepth 1 -iname 'ParaView*.app' -print -quit)

.PHONY: help setup \
        run run-2d-dir run-2d-mix run-3d-dir run-3d-mix run-disk2d \
        test test-2d test-3d \
        view-2d view-3d clean really-clean

help:
	@echo "Make targets:"
	@echo "  setup             - create results/ directory"
	@echo "  run CASE=<case>   - run any case (2d_dirichlet | 2d_mixed | 3d_dirichlet | 3d_mixed | disk2d)"
	@echo "  run-2d-dir        - 2D manufactured solution, pure Dirichlet"
	@echo "  run-2d-mix        - 2D manufactured solution, Dirichlet + Neumann"
	@echo "  run-3d-dir        - 3D manufactured solution, pure Dirichlet"
	@echo "  run-3d-mix        - 3D manufactured solution, Dirichlet + Neumann"
	@echo "  run-disk2d        - 2D disk-in-box Laplace (hole=0.3V, outer=0V)"
	@echo "  test              - run both tests"
	@echo "  test-2d           - run 2D MMS test only"
	@echo "  test-3d           - run 3D MMS test only"
	@echo "  view-2d           - open last 2D output in ParaView (macOS)"
	@echo "  view-3d           - open last 3D output in ParaView (macOS)"
	@echo "  clean             - remove generated XDMF/H5 in results/"
	@echo ""
	@echo "Examples:"
	@echo "  make run-2d-mix"
	@echo "  make run CASE=3d_mixed"
	@echo "  make test"

setup:
	@mkdir -p $(OUTDIR)

# ----- Runs (explicit shortcuts)

run-2d-dir: setup
	$(RUN) $(ENTRY) --case 2d_dirichlet --outfile $(OUTDIR)/phi_2d_dir_$(DATE)

run-2d-mix: setup
	$(RUN) $(ENTRY) --case 2d_mixed     --outfile $(OUTDIR)/phi_2d_mix_$(DATE)

run-3d-dir: setup
	$(RUN) $(ENTRY) --case 3d_dirichlet --outfile $(OUTDIR)/phi_3d_dir_$(DATE)

run-3d-mix: setup
	$(RUN) $(ENTRY) --case 3d_mixed     --outfile $(OUTDIR)/phi_3d_mix_$(DATE)

run-disk2d: setup
	$(RUN) $(ENTRY) --case disk2d       --outfile $(OUTDIR)/phi_disk2d_$(DATE)

# ----- Generic run: pass CASE=<case>
run: setup
ifndef CASE
	$(error Please provide CASE, e.g. 'make run CASE=2d_mixed')
endif
	$(RUN) $(ENTRY) --case $(CASE) --outfile $(OUTDIR)/phi_$(CASE)_$(DATE)

# ----- Tests (use the MMS scripts under tests/)
test: test-2d test-3d

test-2d:
	$(RUN) tests/test_mms_2d.py

test-3d:
	$(RUN) tests/test_mms_3d.py

# ----- Quick viewers (macOS)
view-2d:
ifeq ("$(PARA)","")
	open -a "ParaView" $(shell ls -t $(OUTDIR)/*2d*.xdmf | head -n1)
else
	"$(PARA)/Contents/MacOS/paraview" $(shell ls -t $(OUTDIR)/*2d*.xdmf | head -n1)
endif

view-3d:
ifeq ("$(PARA)","")
	open -a "ParaView" $(shell ls -t $(OUTDIR)/*3d*.xdmf | head -n1)
else
	"$(PARA)/Contents/MacOS/paraview" $(shell ls -t $(OUTDIR)/*3d*.xdmf | head -n1)
endif

# ----- Cleanup
clean:
	@rm -f $(OUTDIR)/*.xdmf $(OUTDIR)/*.h5

really-clean: clean
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache build
