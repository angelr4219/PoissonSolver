# --- User-tunable defaults ---
H      ?= 0.03
LX     ?= 1.0
LY     ?= 1.0
LZ     ?= 1.0
R      ?= 0.20
VINC   ?= 0.20
VGATE  ?= 0.20
Q      ?= 1.0
Z_SEM  ?= 0.20

# Output names
OUT2D  ?= results/phi_disk2d
OUTROD ?= results/phi_rod2d
OUT3D  ?= results3d

# Wrapper (already in your repo)
RUN    := ./run_dolfinx.sh

.PHONY: help disk2d rod2d hole3d point3d clean ls

help:
	@echo "Targets:"
	@echo "  make disk2d      # disk in a box, Dirichlet gate (2D)"
	@echo "  make rod2d       # rectangular rod with circular bore (2D)"
	@echo "  make hole3d      # 3D box with cylindrical hole (Dirichlet gate)"
	@echo "  make point3d     # 3D point charge + 2D imprint (if script exists)"
	@echo "Vars override: H=0.02 R=0.15 VINC=0.1 LX=1 LY=1 LZ=1 ..."

disk2d:
	@mkdir -p results
	$(RUN) src/cli/solve_shapes.py \
	  --mode disk2d \
	  --Lx $(LX) --Ly $(LY) \
	  --R $(R) \
	  --h $(H) \
	  --Vinc $(VINC) \
	  --outfile $(OUT2D)

rod2d:
	@mkdir -p results
	$(RUN) src/cli/solve_shapes.py \
	  --mode rod2d \
	  --Lx $(LX) --Ly $(LY) \
	  --R $(R) \
	  --h $(H) \
	  --Vinc $(VINC) \
	  --outfile $(OUTROD)

hole3d:
	@mkdir -p results
	$(RUN) poisson_rod_with_hole_3d.py \
	  --Lx $(LX) --Ly $(LY) --Lz $(LZ) \
	  --R $(R) \
	  --h $(H) \
	  --Vgate $(VGATE) \
	  --outfile $(OUT3D)

# NOTE: if poisson_point_imprint.py doesn't parse CLI args, it's fine—
# Python will ignore them unless argparse is used.
point3d:
	@mkdir -p results
	$(RUN) poisson_point_imprint.py \
	  --Lx $(LX) --Ly $(LY) --Lz $(LZ) \
	  --h $(H) \
	  --q $(Q) \
	  --z_sem $(Z_SEM) \
	  --outfile results/phi_point

clean:
	rm -rf results *.xdmf *.h5 *.vtu *.pvtu *.pvd *.log

ls:
	@echo "Listing host dir via Alpine (verifies mount):"; \
	docker run --rm -v "$$(pwd)":/app -w /app alpine ls -l
