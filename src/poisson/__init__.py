from .refinement import RefinementBox, build_refined_mesh_3d, mark_refinement_regions
from .geom3d import box
from .geom2d import rectangle
from .geom_sphere import sphere_refined_box, sphere_in_box
from .fem_solve import assemble_solve_poisson
from .materials import piecewise_eps_DG0_2regions
from .fft_solve import fft_poisson_3d, gaussian_rho_grid, sphere_charge_shell_grid, evaluate_fft_phi_at_points
from .analytics import phi_conducting_sphere, phi_point_charge, EPS0
from .io_utils import write_field_xdmf
