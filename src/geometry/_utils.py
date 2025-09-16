from .helpers import (
    gmsh_model_to_mesh,
    tag_boundaries if 'tag_boundaries' in dir() else None,
)
 PY
cat > src/geometry/_utils.py <<'PY'
from .helpers import (
    gmsh_model_to_mesh,
    tag_boundaries if 'tag_boundaries' in dir() else None,
)
# If you have more names in helpers you rely on, export them here too.
__all__ = [name for name in globals() if not name.startswith("_")]
