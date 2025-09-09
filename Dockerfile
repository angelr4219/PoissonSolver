# syntax=docker/dockerfile:1
FROM dolfinx/dolfinx:stable

WORKDIR /app
COPY . .

# Extra Python deps your scripts/tests use
RUN pip3 install --no-cache-dir gmsh matplotlib pandas pyvista pytest

# Make your src/ visible as a package
ENV PYTHONPATH=/app

CMD ["python3", "main.py"]
