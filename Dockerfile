FROM dolfinx/dolfinx:stable

WORKDIR /app
COPY . .

# Ensure dolfinx and gmsh/matplotlib/etc are available
RUN pip3 install --no-cache-dir gmsh matplotlib pyvista pandas pytest

# Important: so Python finds src/
ENV PYTHONPATH=/app

CMD ["python3", "main.py"]
