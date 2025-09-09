# syntax=docker/dockerfile:1

FROM dolfinx/dolfinx:stable

# Set working directory
WORKDIR /app

# Copy repo contents into container
COPY . .

# Make sure Python finds your src/ package
ENV PYTHONPATH=/app

# Install extra Python dependencies you need for validation & plots
RUN pip3 install --no-cache-dir gmsh matplotlib pandas pyvista pytest

# Default entrypoint
CMD ["python3", "main.py"]
