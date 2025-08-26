# syntax=docker/dockerfile:1

FROM quay.io/fenicsproject/dolfinx:stable

# Set working directory inside container
WORKDIR /app

# Copy your source code into the container
COPY . .

# Default command: run your solver
CMD ["python3", "main.py"]
