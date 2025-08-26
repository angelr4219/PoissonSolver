# syntax=docker/dockerfile:1

FROM dolfinx/dolfinx:stable
WORKDIR /app
COPY . .
CMD ["python3", "main.py"]
