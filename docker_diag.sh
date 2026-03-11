#!/usr/bin/env bash
set -euo pipefail

echo "== docker version =="
docker version || exit 1

echo
echo "== docker daemon check =="
docker info >/dev/null && echo "docker daemon ok"

echo
echo "== local images =="
docker image ls

echo
echo "== run hello-world without pulling =="
set +e
docker run --rm --pull=never hello-world
status=$?
set -e
echo "exit code: $status"

echo
echo "== registry reachability =="
curl -I https://registry-1.docker.io/v2/

echo
echo "== pull hello-world =="
docker pull hello-world

echo
echo "== run hello-world after pull =="
docker run --rm hello-world
