#!/usr/bin/env bash
# Force-clean old cancer-* containers and any stray port holders (orphan
# docker-proxy from a crashed prior run), then bring stack up fresh.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_FILE=docker-compose.yml

echo "== removing old containers =="
names=$(grep -oP 'container_name:\s*\K\S+' "$COMPOSE_FILE")
for n in $names; do
  if docker rm -f "$n" >/dev/null 2>&1; then
    echo "  removed $n"
  fi
done

echo "== freeing host ports =="
ports=$(grep -oP '^\s*- "\K[0-9]+(?=:)' "$COMPOSE_FILE")
for p in $ports; do
  pids=$(sudo lsof -ti tcp:"$p" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    echo "  port $p busy (pid $pids) -> kill"
    sudo kill -9 $pids
  fi
done

echo "== docker compose up -d =="
docker compose up -d
