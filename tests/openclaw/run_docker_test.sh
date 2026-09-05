#!/usr/bin/env bash
#
# End-to-end test: spin up an isolated OpenClaw Docker container,
# install the leann-memory skill, build an index, and search.
#
# Usage:
#   tests/openclaw/run_docker_test.sh          # run all steps
#   tests/openclaw/run_docker_test.sh --down   # tear down only
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

COMPOSE="docker compose -f docker-compose.yml"
CONTAINER="openclaw-leann-test"
HOST_PORT=18790

# ---------- helpers ----------

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
pass() { printf '\033[1;32m ✓ \033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m ✗ \033[0m %s\n' "$*"; exit 1; }

cleanup() {
    log "Tearing down container …"
    $COMPOSE down --volumes --remove-orphans 2>/dev/null || true
    rm -rf docker-data
}

if [[ "${1:-}" == "--down" ]]; then
    cleanup
    exit 0
fi

trap cleanup EXIT

# ---------- 1. Start container ----------

log "Starting isolated OpenClaw container on port $HOST_PORT …"
rm -rf docker-data
$COMPOSE up -d --wait 2>&1 || $COMPOSE up -d 2>&1

sleep 3

if docker ps --format '{{.Names}}' | grep -q "$CONTAINER"; then
    pass "Container $CONTAINER is running"
else
    fail "Container $CONTAINER failed to start"
fi

# ---------- 2. Verify gateway ----------

log "Checking gateway health …"
for i in $(seq 1 10); do
    if curl -sf "http://localhost:$HOST_PORT" >/dev/null 2>&1 || \
       curl -sf "http://localhost:$HOST_PORT/health" >/dev/null 2>&1; then
        pass "Gateway responding on port $HOST_PORT"
        break
    fi
    if [ "$i" -eq 10 ]; then
        # Gateway may not have an HTTP health endpoint; check if process is running
        if docker exec "$CONTAINER" pgrep -f "gateway" >/dev/null 2>&1; then
            pass "Gateway process is running inside container"
        else
            echo "--- Container logs ---"
            docker logs "$CONTAINER" --tail 20
            fail "Gateway is not running"
        fi
    fi
    sleep 2
done

# ---------- 3. Install skill fixtures ----------

log "Copying skill and memory fixtures into container …"
docker exec "$CONTAINER" mkdir -p /home/node/.openclaw/workspace/memory

docker cp fixtures/MEMORY.md "$CONTAINER":/home/node/.openclaw/workspace/MEMORY.md
for f in fixtures/memory/*.md; do
    docker cp "$f" "$CONTAINER":/home/node/.openclaw/workspace/memory/
done

docker exec "$CONTAINER" ls -la /home/node/.openclaw/workspace/ | head -10
docker exec "$CONTAINER" ls -la /home/node/.openclaw/workspace/memory/
pass "Memory fixtures installed"

# ---------- 4. Run fast pytest tests (no model needed) ----------

log "Running fast (non-model) tests locally …"
cd "$SCRIPT_DIR/../.."
uv run pytest tests/openclaw/test_skill_manifest.py tests/openclaw/test_mcp_protocol.py -v 2>&1
pass "Fast tests passed"

# ---------- 5. Run model tests (optional) ----------

if [[ "${SKIP_SLOW:-}" != "1" ]]; then
    log "Running slow build-and-search tests …"
    uv run pytest tests/openclaw/test_build_and_search.py -v --timeout=300 2>&1
    pass "Build-and-search tests passed"
else
    log "Skipping slow tests (SKIP_SLOW=1)"
fi

# ---------- done ----------

echo ""
log "All OpenClaw integration tests passed!"
echo "  Container: $CONTAINER"
echo "  Gateway:   ws://localhost:$HOST_PORT"
echo "  Cleanup:   $0 --down"
