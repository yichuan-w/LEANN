#!/bin/bash
# Diagnostic script for debugging CI hangs

echo "========================================="
echo "      CI HANG DIAGNOSTIC SCRIPT"
echo "========================================="
echo ""

echo "📅 Current time: $(date)"
echo "🖥️  Hostname: $(hostname)"
echo "👤 User: $(whoami)"
echo "📂 Working directory: $(pwd)"
echo ""

echo "=== PYTHON ENVIRONMENT ==="
python --version 2>&1 || echo "Python not found"
pip list 2>&1 | head -20 || echo "pip not available"
echo ""

echo "=== PROCESS INFORMATION ==="
echo "Current shell PID: $$"
echo "Parent PID: $PPID"
echo ""

echo "All Python processes:"
ps aux | grep -E "[p]ython" || echo "No Python processes"
echo ""

echo "All pytest processes:"
ps aux | grep -E "[p]ytest" || echo "No pytest processes"
echo ""

echo "Embedding server processes:"
ps aux | grep -E "[e]mbedding_server" || echo "No embedding server processes"
echo ""

echo "Zombie processes:"
ps aux | grep "<defunct>" || echo "No zombie processes"
echo ""

echo "=== NETWORK INFORMATION ==="
echo "Network listeners on typical embedding server ports:"
ss -ltn 2>/dev/null | grep -E ":555[0-9]|:556[0-9]" || netstat -ltn 2>/dev/null | grep -E ":555[0-9]|:556[0-9]" || echo "No listeners on embedding ports"
echo ""

echo "All network listeners:"
ss -ltn 2>/dev/null | head -20 || netstat -ltn 2>/dev/null | head -20 || echo "Cannot get network info"
echo ""

echo "=== FILE DESCRIPTORS ==="
echo "Open files for current shell:"
lsof -p $$ 2>/dev/null | head -20 || echo "lsof not available"
echo ""

if [ -d "/proc/$$" ]; then
    echo "File descriptors for current shell (/proc/$$/fd):"
    ls -la /proc/$$/fd 2>/dev/null | head -20 || echo "Cannot access /proc/$$/fd"
    echo ""
fi

echo "=== SYSTEM RESOURCES ==="
echo "Memory usage:"
free -h 2>/dev/null || vm_stat 2>/dev/null || echo "Cannot get memory info"
echo ""

echo "Disk usage:"
df -h . 2>/dev/null || echo "Cannot get disk info"
echo ""

echo "CPU info:"
nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "Cannot get CPU info"
echo ""

echo "=== PYTHON SPECIFIC CHECKS ==="
python -c "
import sys
import os
print(f'Python executable: {sys.executable}')
print(f'Python path: {sys.path[:3]}...')
print(f'Environment PYTHONPATH: {os.environ.get(\"PYTHONPATH\", \"Not set\")}')
print(f'Site packages: {[p for p in sys.path if \"site-packages\" in p][:2]}')
" 2>&1 || echo "Cannot run Python diagnostics"
echo ""

echo "=== ZMQ SPECIFIC CHECKS ==="
python -c "
try:
    import zmq
    print(f'ZMQ version: {zmq.zmq_version()}')
    print(f'PyZMQ version: {zmq.pyzmq_version()}')
    ctx = zmq.Context.instance()
    print(f'ZMQ context instance: {ctx}')
except Exception as e:
    print(f'ZMQ check failed: {e}')
" 2>&1 || echo "Cannot check ZMQ"
echo ""

echo "=== PYTEST CHECK ==="
pytest --version 2>&1 || echo "pytest not found"
echo ""

echo "=== END OF DIAGNOSTICS ==="
echo "Generated at: $(date)"
