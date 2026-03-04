#!/bin/bash
# =============================================================================
# gpu_check.sh — Full GPU + Ollama GPU diagnostics for WSL/Ubuntu
# Usage: chmod +x gpu_check.sh && ./gpu_check.sh
# =============================================================================

# --- Colours ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

PASS="${GREEN}[  PASS  ]${NC}"
FAIL="${RED}[  FAIL  ]${NC}"
WARN="${YELLOW}[  WARN  ]${NC}"
INFO="${CYAN}[  INFO  ]${NC}"
SECTION="${BLUE}${BOLD}"

ERRORS=0
WARNINGS=0

print_section() {
    echo ""
    echo -e "${SECTION}════════════════════════════════════════════════════════${NC}"
    echo -e "${SECTION}  $1${NC}"
    echo -e "${SECTION}════════════════════════════════════════════════════════${NC}"
}

pass()  { echo -e "${PASS}  $1"; }
fail()  { echo -e "${FAIL}  $1"; ((ERRORS++)); }
warn()  { echo -e "${WARN}  $1"; ((WARNINGS++)); }
info()  { echo -e "${INFO}  $1"; }

# =============================================================================
print_section "1. SYSTEM ENVIRONMENT"
# =============================================================================

# WSL check
if grep -qi "microsoft" /proc/version 2>/dev/null; then
    pass "Running inside WSL"
    WSL=true
else
    warn "Not running in WSL — some checks may differ"
    WSL=false
fi

# OS info
OS=$(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
info "OS: $OS"

# Kernel
KERNEL=$(uname -r)
info "Kernel: $KERNEL"

# =============================================================================
print_section "2. NVIDIA GPU HARDWARE CHECKS"
# =============================================================================

# Check nvidia-smi exists
if command -v nvidia-smi &>/dev/null; then
    pass "nvidia-smi found at: $(which nvidia-smi)"
else
    fail "nvidia-smi not found — NVIDIA drivers may not be installed on Windows host"
    echo "    → Fix: Update NVIDIA drivers on Windows from https://nvidia.com/drivers"
    echo "    → Then restart WSL: run 'wsl --shutdown' in Windows PowerShell"
fi

# Run nvidia-smi
echo ""
info "Running nvidia-smi..."
if nvidia-smi &>/dev/null; then
    pass "nvidia-smi ran successfully"
    echo ""
    nvidia-smi
    echo ""

    # Extract GPU details
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_CUDA=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
    GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
    VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader 2>/dev/null | head -1)

    pass "GPU detected:     $GPU_NAME"
    pass "VRAM total:       $GPU_VRAM"
    pass "VRAM used:        $VRAM_USED"
    pass "VRAM free:        $VRAM_FREE"
    info "Driver version:   $GPU_DRIVER"
    info "CUDA version:     $GPU_CUDA"
    info "GPU temperature:  ${GPU_TEMP}°C"
    info "GPU utilisation:  $GPU_UTIL"

    # Warn if driver is old
    DRIVER_MAJOR=$(echo $GPU_DRIVER | cut -d'.' -f1)
    if [ "$DRIVER_MAJOR" -lt 525 ]; then
        warn "Driver $GPU_DRIVER may be outdated — recommend 525+ for WSL2 CUDA support"
    else
        pass "Driver version $GPU_DRIVER is sufficient for WSL2 CUDA"
    fi

else
    fail "nvidia-smi failed — GPU not accessible from WSL"
    echo "    → Fix: Ensure NVIDIA drivers are installed on Windows (not inside WSL)"
    echo "    → Run 'wsl --shutdown' in PowerShell then reopen WSL"
fi

# =============================================================================
print_section "3. CUDA LIBRARY CHECKS"
# =============================================================================

# Check for CUDA in WSL paths
CUDA_PATHS=(
    "/usr/local/cuda/bin/nvcc"
    "/usr/local/cuda-12/bin/nvcc"
    "/usr/local/cuda-11/bin/nvcc"
)

NVCC_FOUND=false
for path in "${CUDA_PATHS[@]}"; do
    if [ -f "$path" ]; then
        pass "nvcc found at: $path"
        info "CUDA toolkit version: $($path --version 2>/dev/null | grep release | awk '{print $6}')"
        NVCC_FOUND=true
        break
    fi
done

if [ "$NVCC_FOUND" = false ]; then
    warn "nvcc (CUDA toolkit) not found — not required for Ollama but needed for building from source"
fi

# Check CUDA shared libraries
if ldconfig -p 2>/dev/null | grep -q "libcuda.so"; then
    pass "libcuda.so found in library path"
else
    warn "libcuda.so not found in ldconfig — Ollama may fall back to CPU"
    echo "    → Fix: sudo ldconfig /usr/local/cuda/lib64"
fi

if ldconfig -p 2>/dev/null | grep -q "libcudart.so"; then
    pass "libcudart.so (CUDA runtime) found"
else
    warn "libcudart.so not found — may cause issues with CUDA applications"
fi

# Check WSL CUDA lib
if [ "$WSL" = true ]; then
    if [ -f "/usr/lib/wsl/lib/libcuda.so.1" ]; then
        pass "WSL CUDA bridge library found: /usr/lib/wsl/lib/libcuda.so.1"
    else
        fail "WSL CUDA bridge library missing at /usr/lib/wsl/lib/libcuda.so.1"
        echo "    → Fix: Update NVIDIA drivers on Windows to a version that supports WSL2 CUDA"
    fi
fi

# =============================================================================
print_section "4. OLLAMA INSTALLATION CHECKS"
# =============================================================================

# Check ollama is installed
if command -v ollama &>/dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null)
    pass "Ollama installed: $OLLAMA_VERSION"
    info "Ollama path: $(which ollama)"
else
    fail "Ollama not installed"
    echo "    → Fix: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo -e "${RED}Cannot continue GPU+Ollama checks — install Ollama first${NC}"
    exit 1
fi

# Check ollama service is running
if pgrep -x "ollama" > /dev/null; then
    pass "Ollama service is running (PID: $(pgrep -x ollama))"
else
    warn "Ollama service is not running — starting it now..."
    ollama serve &>/dev/null &
    sleep 3
    if pgrep -x "ollama" > /dev/null; then
        pass "Ollama service started successfully"
    else
        fail "Failed to start Ollama service"
        echo "    → Try manually: ollama serve"
    fi
fi

# Check Ollama API is reachable
if curl -s http://localhost:11434 &>/dev/null; then
    pass "Ollama API reachable at http://localhost:11434"
else
    fail "Ollama API not reachable at http://localhost:11434"
    echo "    → Run: ollama serve"
fi

# =============================================================================
print_section "5. OLLAMA GPU DETECTION CHECKS"
# =============================================================================

info "Checking Ollama GPU detection via debug output..."
echo ""

# Run ollama serve briefly with debug and capture output
OLLAMA_DEBUG_LOG=$(OLLAMA_DEBUG=1 timeout 5 ollama serve 2>&1 || true)

# Check if CUDA is detected in Ollama logs
if echo "$OLLAMA_DEBUG_LOG" | grep -qi "cuda"; then
    pass "Ollama detected CUDA in debug output"
else
    fail "Ollama did not detect CUDA — will use CPU only"
    echo "    → Fix: export OLLAMA_LLM_LIBRARY=cuda_v12  (or cuda_v11)"
fi

if echo "$OLLAMA_DEBUG_LOG" | grep -qi "library=cuda"; then
    pass "Ollama is using CUDA inference library"
else
    warn "Ollama 'library=cuda' not confirmed in debug output"
fi

if echo "$OLLAMA_DEBUG_LOG" | grep -qi "gpu"; then
    pass "Ollama mentions GPU in debug output"
    # Extract and show the GPU inference line
    GPU_LINE=$(echo "$OLLAMA_DEBUG_LOG" | grep -i "inference compute" | head -1)
    if [ -n "$GPU_LINE" ]; then
        info "Inference compute: $GPU_LINE"
    fi
else
    fail "Ollama does not mention GPU in debug output"
fi

# Check environment variables
echo ""
info "Checking Ollama GPU environment variables..."

if [ -n "$OLLAMA_NUM_GPU" ]; then
    pass "OLLAMA_NUM_GPU is set: $OLLAMA_NUM_GPU"
else
    warn "OLLAMA_NUM_GPU not set — Ollama may auto-detect (usually fine, but set to 999 to force)"
    echo "    → Fix: export OLLAMA_NUM_GPU=999"
fi

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    pass "CUDA_VISIBLE_DEVICES is set: $CUDA_VISIBLE_DEVICES"
else
    warn "CUDA_VISIBLE_DEVICES not set — defaulting to all GPUs"
    echo "    → Fix: export CUDA_VISIBLE_DEVICES=0"
fi

if [ -n "$OLLAMA_LLM_LIBRARY" ]; then
    pass "OLLAMA_LLM_LIBRARY is set: $OLLAMA_LLM_LIBRARY"
else
    info "OLLAMA_LLM_LIBRARY not set — Ollama will auto-select (usually fine)"
fi

# =============================================================================
print_section "6. OLLAMA GPU LIVE INFERENCE TEST"
# =============================================================================

info "Running a live inference test with a small model to confirm GPU is used..."
echo ""

# Check if any model is available to test with
AVAILABLE_MODEL=$(ollama list 2>/dev/null | awk 'NR>1 {print $1; exit}')

if [ -z "$AVAILABLE_MODEL" ]; then
    warn "No models pulled yet — pulling a small test model (qwen3:0.6b)..."
    ollama pull qwen3:0.6b 2>/dev/null
    AVAILABLE_MODEL="qwen3:0.6b"
fi

info "Testing with model: $AVAILABLE_MODEL"

# Get VRAM before inference
VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

# Run a quick inference
RESPONSE=$(timeout 30 ollama run "$AVAILABLE_MODEL" "Reply with only the word CONFIRMED" 2>/dev/null)

# Get VRAM after inference
VRAM_AFTER=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

if [ -n "$RESPONSE" ]; then
    pass "Inference completed — model responded: '$RESPONSE'"
else
    fail "Inference test failed or timed out"
fi

# Compare VRAM usage
if [ -n "$VRAM_BEFORE" ] && [ -n "$VRAM_AFTER" ]; then
    VRAM_DIFF=$((VRAM_AFTER - VRAM_BEFORE))
    if [ "$VRAM_DIFF" -gt 100 ]; then
        pass "VRAM increased by ${VRAM_DIFF}MB during inference — GPU is being used"
    else
        fail "VRAM did not increase (before: ${VRAM_BEFORE}MB, after: ${VRAM_AFTER}MB) — model ran on CPU"
        echo "    → Fix: export OLLAMA_NUM_GPU=999 then restart ollama"
    fi
fi

# Check ollama ps for GPU %
info "Checking ollama ps for GPU usage..."
OLLAMA_PS=$(ollama ps 2>/dev/null)
echo "$OLLAMA_PS"

if echo "$OLLAMA_PS" | grep -qi "100%\|gpu"; then
    pass "ollama ps confirms GPU usage"
else
    warn "ollama ps did not confirm GPU — model may have been offloaded to CPU"
fi

# =============================================================================
print_section "7. SUMMARY & FIXES"
# =============================================================================

echo ""
if [ "$ERRORS" -eq 0 ] && [ "$WARNINGS" -eq 0 ]; then
    echo -e "${GREEN}${BOLD}  ✅ All checks passed — GPU is fully working with Ollama!${NC}"
elif [ "$ERRORS" -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}  ⚠️  Passed with $WARNINGS warning(s) — GPU likely working but review warnings above${NC}"
else
    echo -e "${RED}${BOLD}  ❌ $ERRORS error(s) and $WARNINGS warning(s) found — review failures above${NC}"
fi

echo ""
info "Quick fixes to apply if GPU not detected:"
echo ""
echo "    # Add these to ~/.bashrc then run: source ~/.bashrc"
echo "    export OLLAMA_NUM_GPU=999"
echo "    export CUDA_VISIBLE_DEVICES=0"
echo "    export OLLAMA_LLM_LIBRARY=cuda_v12"
echo ""
echo "    # Restart Ollama after applying"
echo "    sudo systemctl restart ollama"
echo "    # or"
echo "    pkill ollama && ollama serve"
echo ""