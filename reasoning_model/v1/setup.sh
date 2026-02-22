#!/bin/bash
# ╔══════════════════════════════════════════════════════╗
# ║   THINKING ENGINE — ONE-CLICK SETUP (macOS/Linux)   ║
# ╚══════════════════════════════════════════════════════╝

set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   🧠  LLM Thinking Engine — Setup Script            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# Step 1: Python deps
echo "📦 Installing Python dependencies..."
pip install rich --quiet

# Step 2: Ollama
if ! command -v ollama &> /dev/null; then
    echo ""
    echo "🦙 Ollama not found. Installing..."
    echo "   (Ollama is FREE, open-source, runs models locally)"
    echo ""
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "   For macOS: download from https://ollama.com/download"
        echo "   Or run: brew install ollama"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.ai/install.sh | sh
    fi
else
    echo "✅ Ollama already installed: $(ollama --version)"
fi

# Step 3: Pull model
echo ""
echo "🦙 Pulling llama3.2 model (4GB download, one-time)..."
echo "   Other good free options: mistral, phi3.5, gemma2:9b"
ollama pull llama3.2

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║   ✅ SETUP COMPLETE!                                 ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  Run the sports demo:                                ║"
echo "║  python thinking_engine.py --demo sports             ║"
echo "║                                                      ║"
echo "║  Run interactively with your own data:               ║"
echo "║  python thinking_engine.py --interactive             ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
