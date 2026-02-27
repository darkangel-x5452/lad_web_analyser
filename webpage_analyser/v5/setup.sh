#!/bin/bash

# Webpage Analyzer Setup Script
# This script sets up the environment for the webpage analyzer

echo "=================================================="
echo "  Webpage Image Analyzer - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi
echo "✓ Python 3 found"
echo ""

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt --break-system-packages
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: Some packages may have failed to install"
fi
echo ""

# Install Playwright browsers
echo "🌐 Installing Playwright Chromium browser..."
playwright install chromium
if [ $? -ne 0 ]; then
    echo "❌ Playwright installation failed"
    exit 1
fi
echo "✓ Playwright installed"
echo ""

# Setup instructions
echo "=================================================="
echo "  Setup Complete! 🎉"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Choose a model and set up API key:"
echo ""
echo "   Option A - Claude (Best Accuracy):"
echo "   • Get API key: https://console.anthropic.com/"
echo "   • Run: export ANTHROPIC_API_KEY='your-key-here'"
echo ""
echo "   Option B - Gemini (Fast & Free):"
echo "   • Get API key: https://aistudio.google.com/app/apikey"
echo "   • Run: export GOOGLE_API_KEY='your-key-here'"
echo ""
echo "   Option C - Ollama (Local & Free):"
echo "   • Install: https://ollama.ai/download"
echo "   • Run: ollama serve"
echo "   • Run: ollama pull llava"
echo ""
echo "2. Test the installation:"
echo "   python webpage_analyzer.py"
echo ""
echo "3. Try different examples:"
echo "   python example_usage.py"
echo ""
echo "=================================================="
echo ""
echo "For more information, see README.md"
echo ""
