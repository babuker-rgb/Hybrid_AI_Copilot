#!/bin/bash

# ================================================================
# Hybrid AI Tablet Optimization - Startup Script
# Nile Valley University · Sudan · v29.28-R32
# ================================================================

echo "🚀 Starting Hybrid AI Tablet Optimization Framework"

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$python_version" != "3.9" ] && [ "$python_version" != "3.10" ] && [ "$python_version" != "3.11" ]; then
    echo "⚠️  Python 3.9+ recommended (found $python_version)"
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create results directory
mkdir -p ./results
mkdir -p ./models
mkdir -p ./logs

# Run application
echo "🌐 Starting Streamlit app..."
streamlit run app_integration.py --server.port=8501 --server.address=0.0.0.0
