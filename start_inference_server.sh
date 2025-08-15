#!/bin/bash

# Start the GPU inference server

echo "Starting GPU Inference Server for Contract Generation"
echo "======================================================"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1  # Use both H100 GPUs
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Navigate to inference directory
cd inference

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r server_requirements.txt

# Start the server
echo "Starting server on port 8000..."
echo "Server will be available at http://0.0.0.0:8000"
echo ""
echo "API Documentation: http://0.0.0.0:8000/docs"
echo "Health Check: http://0.0.0.0:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python inference_server.py --host 0.0.0.0 --port 8000 --config server_config.yaml