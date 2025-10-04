#!/bin/bash

# Start script for Exoplanet AI Classifier
echo "🚀 Starting Exoplanet AI Classifier..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found"
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "models/trained_model.pkl" ]; then
    echo "❌ Trained model not found. Training..."
    python train_model.py
else
    echo "✅ Trained model found"
fi

# Start the application
echo "🌐 Starting web server on http://localhost:8001"
echo "Press Ctrl+C to stop"
echo "=================================="

python main.py
