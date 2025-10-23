#!/bin/bash

# ATS AI Application Startup Script for Railpack
echo "Starting ATS AI Application..."

# Set environment variables
export FLASK_APP=backend/ats_api.py
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements_ats.txt

# Download spaCy model if not present
echo "Setting up spaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p backend/uploads
mkdir -p logs

# Start the Flask application
echo "Starting Flask application on port $PORT..."
cd backend
python ats_api.py
