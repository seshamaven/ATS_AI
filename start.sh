#!/bin/bash

# ATS AI Application Startup Script for Railpack
echo "Starting ATS AI Application..."

# Set environment variables
export FLASK_APP=backend/ats_api.py
export FLASK_ENV=production
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Debug: Show environment variables
echo "=== Environment Variables Debug ==="
echo "OPENAI_API_KEY: ${OPENAI_API_KEY:+SET}"
echo "AZURE_OPENAI_API_KEY: ${AZURE_OPENAI_API_KEY:+SET}"
echo "AZURE_OPENAI_ENDPOINT: ${AZURE_OPENAI_ENDPOINT:+SET}"
echo "MYSQL_HOST: ${MYSQL_HOST:+SET}"
echo "MYSQL_USER: ${MYSQL_USER:+SET}"
echo "MYSQL_PASSWORD: ${MYSQL_PASSWORD:+SET}"
echo "MYSQL_DATABASE: ${MYSQL_DATABASE:+SET}"
echo "PINECONE_API_KEY: ${PINECONE_API_KEY:+SET}"
echo "===================================="

# Install dependencies
echo "Installing Python dependencies..."
pip install -r backend/requirements_ats.txt

# Download spaCy model if not present
echo "Setting up spaCy model..."
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p backend/uploads
mkdir -p logs

# Check if .env exists in backend directory
if [ ! -f "backend/.env" ]; then
    echo "WARNING: backend/.env file not found!"
    echo "Please ensure environment variables are set in Railpack dashboard"
fi

# Start the Flask application
echo "Starting Flask application on port $PORT..."
cd backend
python ats_api.py
