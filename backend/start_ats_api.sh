#!/bin/bash
# Shell script to start ATS API on Linux/Mac

echo "=========================================="
echo "Starting ATS API Server"
echo "=========================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
echo "Checking dependencies..."
if ! python -c "import flask" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements_ats.txt
    python -m spacy download en_core_web_sm
fi

# Create uploads directory if not exists
if [ ! -d "uploads" ]; then
    echo "Creating uploads directory..."
    mkdir uploads
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "WARNING: .env file not found!"
    echo "Please create .env file from env_ats_template.txt"
    echo
    read -p "Press enter to continue..."
fi

# Start the API
echo
echo "Starting ATS API on port 5002..."
echo
python ats_api.py

