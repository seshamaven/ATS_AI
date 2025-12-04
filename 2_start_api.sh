#!/bin/bash
# 2. Start ATS API Server (Linux version)

echo "=========================================="
echo "Step 2: Starting ATS API Server"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import flask" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements_ats.txt
    echo ""
    echo "Downloading spaCy NLP model..."
    python -m spacy download en_core_web_sm 2>/dev/null || true
fi

# Create uploads directory if not exists
if [ ! -d "uploads" ]; then
    echo "Creating uploads directory..."
    mkdir -p uploads
fi

# Load environment variables from .env or env file
load_env_file() {
    local file=$1
    if [ -f "$file" ]; then
        echo "Loading environment variables from $file..."
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip comments and empty lines
            [[ "$line" =~ ^[[:space:]]*# ]] && continue
            [[ -z "${line// }" ]] && continue
            # Export variable if it's in KEY=VALUE format
            if [[ "$line" =~ ^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then
                export "$line"
            fi
        done < "$file"
    fi
}

# Try loading .env first, then env
load_env_file ".env"
if [ -z "$OPENAI_API_KEY" ]; then
    load_env_file "env"
fi

# Set environment variables for MySQL (with fallback defaults)
export ATS_MYSQL_HOST=${ATS_MYSQL_HOST:-${MYSQL_HOST:-localhost}}
export ATS_MYSQL_USER=${ATS_MYSQL_USER:-${MYSQL_USER:-root}}
export ATS_MYSQL_PASSWORD=${ATS_MYSQL_PASSWORD:-${MYSQL_PASSWORD:-Mst@2026}}
export ATS_MYSQL_DATABASE=${ATS_MYSQL_DATABASE:-${MYSQL_DATABASE:-ats_db}}
export ATS_MYSQL_PORT=${ATS_MYSQL_PORT:-${MYSQL_PORT:-3306}}
export USE_PINECONE=${USE_PINECONE:-False}

# OpenAI API Key (must be set - no default placeholder)
if [ -z "$OPENAI_API_KEY" ]; then
    echo ""
    echo "WARNING: OPENAI_API_KEY not found in environment!"
    echo "Please set it in .env or env file, or export it before running this script."
    echo ""
fi

echo ""
echo "=========================================="
echo "Starting ATS API on http://localhost:5002"
echo "=========================================="
echo ""
echo "Available endpoints:"
echo "  - POST /api/searchResume"
echo "  - POST /api/searchResumes"
echo "  - POST /api/processResume"
echo "  - POST /api/profileRankingByJD"
echo "  - GET  /api/candidate/<id>"
echo "  - GET  /health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python ats_api.py

