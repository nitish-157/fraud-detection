#!/bin/bash

echo "🚀 Starting Fraud Detection App..."
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    python -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
fi

# Install requirements
pip install -q -r requirements.txt 2>/dev/null

# Run app
echo "✅ Ready! Starting Streamlit app..."
echo ""
streamlit run app/streamlit_app.py
echo ""
