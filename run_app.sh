#!/bin/bash

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║       🚀 FRAUD DETECTION APP - SETUP & RUN SCRIPT                   ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Navigate to project
echo "📁 Step 1: Navigating to project directory..."
cd /Users/nitish157/fraud-detection
echo "   ✓ Current directory: $(pwd)"
echo ""

# Step 2: Activate virtual environment
echo "🔧 Step 2: Activating virtual environment..."
source venv/bin/activate
echo "   ✓ Virtual environment activated"
echo "   ✓ Prompt should show (venv)"
echo ""

# Step 3: Check dependencies
echo "📦 Step 3: Verifying dependencies..."
if python -c "import streamlit" 2>/dev/null; then
    echo "   ✓ streamlit is installed"
else
    echo "   ⚠ Installing streamlit..."
    pip install streamlit -q
fi

if python -c "import plotly" 2>/dev/null; then
    echo "   ✓ plotly is installed"
else
    echo "   ⚠ Installing plotly..."
    pip install plotly -q
fi

if python -c "from src.model_manager import ModelManager" 2>/dev/null; then
    echo "   ✓ model_manager is available"
fi
echo ""

# Step 4: Verify model files
echo "🤖 Step 4: Verifying model files..."
if [ -f "models/Random_Forest_tuned.pkl" ]; then
    size=$(ls -lh models/Random_Forest_tuned.pkl | awk '{print $5}')
    echo "   ✓ Random_Forest_tuned.pkl found ($size)"
else
    echo "   ❌ Random_Forest_tuned.pkl NOT found!"
    echo "   Run: python hyperparameter_tuning_fast.py"
    exit 1
fi

if [ -f "scaler.pkl" ]; then
    size=$(ls -lh scaler.pkl | awk '{print $5}')
    echo "   ✓ scaler.pkl found ($size)"
else
    echo "   ❌ scaler.pkl NOT found!"
    exit 1
fi
echo ""

# Step 5: Verify streamlit app
echo "📱 Step 5: Verifying Streamlit app..."
if [ -f "app/streamlit_app.py" ]; then
    echo "   ✓ app/streamlit_app.py found"
else
    echo "   ❌ app/streamlit_app.py NOT found!"
    exit 1
fi
echo ""

# Step 6: Show final instructions
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                         ✅ ALL READY!                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "🎯 Next Step: Start the Streamlit app"
echo ""
echo "   Run this command:"
echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   streamlit run app/streamlit_app.py"
echo "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   Then:"
echo "   • Browser will open at http://localhost:8501"
echo "   • Go to Tab 1: Single Prediction"
echo "   • Enter example values"
echo "   • Click '🔮 Predict Fraud' button"
echo "   • See the result!"
echo ""
echo "Or paste this complete command:"
echo ""
echo "   streamlit run app/streamlit_app.py"
echo ""
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                    🎉 YOU'RE ALL SET!                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""
