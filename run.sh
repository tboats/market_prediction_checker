#!/bin/bash
# Pipeline execution script for prediction-evaluator

# Get directory of the script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
WS_ROOT="$(dirname "$(dirname "$(dirname "$DIR")")")"

echo "==============================================="
echo "📈 Running Stock Market Prediction Evaluator..."
echo "==============================================="

# Activate virtual environment
if [ -d "$WS_ROOT/.venv" ]; then
    source "$WS_ROOT/.venv/bin/activate"
else
    echo "❌ Error: Virtual environment not found at $WS_ROOT/.venv"
    exit 1
fi

# Run evaluation
python3 "$DIR/evaluate.py"
if [ $? -ne 0 ]; then
    echo "❌ Error: Evaluation failed."
    exit 1
fi

# Run dashboard generator
python3 "$DIR/generate_dashboard.py"
if [ $? -ne 0 ]; then
    echo "❌ Error: Dashboard generation failed."
    exit 1
fi

echo "==============================================="
echo "✅ Pipeline completed successfully!"
echo "   📊 Results:  $DIR/results.json"
echo "   🖼️  Chart:    $DIR/actual_vs_predicted.png"
echo "   🖥️  Dashboard:$DIR/dashboard.html"
echo "==============================================="
