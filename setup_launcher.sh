#!/bin/bash

# Exit script on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Define model directory variables
BASE_DIR=$(pwd)/models
PRECISION_SELECTOR="INT4" # Change this to the desired precision (e.g., FP16, FP32)
MODEL_DIR="$BASE_DIR/$PRECISION_SELECTOR"

# Create and activate a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv

    echo "Activating virtual environment..."
    source venv/bin/activate

    # Ensure pip is up to date
    pip install --upgrade pip

    # Install necessary dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Virtual environment already exists. Activating it..."
    source venv/bin/activate
fi

# Check if the model already exists
if [ ! -f "$MODEL_DIR/openvino_language_model.xml" ]; then
    echo "Model not found in $MODEL_DIR. Exporting model..."
    mkdir -p "$MODEL_DIR"
    optimum-cli export openvino \
        -m "mistral-community/pixtral-12b" \
        --weight-format "$(echo $PRECISION_SELECTOR | tr '[:upper:]' '[:lower:]')" \
        "$MODEL_DIR"
else
    echo "Model already exists in $MODEL_DIR."
fi

# Run Streamlit and Python server in parallel
echo "Starting the Streamlit frontend..."
streamlit run frontend.py &

echo "Starting the Python server..."
python3 server.py &

# Wait for both processes to finish
wait
