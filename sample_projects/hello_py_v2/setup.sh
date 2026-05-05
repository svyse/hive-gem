#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

VENV_DIR="venv"
KERNEL_NAME="hello_py_v2_kernel"
DISPLAY_NAME="Python (hello_py_v2)"
REQUIREMENTS_FILE="requirements.txt"

echo "Starting Jupyter setup for project: $(pwd)"

# 1. Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment '$VENV_DIR'..."
    python -m venv "$VENV_DIR"
else
    echo "Virtual environment '$VENV_DIR' already exists."
fi

# 2. Activate the virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# 3. Install dependencies from requirements.txt
echo "Installing dependencies from '$REQUIREMENTS_FILE'..."
pip install -r "$REQUIREMENTS_FILE"

# 4. Install ipykernel and register it with Jupyter
echo "Installing ipykernel and registering kernel '$KERNEL_NAME'..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo "Jupyter setup complete."
echo "To run Jupyter Notebook:"
echo "1. Activate the virtual environment: source $VENV_DIR/bin/activate"
echo "2. Launch Jupyter: jupyter notebook"
echo "You should then be able to select the '$DISPLAY_NAME' kernel in your notebook."

# Deactivate the virtual environment (optional, but good practice)
deactivate
