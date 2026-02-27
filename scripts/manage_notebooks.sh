#!/bin/bash
# Manages pairing and syncing of Jupyter notebooks and Python scripts
# in the data_analysis folder using jupytext.

# --- Configuration ---
TARGET_DIR="data_analysis"

# --- Functions ---
pair_files() {
    echo "--- Pairing all notebooks with Python scripts ---"
    find . -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook; do
        echo "Pairing: $notebook"
        poetry run jupytext --set-formats ipynb,py:percent "$notebook"
    done
    echo "--- Pairing complete. ---"
}

sync_files() {
    echo "--- Syncing all paired notebooks and scripts ---"
    # Find all .ipynb files and sync them with their paired scripts
    find . -name "*.ipynb" -print0 | while IFS= read -r -d $'\0' notebook; do
        echo "Syncing: $notebook"
        poetry run jupytext --sync "$notebook"
    done
    echo "--- Synchronization complete. ---"
}

# --- Main Logic ---
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' not found."
    exit 1
fi

cd "$TARGET_DIR" || exit 1

case "$1" in
    pair)
        pair_files
        ;;
    sync)
        sync_files
        ;;
    *)
        echo "Usage: $(basename "$0") {pair|sync}"
        echo "  pair: Sets up the pairing between .ipynb and .py files. (One-time setup)"
        echo "  sync: Synchronizes the paired files, updating the older file with the newer one."
        exit 1
        ;;
esac
