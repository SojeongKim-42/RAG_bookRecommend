#!/bin/bash

# Activate environment
eval "$(~/miniforge3/bin/mamba shell hook --shell bash)"
mamba activate bkms

# Clear streamlit cache
echo "Clearing Streamlit cache..."
rm -rf ~/.streamlit/cache

# Run streamlit with cache clearing
echo "Starting Streamlit app..."
streamlit run streamlit_app.py --server.runOnSave true

