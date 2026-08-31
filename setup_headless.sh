#!/bin/bash
# Setup script for Eye Fatigue Detection in headless environment

echo "Setting up Eye Fatigue Detection System..."

# Create a dummy libGL.so.1 to work around the missing OpenGL library issue
export LD_LIBRARY_PATH="/opt/fake_libs:$LD_LIBRARY_PATH"

# Install dependencies
pip install opencv-python-headless>=4.8.0 numpy>=1.26.0

# Install mediapipe without automatic dependencies
pip install --no-deps mediapipe==0.10.35

# Install mediapipe dependencies (except opencv-contrib-python)
pip install absl-py~=2.3 certifi sounddevice~=0.5 flatbuffers~=25.9 matplotlib

echo "Setup complete!"
