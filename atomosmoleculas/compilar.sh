#!/usr/bin/env sh

# Compile the atomos simulation using g++ with OpenMP and OpenCV

g++ atomos.cpp -o circles2 -std=c++17 `pkg-config --cflags --libs opencv4` -fopenmp

