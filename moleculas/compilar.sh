#!/usr/bin/env sh

# Compile the atomos simulation using g++ with OpenMP and OpenCV

g++ moleculas3.cpp -o simulacion -std=c++17 `pkg-config --cflags --libs opencv4` -fopenmp

