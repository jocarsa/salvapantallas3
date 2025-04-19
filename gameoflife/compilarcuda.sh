#!/usr/bin/env sh

# Compile the simulation using nvcc with CUDA, OpenMP, and OpenCV

nvcc simulacion.cu -o simulacion -std=c++17 `pkg-config --cflags --libs opencv4` -Xcompiler -fopenmp -lcuda -lnppc -lnppial -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lnpps

