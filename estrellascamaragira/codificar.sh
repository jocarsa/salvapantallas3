#!/bin/bash

# Create codificado directory if it doesn't exist
mkdir -p codificado

# Loop through all .mp4 files in the current directory
for file in *.mp4; do
  # Skip if no .mp4 files are found
  [ -e "$file" ] || continue

  # Get the base filename without extension
  base="${file%.mp4}"

  # Output filename
  output="codificado/${base}_codificado.mp4"

  # Encode using ffmpeg
  ffmpeg -i "$file" "$output"
done

