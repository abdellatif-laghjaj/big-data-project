#!/bin/bash
# Script to download videos and store in a specific directory

# Define directory to store downloaded videos
download_dir="/home/akame/Downloads/Video Resources"

# Ensure directory exists
mkdir -p "$download_dir"

# List of URLs
urls=(
  "https://v1.cdnpk.net/videvo_files/video/premium/getty_108/large_watermarked/istock-993951822_preview.mp4"
  "https://v1.cdnpk.net/videvo_files/video/premium/getty_132/large_watermarked/istock-962828842_preview.mp4"
)

# Loop through URLs and download each video
for url in "${urls[@]}"; do
  wget -P "$download_dir" "$url"
done
