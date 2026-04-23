#!/bin/bash

# Ensure Kaggle CLI is installed
if ! command -v kaggle &> /dev/null
then
    echo "Kaggle CLI could not be found. Please install it using 'pip install kaggle'."
    echo "Make sure your kaggle.json is placed in ~/.kaggle/."
    exit 1
fi

echo "Creating dataset directory..."
mkdir -p ./dataset

echo "Downloading dataset joykaihatu/image-caption-indonesia..."
kaggle datasets download -d joykaihatu/image-caption-indonesia -p ./dataset

echo "Unzipping dataset..."
unzip -o ./dataset/image-caption-indonesia.zip -d ./dataset

echo "Cleaning up zip file..."
rm ./dataset/image-caption-indonesia.zip

echo "Dataset setup complete! Images and metadata.csv should be in ./dataset"
