#!/bin/bash

# Check if a directory path is provided as an argument
if [ $# -eq 0 ]; then
  echo "Usage: $0 <directory_path>"
  exit 1
fi

# Get the directory path from the command line argument
directory="$1"

# Check if the provided path is a directory
if [ ! -d "$directory" ]; then
  echo "Error: '$directory' is not a valid directory."
  exit 1
fi

# Get the directory name without the full path
dir_name=$(basename "$directory")

mkdir "./$dir_name"

# Create a zip file for each subdirectory
for subdirectory in "$directory"/*; do
  if [ -d "$subdirectory" ]; then
    sub_dir_name=$(basename "$subdirectory")
    zip_file="$dir_name/$sub_dir_name.zip"
    zip -r "$zip_file" "$subdirectory"

    # Check if the zip operation was successful
    if [ $? -eq 0 ]; then
      echo "Successfully zipped '$subdirectory' to '$zip_file'."
    else
      echo "Error: Failed to zip '$subdirectory'."
    fi
  fi
done
