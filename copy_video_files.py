# Given an input and output directory, this script copies video files from the input directory to the output directory.

# Import the necessary libraries
import os
import shutil

# Define the input and output directories
input_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\VideoFlash"
output_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST\HI"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Copy all video files from the input directory to the output directory
for file in os.listdir(input_dir):
    # Only copy files that end with "HI.flv"
    if file.endswith("HI.flv"):
        input_file = os.path.join(input_dir, file)
        output_file = os.path.join(output_dir, file)
        shutil.copy2(input_file, output_file)