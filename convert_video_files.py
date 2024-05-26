# This script converts video files from .flv to .mp4
import os
from moviepy.editor import VideoFileClip
import sys
from contextlib import contextmanager
# import logging

# class NullLogger(logging.Handler):
#     def emit(self, record):
#         pass

# logger = logging.getLogger("moviepy")
# logger.setLevel(logging.CRITICAL)
# logger.addHandler(logging.NullHandler())
# Add NullHandler to avoid any logs being processed
# null_handler = logging.NullHandler()
# logger.addHandler(null_handler)

# Configure the logger to direct logs to os.devnull
# file_handler = logging.FileHandler(os.devnull, "w")
# logger.addHandler(file_handler)

# logger.addHandler(NullLogger())

# @contextmanager
# def suppress_stdout_stderr():
#     """A context manager that redirects stdout and stderr to devnull"""
#     with open(os.devnull, 'w') as fnull:
#         old_stdout, old_stderr = sys.stdout, sys.stderr
#         sys.stdout, sys.stderr = fnull, fnull
#         try:
#             yield
#         finally:
#             sys.stdout, sys.stderr = old_stdout, old_stderr

def main():
    # Convert all files in a directory
    input_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\VideoFlash"
    output_dir = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\VideoMP4"
    delete_flv = False
    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    files_converted = 0
    Skipped_files = 0
    convert_files_limit = 1500
    
    for file in os.listdir(input_dir):
        if file.endswith(".flv"):
            input_file = os.path.join(input_dir, file)
            output_file = os.path.join(output_dir, file.replace(".flv", ".mp4"))
            # convert_flv_to_mp4(input_file, output_file, delete_flv=False)
            # Check if output file already exists
            if os.path.exists(output_file):
                # print(f"Output file {output_file} already exists. Skipping conversion.")
                Skipped_files += 1
                continue
            video = VideoFileClip(input_file)
            video.write_videofile(output_file, codec='libx264', logger=None)  # 'libx264' is typically used for MP4 files
            video.close()
            if delete_flv:
                os.remove(input_file)
            files_converted += 1
            # Show progress X out of X files converted
            if files_converted % 20 == 0:
                print(f"Progress: {files_converted}/{convert_files_limit}.")
            if files_converted >= convert_files_limit:
                break
    
    print(f"Converted {files_converted} files.")
    print(f"Skipped {Skipped_files} files.")

if __name__ == "__main__":
    main()




