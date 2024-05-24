# Generates csv file with the columns: filename, emotion, label

import os
import pandas as pd
import numpy as np

# Define the path to the directory containing the images
directory = r"C:\MyDocs\DTU\MSc\Thesis\Data\CREMA-D\CREMA-D\TEST\HI"

# Get the list of files in the directory
files = []
for file in os.listdir(directory):
    if file.endswith(".mp4"):
        files.append(file)

# Create a DataFrame
df = pd.DataFrame(columns=['filename', 'emotion'])
# Fill dataframe with the filenames
df['filename'] = files
emotion_dict = {"NEU": "Neutral", "HAP": "Happiness", "SAD": "Sadness", "SUR": "Surprise", "FEA": "Fear", "DIS": "Disgust", "ANG": "Anger"}
# Fill dataframe with the emotions
df['emotion'] = [emotion_dict[file.split("_")[2]] for file in files]

# View the first 20 rows
print(df.head(20))

# Save the DataFrame to a csv file
df.to_csv('video_labels.csv', index=False)