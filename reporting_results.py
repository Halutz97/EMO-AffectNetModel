# Description: This script generates an overview of all results to be reported.

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy import stats

def argmax_str_to_array(argmax_str):
    argmax_str = argmax_str[1:-1]
    argmax_str = argmax_str.split()
    argmax = [int(i) for i in argmax_str]
    return argmax

# Load CSV file(s) with true and predicted labels
data1 = pd.read_csv(r'save_results\run_5_predicted_checkpoint_1150.csv')
data2 = pd.read_csv(r'save_results\run_5_predicted_checkpoint_2150.csv')
data3 = pd.read_csv(r'save_results\run_5_predicted_checkpoint_6150.csv')
data4 = pd.read_csv(r'save_results\run_5_predicted_checkpoint_7442.csv')

# Concatenate vertically
data = pd.concat([data1, data2, data3, data4], axis=0)

# get the shape of the dataframe
print(data.shape)

# remove the rows with missing values
# First check if there are any missing values
print(data.isnull().sum())
# Print the rows with missing values
print(data[data.isnull().any(axis=1)])

# Drop the rows with missing values
data = data.dropna()
# Print the new shape of the dataframe
print(data.shape)

label_model = ['Neutral', 'Happiness', 'Sadness', 'Surprise', 'Fear', 'Disgust', 'Anger']

encoding_dict_dataset = {'Neutral': 0, 'Happiness': 1, 'Sadness': 2, 'Surprise': 3, 'Fear': 4, 'Disgust': 5, 'Anger': 6}

# my_encoding_dict_model = {'neu': 0, 'ang': 1, 'hap': 2, 'sad': 3}
label_names = label_model
true_labels = data['emotion']
# label_keys = true_labels.map(encoding_dict_dataset).values

label_keys = [encoding_dict_dataset[label] for label in true_labels]

# Assert that label_keys and label_keys2 are the same
# are_equal = np.all(np.equal(label_keys, label_keys2))
# print("Arrays are equal:", are_equal)




# print("Label keys: ", label_keys)

predicted_classes= data['predicted']
# classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")

# for i, file in enumerate(files):

#     out_prob, score, index, text_lab = classifier.classify_file(os.path.join(directory,file))
#     predicted_classes.append(text_lab[0])

# # print("out prob:", out_prob)
# # print("score:", score)
# # print("index:", index)
# # print("text lab:", text_lab)
# # print("type of text lab:", type(text_lab))

predicted_keys = [encoding_dict_dataset[label] for label in predicted_classes]

# # Print the predicted classes and the actual labels
# for i, prediction in enumerate(predicted_classes):
#     print("Predicted:", prediction, "Actual:", true_labels[i])


# print("predicted_keys: ")
# print(predicted_keys)

# print("label_keys: ")
# print(label_keys)

print(predicted_keys == label_keys)

print("type of predicted_keys: ", type(predicted_keys))
print("type of label_keys: ", type(label_keys))

# Convert predicted_keys and label_keys to numpy arrays
predicted_keys = np.array(predicted_keys)
label_keys = np.array(label_keys)


# # Calculate accuracy
accuracy = (predicted_keys == label_keys).mean()
print("Accuracy:", accuracy)

# # Calculate F1 Scores
f1_micro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='micro')
print("F1 Score (Micro):", f1_micro)

f1_macro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='macro')
print("F1 Score (Macro):", f1_macro)

f1_weighted = sklearn.metrics.f1_score(label_keys, predicted_keys, average='weighted')
print("F1 Score (Weighted):", f1_weighted)

# # Generate confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(label_keys, predicted_keys)

confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

# # Fill the confusion matrix with the values from the actual confusion matrix
for i, label in enumerate(true_labels):
    confusion_matrix_full[label_keys[i], predicted_keys[i]] +=1

# # Create a DataFrame for the confusion matrix
confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# # Add a row and column for the total counts
confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

print("Confusion Matrix:")
print(confusion_matrix_df)

# # Calculate the maximum value for the heatmap color scale
max_value = confusion_matrix_df.iloc[:-1,:].values.max()

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix_df.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, vmin=0, vmax=max_value)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Continue executing the script while the plot is displayed



# # Read the 'argmax' column from the first row
# argmax = data.iloc[104]['argmax']
# # print(argmax)
# argmax = argmax_str_to_array(argmax)
# print(argmax)
# print(type(argmax[0]))

# # Find the unique numbers in argmax
# unique_numbers = np.unique(argmax)
# print(unique_numbers)

# data_corrected = data.copy()
# # add column 'prediction_corrected' to data_corrected
# data_corrected['prediction_corrected'] = np.nan
# # itreate over the rows of data_corrected
# for i, row in data_corrected.iterrows():
#     # get the argmax value for the current row
#     argmax = row['argmax']
#     argmax = argmax_str_to_array(argmax)
#     unique_numbers = np.unique(argmax)
#     # if the length of unique_numbers is greater than 1
#     if len(unique_numbers) == 2:
#         # Get the max value
#         max_value = np.max(argmax)
#         # Append this value to the end of argmax
#         argmax = np.append(argmax, max_value)
#         mode = max_value
#         # mode_result = stats.mode(argmax)
#         # if mode_result.mode.shape == ():
#             # Scalar scenario: mode is a single value scalar
#             # mode = mode_result.mode
#         # else:
#             # Array scenario: mode is an array
#             # mode = mode_result.mode[0]
#         corrected_prediction = label_model[mode]
#         # get the index of the maximum value in argmax
#         # index_max = np.argmax(argmax)
#         # get the corresponding label
#         # label_max = label_model[index_max]
#         # assign the label to the 'prediction_corrected' column
#         data_corrected.at[i, 'prediction_corrected'] = corrected_prediction
#     else:
#         # assign the value in argmax to the 'prediction_corrected' column
#         data_corrected.at[i, 'prediction_corrected'] = data_corrected.at[i, 'predicted']


# predicted_classes= data_corrected['prediction_corrected']
# predicted_keys = [encoding_dict_dataset[label] for label in predicted_classes]
# predicted_keys = np.array(predicted_keys)
# label_keys = np.array(label_keys)

# # # Calculate accuracy
# accuracy = (predicted_keys == label_keys).mean()
# print("Accuracy:", accuracy)

# # # Calculate F1 Scores
# f1_micro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='micro')
# print("F1 Score (Micro):", f1_micro)

# f1_macro = sklearn.metrics.f1_score(label_keys, predicted_keys, average='macro')
# print("F1 Score (Macro):", f1_macro)

# f1_weighted = sklearn.metrics.f1_score(label_keys, predicted_keys, average='weighted')
# print("F1 Score (Weighted):", f1_weighted)

# # # Generate confusion matrix
# confusion_matrix = sklearn.metrics.confusion_matrix(label_keys, predicted_keys)

# confusion_matrix_full = np.zeros((len(label_names), len(label_names)), dtype=int)

# # # Fill the confusion matrix with the values from the actual confusion matrix
# for i, label in enumerate(true_labels):
#     confusion_matrix_full[label_keys[i], predicted_keys[i]] +=1

# # # Create a DataFrame for the confusion matrix
# confusion_matrix_df = pd.DataFrame(confusion_matrix_full, index=label_names, columns=label_names)

# # # Add a row and column for the total counts
# confusion_matrix_df['Total'] = confusion_matrix_df.sum(axis=1)
# confusion_matrix_df.loc['Total'] = confusion_matrix_df.sum()

# print("Confusion Matrix:")
# print(confusion_matrix_df)

# # # Calculate the maximum value for the heatmap color scale
# max_value = confusion_matrix_df.iloc[:-1,:].values.max()

# plt.figure(figsize=(5, 4))
# sns.heatmap(confusion_matrix_df.iloc[:-1, :-1], annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, vmin=0, vmax=max_value)
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

## Investigate if we only take files ending with "HI.mp4"
# Read the 'argmax' column from the first row
file = data.iloc[104]['filename']
print(file)

# New empty dataframe with the same columns
data_filtered = pd.DataFrame(columns=data.columns)
# Iterate through data, if a condition is met, append the row to data_filtered
print(pd.__version__)
for i, row in data.iterrows():
    if row['filename'].endswith('HI.mp4'):
        data_filtered = data_filtered.append(row, ignore_index=True)

print(data_filtered.shape)
print(data_filtered.head(10))
# print(argmax)
# argmax = argmax_str_to_array(argmax)
# print(argmax)
# print(type(argmax[0]))

# # Find the unique numbers in argmax
# unique_numbers = np.unique(argmax)
# print(unique_numbers)