import os
import numpy as np
import pandas as pd

list_of_files = [f"file_{i}.csv" for i in range(101)][1:]
# print(list_of_files)

# fun_test = range(10)
# print type of fun_test
# print(type(fun_test))
# Convert range to np array
# fun_test = np.array(fun_test)

# checkpoints = list(100*np.array(range(11))[1:])
# print(checkpoints)

# Create dataframe with columns "filename", "checkpoint"
df = pd.DataFrame(columns=["filename", "result", "checkpoint"])

# test_df = pd.DataFrame(columns=["filename", "result", "checkpoint"])

progress = 1

continue_from_checkpoint = True
# if continue from checkpoint, then load the last checkpoint
# Create a list with all files in current directory starting with "TESTING_predicted_checkpoint_"

if continue_from_checkpoint:
    list_of_checkpoint_files = [f for f in os.listdir() if f.startswith("TESTING_predicted_checkpoint_")]
    # print(list_of_files)
    # Get the last checkpoint
    last_checkpoint = max([int(f.split("_")[-1].split(".")[0]) for f in list_of_checkpoint_files])
    print(last_checkpoint)
    # Load the last checkpoint
    # df = pd.read_csv(f'TESTING_predicted_checkpoint_{last_checkpoint}.csv')

    list_of_files = list_of_files[last_checkpoint:]

    # Inspect the dataframe
    # print(df.head(10))
    # Get the last checkpoint
    progress = last_checkpoint + 1
    # print(progress)
    # Get the list of files that have already been processed
    # list_of_files = df["filename"].tolist()
    # print(list_of_files)
    # Get the list of files that have not been processed
    # list_of_files = [f for f in list_of_files if f not in list_of_files]
    # print(list_of_files)


df['filename'] = list_of_files
checkpoint_row = 1
for file in list_of_files:
    result = np.random.randint(0, 100)*np.random.randn(1)
    # print("Result: ", result)
    df.loc[df['filename'] == file, 'result'] = result
    df.loc[df['filename'] == file, 'checkpoint'] = progress
    if progress % 10 == 0:
        # save the DataFrame to a csv file
        checkpoint_df = df.head(checkpoint_row)
        checkpoint_df.to_csv(f'TESTING_predicted_checkpoint_{progress}.csv', index=False)
    progress += 1
    checkpoint_row += 1
    # if progress == 55:
        # break
    # Inspect the dataframe

# df["filename"] = list_of_files
# df["checkpoint"] = checkpoints
# 
print(df.head(10))

# checkpoint_list = df["checkpoint"].tolist()

# # Check type of checkpoint_list
# print(type(checkpoint_list))

# # Get the max value of checkpoint_list
# max_checkpoint = max(checkpoint_list)
# print(max_checkpoint)