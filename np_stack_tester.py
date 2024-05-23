# This script tests the np_stack module
import numpy as np

# Create a list of 10 elements, each element is a 224x224x3 numpy array
face_areas = [np.random.rand(224, 224, 3) for i in range(10)]
# Stack the 10 elements into a single numpy array
face_areas_stacked = np.stack(face_areas)
# Print the shape of the stacked numpy array
print("Shape of face_areas_stacked: ", face_areas_stacked.shape)
# Print the type of the stacked numpy array
print("Type of face_areas_stacked: ", type(face_areas_stacked))
# Print the first element of the stacked numpy array
print("First element of face_areas_stacked: ", face_areas_stacked[0])
# Print the shape of the first element of the stacked numpy array
print("Shape of first element of face_areas_stacked: ", face_areas_stacked[0].shape)
