import pickle

# Open the file in 'rb' (read binary) mode
with open('wise_hi_absorption_8', 'rb') as f:
    data = pickle.load(f)

# Display the contents
print(data)