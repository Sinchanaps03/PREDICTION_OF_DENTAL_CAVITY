import pickle

with open('class_names.pkl', 'rb') as f:
    class_names = pickle.load(f)

print("Class names:", class_names)
