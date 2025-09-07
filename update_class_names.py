import pickle

class_names = ['no cavity', 'mild cavity', 'moderate cavity', 'severe cavity']

with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)

print("Updated class_names.pkl with classes:", class_names)
