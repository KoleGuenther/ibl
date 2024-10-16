import joblib

# Path to model you want to load
model_file = 'models/101524Model'

# Loads saved model
model = joblib.load(model_file)

# This is to display the parameters to the model
print("Model Parameters:")
print(model.get_params())
