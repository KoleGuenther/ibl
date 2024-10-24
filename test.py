import joblib

# Path to model you want to load
model_file = 'models/101524Model'

# Loads saved model
model = joblib.load(model_file)

# This is to display the parameters to the model
print("Model Parameters:")
print(model.get_params())


"""LBFGS Solver Best Parameters"""
# solver = 'lbfgs',
# hidden_layer_sizes = (8, 4,),
# max_iter = 2000,
# alpha = 0.1,
# activation = 'relu',
# tol = 1e-5,
# random_state = 4,
# learning_rate_init = 0.001


"""Adam Solver Best Parameters"""
# solver = 'adam',
# hidden_layer_sizes = (15, 8,),
# max_iter = 4000,
# alpha = 0.5,
# activation = 'relu',
# tol = 1e-5,
# random_state = 4,
# learning_rate_init = 0.005