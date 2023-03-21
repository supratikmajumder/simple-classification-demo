from sklearn.neural_network import MLPClassifier

import os
import numpy as np
import pickle

features_train = np.genfromtxt("data/processed/train_features.csv")
label_train    = np.genfromtxt("data/processed/train_labels.csv")

max_iteration = 10

model = MLPClassifier(random_state=17, max_iter=max_iteration)
model.fit(features_train, label_train)

# Manual process
filename = "mlops_demo_model.sav"
pickle.dump(model, open(os.path.join("models", filename), "wb"))
print("Model saved successfully in local drive")

# Remote process
#   Can be used to store the model in a remote location -- AWS datastore, Google cloud, Azure, or anywhere else
