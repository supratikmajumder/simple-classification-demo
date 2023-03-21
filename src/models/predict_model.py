import os
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import recall_score, precision_score
import json

filename = "mlops_demo_model.sav"
X_test = np.genfromtxt("data/processed/test_features.csv")
Y_test = np.genfromtxt("data/processed/test_labels.csv")

# Model loading for prediction
model = pickle.load(open(os.path.join("models", filename), 'rb'))

Y_pred = model.predict(X_test)
acc = model.score(X_test, Y_test)

# Actual value (Y_test) vs. Predicted value (Y_pred)
prec = precision_score(Y_test, Y_pred)
rec = recall_score(Y_test, Y_pred)

# Get the loss
loss = model.loss_curve_
pd.DataFrame(loss, columns=['loss']).to_csv("reports/loss.csv", index=False)

with open("reports/metrics.json", "w") as outfile:
    json.dump({"accuracy": acc, "precision": prec, "recall": rec}, outfile)

print("Predictions completed with saved model")