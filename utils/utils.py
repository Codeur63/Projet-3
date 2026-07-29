import json

import joblib


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)


def save_metrics(metrics, path):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
