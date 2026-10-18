import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "age",
    "time_on_site_min",
    "pages_viewed",
    "cart_items",
    "previous_purchases",
    "discount_offered",
]

# ---- 1. Create dataset ----
rng = np.random.default_rng(42)
n = 2000

age = rng.integers(18, 65, n)
time_on_site_min = rng.gamma(2.0, 4.0, n).round(1)
pages_viewed = rng.integers(1, 40, n)
cart_items = rng.integers(0, 8, n)
previous_purchases = rng.integers(0, 15, n)
discount_offered = rng.integers(0, 2, n)

logit = (
    -4.5
    + 0.08 * time_on_site_min
    + 0.04 * pages_viewed
    + 0.6 * cart_items
    + 0.12 * previous_purchases
    + 0.9 * discount_offered
)
prob = 1 / (1 + np.exp(-logit))
purchase = (rng.random(n) < prob).astype(int)

df = pd.DataFrame(
    {
        "age": age,
        "time_on_site_min": time_on_site_min,
        "pages_viewed": pages_viewed,
        "cart_items": cart_items,
        "previous_purchases": previous_purchases,
        "discount_offered": discount_offered,
        "purchase": purchase,
    }
)

os.makedirs("data", exist_ok=True)
df.to_csv("data/shoppers.csv", index=False)

# ---- 2. Train ----
X, y = df[FEATURES], df["purchase"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline(
    [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]
)
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
acc = accuracy_score(y_test, pred)
print("Accuracy:", acc)
print(classification_report(y_test, pred, target_names=["No Purchase", "Purchase"]))

joblib.dump({"model": pipe, "features": FEATURES}, "model.pkl")
json.dump({"accuracy": acc}, open("metrics.json", "w"))
