EXLOG

A single line to log, save, and explain any ML model’s predictions with SHAP. Stop wasting time writing broiler code, focus on creativity.
Why exlog?

Without exlog:

```import shap, json, numpy as np
from sklearn.ensemble import RandomForestClassifier```

# This is your model training phase
```model = RandomForestClassifier().fit(X_train, y_train)```

# Here, the user has to carefully choose a proper explainer and this 	requires a fair bit of knowledge on SHAP.

```explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

logs = []
for i in range(len(X_test)):
row = X_test[i]
pred = model.predict(row.reshape(1, -1))[0]
if isinstance(pred, np.generic):
pred = pred.item()
record = {
"framework": "sklearn",
"Explainer": "TreeExplainer",
"Input": row.tolist(),
"Prediction": pred,
"Explanation": shap_values.values[i].tolist(),
"Prediction State": bool(pred == y_test[i]),
}
logs.append(record)```

#This is your json saver
```with open("logs.json", "w") as f:
json.dump(logs, f, indent=2)```

With exlog:

```from exlog import log
model = RandomForestClassifier().fit(X_train, y_train)
log(model, X_test, y_test, path="logs.json")```

This results in a reduction of both time and errors during logging which is useful for both novice and experienced ML devs.

Example Output:
```[
  {
    "framework": "sklearn",
    "Explainer": "TreeExplainer",
    "Input": [12.47, 18.6, 81.09, 481.9, ...],
    "Prediction": 1,
    "Explanation": [0.01, -0.03, 0.07, ...],
    "Prediction State": true
  }
]```

Here is how to install it:

Open command prompt(windows) or powershell(windows) or terminal(mac) and type this out:
	pip install mlexlog


Supported Frameworks

. scikit-learn (100%)

. XGBoost (100%)

. LightGBM (100%)

. PyTorch (still in development)

. TensorFlow/Keras (still in development)

Contributing

Want to make ML explainability truly universal? Add support for your favorite framework, improve performance, or suggest better log formats. Contributions are welcome!





