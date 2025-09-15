import json
from pathlib import Path
import numpy as np
##json for saving, pathlib for paths, numpy to convert numpy items (be it lists or numbers) to normal items.
import warnings
##We will use warnings in order to warn the user that the desired import of either torch or tensorflow didn't work
def log_saver(framework, explainer, model, shap_values, X , y, path, family = None, task = None, import_error = False):
        logs = []       
        if import_error: ## If we get a failure of import in any of our x_logger files, we will save a warning json file that import failed. The user will then fix the import issue.
            warning = {
                "type": "dependency_warning",
                "message": "A framework that you are trying to use isn't installed",
                "dependencies": framework
            }
            logs.append(warning)
        else:### We have no import issues.

            flag = False
            if framework == "torch": ### This block is to check if torch is importable or not. If it is not, we need to save a failure message in json format. Therefore, we are using the flag variable.
                try:
                    import torch
                except ImportError:
                    warnings.warn(f"{framework} not found... install {framework} and try again")
                    flag = True

            
            for i in range(len(shap_values.values)):
                if isinstance(X, np.ndarray): ### We are checking if the given rows of the features are numpy related. If they are, we can just access them with X[i]
                    row = X[i]
                    if framework == "torch":
                        if flag:
                            break
                        prediction = model(torch.tensor(row.reshape(1, - 1), dtype = torch.float32))[0]
                    else:
                        prediction = model.predict(row.reshape(1, -1))[0]
                else: ##The exact same situation as above, but currently, we are dealing with pandas dataframes instead of numpy arrays. These have to be dealt with in a different way.
                    row = X.iloc[i]
                    if framework == "torch":
                        if flag:
                            break
                        prediction = model(torch.tensor([row.values], dtype = torch.float32))[0]
                    else:
                        prediction = model.predict([row.values])[0]
                ### Now that we successfully have predictions in our hands, we are now going to deal with the type of predictions. 
                ### Some predictions are returned in numpy terms and some are returned in tensors. We are converting both types into standard python types in order to be able to save them in a json file. Otherwise, json throws errors at you that the given type is not supported. 
                if isinstance(prediction, np.generic):
                    prediction = prediction.item()
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.tolist()
                if isinstance(prediction, torch.Tensor) and prediction.dim() == 0:
                    prediction = prediction.item()
                if isinstance(prediction, torch.Tensor):
                    prediction = prediction.tolist()
                
                ### This is where we create dictionaries to send into log array which then will be converted into a json file.
                record = {
                    "framework": framework,
                    "family" : family,
                    "task" : task,
                    "Explainer": explainer,
                    "Input": row.to_dict() if hasattr(row, "to_dict") else row.tolist(),
                    "Prediction": prediction,
                    "Explanation": shap_values.values[i].tolist()
                }

                ### edge case management.
                if framework == "unknown":
                    record["Message"] = "Framework not recognized"
                if y is not None:
                    if task in ("classification", "classifier"):
                        record["Prediction State"] = bool(prediction == y[i])
                    else:
                        record["Prediction State"] = None 

                # We are now appending each record to logs.
                logs.append(record)

        ### This if statement saves the log_saver from breaking if there is an import error.
        if flag:
            logs = [{
                "Error": "Couldn't import torch"
            }]
        return json_file_saver(logs, path)

    ### We are converting the logs array into a json file and saving it in the given path. We are also returning the logs for further analysis by the user.
def json_file_saver(logs, path):
    Path(path).write_text(json.dumps(logs ,indent = 2))
    print(f"Logs have been saved to {path}.")
    print("Contributions for the development of exlog are welcome... Let's make explainability easier.")
    return logs