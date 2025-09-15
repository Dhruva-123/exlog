import shap
from sklearn.base import is_classifier, is_regressor
from ..saver import log_saver

def xgboost_logger(model, X, y, path):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    if is_classifier(model):
        task = "classifier"
    if is_regressor(model):
        task = "regressor"
    return log_saver("xgboost", "TreeExplainer",model, shap_values, X, y, path,"tree" ,task)

