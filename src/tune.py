import optuna
import warnings
import mlflow
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from config import RANDOM_STATE, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")


def tune_xgboost(X_train, y_train, n_trials=30):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "eval_metric":      "logloss",
            "verbosity":        0,
            "random_state":     RANDOM_STATE,
        }
        model = XGBClassifier(**params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_auc    = study.best_value

    with mlflow.start_run(run_name="xgboost_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_auc)

    print(f"[tune] XGBoost best AUC: {best_auc:.4f}")
    print(f"[tune] Best params: {best_params}")
    return best_params


def tune_lightgbm(X_train, y_train, n_trials=30):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    def objective(trial):
        params = {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 500),
            "max_depth":     trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "random_state":  RANDOM_STATE,
            "verbose":       -1,
        }
        model = LGBMClassifier(**params)
        return cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_auc    = study.best_value

    with mlflow.start_run(run_name="lightgbm_tuned"):
        mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_roc_auc", best_auc)

    print(f"[tune] LightGBM best AUC: {best_auc:.4f}")
    print(f"[tune] Best params: {best_params}")
    return best_params


def get_tuned_model(X_train, y_train, model_name="xgboost"):
    if model_name == "xgboost":
        best_params = tune_xgboost(X_train, y_train)
        model = XGBClassifier(**best_params, eval_metric="logloss",
                              verbosity=0, random_state=RANDOM_STATE)
    elif model_name == "lightgbm":
        best_params = tune_lightgbm(X_train, y_train)
        model = LGBMClassifier(**best_params, random_state=RANDOM_STATE, verbose=-1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fit(X_train, y_train)
    print(f"[tune] Tuned {model_name} fitted.")
    return model
