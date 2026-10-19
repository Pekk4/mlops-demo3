import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

with mlflow.start_run() as run:
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)
    mlflow.log_metric("accuracy", acc)

    model_info = mlflow.sklearn.log_model(model, "wine-model")
    model_id = model_info.model_id
    experiment_id = run.info.experiment_id
    model_uri = f"s3://mlflow/{experiment_id}/models/{model_id}/artifacts"

    with open("/tmp/model-uri.txt", "w") as f:
        f.write(model_uri)

print("Training complete.")
