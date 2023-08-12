import mlflow
import mlflow.tensorflow
from main import model

with mlflow.start_run() as run:

    mlflow.log_param("epochs", 5)
    mlflow.log_param("optimizer", "adam")


    mlflow.tensorflow.autolog()
    model.fit(X_train, y_train, epochs=5)


    loss, accuracy = model.evaluate(X_test, y_test, verbose=2)


    mlflow.log_metric("loss", loss)
    mlflow.log_metric("accuracy", accuracy)


    mlflow.tensorflow.log_model(model, "mnist_model")
