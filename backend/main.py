import logging
import os
from urllib.parse import urlparse

import mlflow
import numpy
from fastapi import BackgroundTasks, FastAPI
from mlflow.tracking import MlflowClient

from backend.models import DeleteApiData, PredictApiData, TrainApiData
from ml.data import load_mnist_data
from ml.models import CNN, LinearModel
from ml.train import Trainer
from ml.utils import set_device

mlflow.set_tracking_uri("sqlite:///db/backend.db")
mlflowclient = MlflowClient(mlflow.get_tracking_uri(), mlflow.get_registry_uri())
app = FastAPI()
logger = logging.getLogger("uvicorn")


def train_model_task(model_name: str, hyperparams: dict, epochs: int):
    """Tasks that trains the model. This is supposed to be running in the background
    Since it's a heavy computation it's better to use a stronger task runner like Celery
    For the simplicity I kept it as a fastapi background task"""

    # Setup env
    device = set_device()
    mlflow.set_experiment("MNIST")

    with mlflow.start_run() as run:
        mlflow.set_tag("mlflow.source.name", os.uname().nodename)
        mlflow.log_params(hyperparams | {"epochs": epochs})

        # Prepare for training
        logger.info("Loading data...")
        train_dataloader, test_dataloader = load_mnist_data()

        if hyperparams["model_type"] == "linear":
            model = LinearModel(hyperparams).to(device)
        elif hyperparams["model_type"] == "cnn":
            model = CNN(hyperparams).to(device)

        # Train
        logger.info("Training model")
        trainer = Trainer(model, device=device)  # Default configs
        trainer.train(epochs, train_dataloader, test_dataloader, mlflow)

        # Register model
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info(f"{tracking_url_type_store=}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            mlflow.pytorch.log_model(
                model,
                "LinearModel",
                registered_model_name=model_name,
                conda_env=mlflow.pytorch.get_default_conda_env(),
            )
        else:
            mlflow.pytorch.log_model(model, "LinearModel-MNIST", registered_model_name=model_name)

        # Take last model version
        mv = mlflowclient.search_model_versions(f"name='{model_name}'")[-1]
        mlflowclient.transition_model_version_stage(
            name=mv.name, version=mv.version, stage="production"
        )


@app.get("/")
async def read_root():
    return {
        "Tracking URI": mlflow.get_tracking_uri(),
        "Registry URI": mlflow.get_registry_uri(),
    }


@app.get("/models")
async def get_models_api():
    """Gets a list with model names"""
    model_list = mlflowclient.list_registered_models()
    names = [model.name for model in model_list]
    return [
        {
            "name": name,
            "version": mlflowclient.get_registered_model(name).latest_versions[-1].version,
        }
        for name in names
    ]


@app.post("/train")
async def train_api(data: TrainApiData, background_tasks: BackgroundTasks):
    """Creates a model based on hyperparameters and trains it."""
    hyperparams = data.hyperparams
    epochs = data.epochs
    model_name = data.model_name

    background_tasks.add_task(train_model_task, model_name, hyperparams, epochs)

    return {"result": "Training task started"}


@app.post("/predict")
async def predict_api(data: PredictApiData):
    """Predicts on the provided image"""
    img = data.input_image
    model = mlflow.pyfunc.load_model(model_uri=f"models:/{data.model_name}/{data.model_version}")
    # Preprocess the image
    # Flatten input, create a batch of one and normalize
    img = numpy.array(img, dtype=numpy.float32).flatten()[numpy.newaxis, ...] / 255
    # Postprocess result
    pred = model.predict(img)
    logger.info(pred)
    res = int(numpy.argmax(pred[0]))
    return {"result": res, "prob": pred[0].tolist()}


@app.post("/delete")
async def delete_model_api(data: DeleteApiData):
    model_name = data.model_name
    version = data.model_version

    if version is None:
        # Delete all versions
        mlflowclient.delete_registered_model(name=model_name)
        response = {"result": f"Deleted all versions of model {model_name}"}
    elif isinstance(version, list):
        for v in version:
            mlflowclient.delete_model_version(name=model_name, version=v)
        response = {"result": f"Deleted versions {version} of model {model_name}"}
    else:
        mlflowclient.delete_model_version(name=model_name, version=version)
        response = {"result": f"Deleted version {version} of model {model_name}"}
    return response
