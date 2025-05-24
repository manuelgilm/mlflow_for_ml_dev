import os

os.environ["KERAS_BACKEND"] = "torch"


from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment
from examples.digit_recognition.utils import get_model_signature
from examples.digit_recognition.utils import get_image_processor
from examples.digit_recognition.data import get_train_val_test_data
from examples.digit_recognition.data import transform_to_image
from examples.utils.mlflow_utils import set_alias_to_latest_version
import os
import keras
import mlflow


@mlflow_tracking_uri
@mlflow_experiment(name="digit_recognition")
@mlflow_client
def main(**kwargs) -> None:
    """
    Train a model for handwritten digit recognition using the MNIST dataset.
    This function prepares the data, builds a Keras model, trains it, and logs the model to MLflow.

    To deploy the model in the local server, run the following command:
    """

    x_train, x_val, _, y_train, y_val, _ = get_train_val_test_data()
    x_train = transform_to_image(x_train)
    x_val = transform_to_image(x_val)

    # building the model
    input_name = "image_input"
    x_im_i, x_im = get_image_processor(input_name=input_name)
    model = keras.Model(inputs=(x_im_i,), outputs=(x_im,))

    optimizer = keras.optimizers.Adamax()
    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    callbacks = [mlflow.keras.MlflowCallback()]
    model_signature = get_model_signature()
    with mlflow.start_run() as run:
        model.fit(
            x={input_name: x_train},
            y=y_train,
            validation_data=({input_name: x_val}, y_val),
            batch_size=32,
            epochs=5,
            validation_split=0.2,
            callbacks=callbacks,
        )

        # log model
        registered_model_name = "Digit_Recognition_Model"
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            signature=model_signature,
            registered_model_name=registered_model_name,
        )
        set_alias_to_latest_version(
            registered_model_name=registered_model_name,
            alias="production",
            client=kwargs["mlflow_client"],
        )
