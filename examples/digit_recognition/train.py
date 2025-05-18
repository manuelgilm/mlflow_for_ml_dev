from examples.utils.decorators import mlflow_tracking_uri
from examples.utils.decorators import mlflow_client
from examples.utils.decorators import mlflow_experiment
from examples.digit_recognition.utils import get_model_signature
from examples.digit_recognition.utils import get_image_processor
from examples.digit_recognition.data import get_train_test_data
from examples.digit_recognition.data import transform_to_image
import os
import keras
import mlflow


@mlflow_tracking_uri
@mlflow_experiment(name="digit_recognition")
@mlflow_client
def main(**kwargs) -> None:
    """ """
    os.environ["KERAS_BACKEND"] = "torch"
    x_train, x_test, y_train, y_test = get_train_test_data()
    x_train = transform_to_image(x_train)
    x_test = transform_to_image(x_test)

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
            validation_data=({input_name: x_test}, y_test),
            batch_size=32,
            epochs=1,
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

        # set model version alias to "production"
        model_version = mlflow.search_model_versions(
            filter_string=f"name='{registered_model_name}'", max_results=1
        )[0]
        client = kwargs["mlflow_client"]
        client.set_registered_model_alias(
            name=registered_model_name,
            version=model_version.version,
            alias="production",
        )
