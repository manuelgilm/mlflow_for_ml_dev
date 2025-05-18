from typing import Tuple
from typing import Optional
import keras
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema
from mlflow.types.schema import TensorSpec
import numpy as np


def get_model_signature(
    input_shape: Optional[Tuple[int, int]] = (28, 28, 1), n_classes: Optional[int] = 10
) -> ModelSignature:
    """
    Get the model signature for the digit recognition model.

    :param input_shape: The target shape of the image (default is (28, 28)).
    :param n_classes: The number of classes (default is 10).
    :return: Model signature.
    """

    input_specification = TensorSpec(
        type=np.dtype("float32"), shape=(-1, *input_shape), name="image_input"
    )
    output_specification = TensorSpec(
        type=np.dtype("float32"), shape=(-1, n_classes), name="output"
    )
    input_schema = Schema([input_specification])
    output_schema = Schema([output_specification])
    model_signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    return model_signature


def get_image_processor(
    image_shape: Tuple[int, int] = (28, 28, 1), n_classes: int = 10, **kwargs
) -> None:
    """
    Creates the CNN architecture for the digit recognition model.

    :param image_shape: The target shape of the image (default is (28, 28)).
    :return: Preprocessed image.
    """
    input_name = kwargs.get("input_name", "image_input")
    x_im_i = keras.Input(shape=image_shape, name=input_name)
    x_im = keras.layers.Rescaling(1.0 / 255, input_shape=image_shape)(x_im_i)
    x_im = keras.layers.Conv2D(32, (3, 3), activation="relu")(x_im)
    x_im = keras.layers.MaxPool2D((2, 2))(x_im)
    x_im = keras.layers.BatchNormalization()(x_im)

    x_im = keras.layers.Flatten()(x_im)
    x_im = keras.layers.Dense(16, activation="relu")(x_im)
    x_im = keras.layers.BatchNormalization()(x_im)
    x_im = keras.layers.Dense(n_classes, activation="softmax")(x_im)

    return x_im_i, x_im
