from io import BytesIO
import logging
import argparse
from utils import predict_captcha
from config import BaseConfig
from tensorflow import keras
from model import ModelLoader

base_config = BaseConfig.get_config()

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def predict(img_obj: BytesIO):
    # load prediction mdoel
    # get model
    model = ModelLoader.get_model()

    prediction_model = keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )
    # prediction_model.summary()

    return predict_captcha(img_obj, prediction_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="predict utils")
    parser.add_argument("Path", type=str, help="image file path")

    args = parser.parse_args()

    with open(args.Path, "rb") as f:
        img_obj = BytesIO(f.read())

    captcha_result = predict(img_obj)
    print(captcha_result)


if __name__ == "__main__":
    main()
