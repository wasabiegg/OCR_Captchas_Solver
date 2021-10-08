import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import BaseConfig
import numpy as np


base_config = BaseConfig.get_config()

char_to_num = layers.StringLookup(
    vocabulary=list(base_config.characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [base_config.image_height, base_config.image_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": img, "label": label}


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]

    max_length = base_config.max_length

    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def encode_single_image(img_obj):
    #     img = keras.preprocessing.image.img_to_array(img_obj, dtype=np.string_)
    #     img = tf.convert_to_tensor(img, tf.string)
    #     print(img)

    img = tf.constant(img_obj.read())
    img = tf.io.decode_png(img, channels=1)
    # img = tf.expand_dims(img, axis=2)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [base_config.image_height, base_config.image_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    # label = char_to_num(tf.strings.unicode_split(label, input_encoding='UTF-8'))
    return img


def predict_captcha(img_obj, prediction_model) -> str:
    img = encode_single_image(img_obj)
    inputs = np.expand_dims(img, axis=0)
    preds = prediction_model.predict(inputs)
    pred_texts = decode_batch_predictions(preds)
    return pred_texts[0]
