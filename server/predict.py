import warnings
warnings.filterwarnings('ignore')

import pickle, joblib, os
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class prediction():
    def __init__(self, input) -> None:
        self.path_models = 'models'
        self.text = input

    def load_models(self, level):
        filename_model = 'model_fallback.h5'
        filename_token = 'tokenizer_fallback.pickle'
        filename_encoder = 'encoder_fallback.joblib'

        if level != '':
            filename_model = 'model_fallback_' + level + '.h5'
            filename_token = 'tokenizer_fallback_' + level + '.pickle'
            filename_encoder = 'encoder_fallback_' + level + '.joblib'

        model = tf.keras.models.load_model(os.path.join(self.path_models, filename_model))

        with open(os.path.join(self.path_models, filename_token), 'rb') as file:
            tokenizer = pickle.load(file)

        encoder = joblib.load(os.path.join(self.path_models, filename_encoder))

        return [model, tokenizer, encoder]

    def get_prediction_label(self, model, tokenizer, encoder, text):
        max_sequence_length = 100
        text_sequence = tokenizer.texts_to_sequences([text])
        text_data = pad_sequences(text_sequence, maxlen = max_sequence_length)
        prediction = model.predict(text_data, verbose = 0)
        prediction_label = encoder.inverse_transform(np.argmax(prediction, axis=1))
        return prediction_label[0]
    
    def run(self):
        text = self.text

        # load the model
        model, tokenizer, encoder = self.load_models('')

        fallback_prediction = self.get_prediction_label(model, tokenizer, encoder, text)

        # load subcategory model
        sub_model, sub_tokenizer, sub_encoder = self.load_models(fallback_prediction)

        subcategory_prediction = self.get_prediction_label(sub_model, sub_tokenizer, sub_encoder, text)

        response = {}
        response['category'] = fallback_prediction
        response['subcategory'] = subcategory_prediction
        
        return response