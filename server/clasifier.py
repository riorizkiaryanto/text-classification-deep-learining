import warnings
warnings.filterwarnings('ignore')

import os, pickle, joblib
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class model_train():
    def __init__(self) -> None:
        self.path_sources = 'sources/raw_data' # dictionary to store our local data train files
        self.path_models = 'models' # dictionary to store our trained models
        self.column_message = 'Message'
        self.column_fallback = 'Fallback Intent'
        self.column_fallback_subcategory = 'Category'
    
    def apply_mapping(self, item, mapping_dict):
        cleanedItem = mapping_dict.get(item.strip().lower(), item)
        return cleanedItem
    
    def clean_message(self, message):
        return str(message).lower().strip()

    def load_data(self):
        df = pd.DataFrame()

        for filename in os.listdir(os.path.join(self.path_sources)):
            if filename.endswith('.xlsx'):
                temp = pd.read_excel(os.path.join(self.path_sources, filename), sheet_name = 'Fallback')
                df = pd.concat([df, temp], ignore_index=True)
        return df
    
    def train(self, text, label):
        max_words = 1000 # maximum number of words to consider as features
        max_sequence_length = 100 # maximum length of each sentence

        tokenizer = Tokenizer(num_words = max_words)
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)
        word_index = tokenizer.word_index

        data = pad_sequences(sequences, maxlen = max_sequence_length) # pad sequence to have the same length

        # convert label to categorical format
        label_encoder = LabelEncoder()
        label_encoder.fit(label)
        encoded_labels = label_encoder.transform(label)
        one_hot_labels = to_categorical(encoded_labels)

        # split dataset for training and validation tests
        X_train, X_val, y_train, y_val = train_test_split(data, one_hot_labels, test_size = 0.2, random_state=42)

        # Model definition
        embedding_dim = 100 # dimensionality of word embeddings
        num_filters = 128 # number of filters in the convolutional layer
        filter_sizes = [3, 4, 5] # sized of filters in the convolutional layer

        model = Sequential()
        model.add(Embedding(max_words, embedding_dim, input_length = max_sequence_length))
        model.add(Conv1D(num_filters, 7, activation = 'relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(64, activation = 'relu'))
        model.add(Dense(len(label_encoder.classes_), activation = 'softmax'))

        # Compile and train the model
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        model.fit(X_train, y_train, epochs = 10, batch_size = 16, validation_data = [X_val, y_val])

        return [model, tokenizer, label_encoder]
    
    def run(self):
        df = self.load_data()
        
        texts = df[self.column_message].astype(str).tolist()
        labels = df[self.column_fallback].astype(str).tolist()

        # Train and save the fallback clasifier model
        results = self.train(texts, labels)

        model, token, encoder = results

        model.save(os.path.join(self.path_models, 'model_fallback.h5'))

        with open(os.path.join(self.path_models, 'tokenizer_fallback.pickle'), 'wb') as file:
            pickle.dump(token, file)

        joblib.dump(encoder, os.path.join(self.path_models, 'encoder_fallback.joblib'))

        # Train and save the fallback second stage models
        for intent in list(set(labels)):
            df_category = df[df[self.column_fallback] == intent].copy()
            category_texts = df_category[self.column_message].astype(str).tolist()
            category_labels = df_category[self.column_fallback_subcategory].astype(str).tolist()

            category_results = self.train(category_texts, category_labels)

            category_model, category_token, category_encoder = category_results

            category_model.save(os.path.join(self.path_models, 'model_fallback_' + intent + '.h5'))

            with open(os.path.join(self.path_models, 'tokenizer_fallback_' + intent + '.pickle'), 'wb') as file:
                pickle.dump(category_token, file)

            joblib.dump(category_encoder, os.path.join(self.path_models, 'encoder_fallback_' + intent + '.joblib'))


if __name__ == '__main__':
    train = model_train()
    train.run()