import os

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping
from flask_cors import CORS
import pickle

import threading


def create_model():
    model = Sequential([
        Conv1D(64, 5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',  # Assuming binary classification
                  metrics=['accuracy'])

    return model

def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    model = create_model()

    model = tf.keras.models.load_model('sentiment_analysis_model.keras')

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass


    @app.route('/predict', methods=['POST'])
    def predict():
        data = request.json
        reviews = data['reviews']

        # Preprocess the reviews as done during the training
        sequences = tokenizer.texts_to_sequences(reviews)
        padded_sequences = pad_sequences(sequences, maxlen=100)

        # Predict sentiment
        predictions = model.predict(padded_sequences)

        # Interpret predictions
        interpreted_predictions = ["Positive" if pred > 0.7 else "Negative" for pred in predictions.flatten()]

        # Return the predictions
        return jsonify({"predictions": interpreted_predictions})


    @app.route('/train', methods=['GET'])
    def train():
        thread = threading.Thread(target=train_sentiment_model)
        thread.start()
        return jsonify({"message": "Training started."}), 202


    @app.route('/scrape', methods=['GET'])
    def scrape():
        thread = threading.Thread(target=scrape_review_data)
        thread.start()
        return jsonify({"message": "Scraping started."}), 202

    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello, World!'

    return app
