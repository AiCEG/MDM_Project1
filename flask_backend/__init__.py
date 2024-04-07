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




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D,BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


def train_sentiment_model():
    #read reviews.json
    df = pd.read_json('reviews.json')
    #remove rows with missing values


    #remove stars text from stars column
    df['stars'] = df['stars'].str.replace('stars', '')
    df['stars'] = df['stars'].str.strip()


    #map stars to sentiment
    df['sentiment'] = df['stars'].map({'1':0, '2':0, '3':0, '4':1, '5':1})

    #remove sentiment with Nan
    df = df.dropna(subset=['sentiment'])

    # Assuming 'df' is your DataFrame and it's already preprocessed
    texts = df['text'].values
    labels = df['sentiment'].values

    # Tokenizing text
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')

    # Padding sequences to ensure uniform input size
    data = pad_sequences(sequences, maxlen=100)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    vocab_size = 10000  # Example vocabulary size
    embedding_dim = 128  # Dimensionality of the embedding layer

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        Conv1D(64, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # For binary classification
    ])

    # Compilation of the model
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.01, verbose=1)

    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])

    # Evaluate the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy: {scores[1]}')


    # Sample new reviews
    new_reviews = [
        "I absolutely loved the food and the service was great!",
        "Worst experience ever. Will not be coming back.",
        "It was okay, nothing special.",
        "The ambiance was wonderful but the food was only average.",
        "Disappointed with the late delivery."
    ]

    # Convert the reviews to sequences
    sequences = tokenizer.texts_to_sequences(new_reviews)

    # Pad the sequences
    padded_sequences = pad_sequences(sequences, maxlen=100)

    # Predicting sentiment
    predictions = model.predict(padded_sequences)

    # Interpreting the predictions
    interpreted_predictions = [f"Positive {pred}" if pred > 0.7 else f"Negative {pred}" for pred in predictions.flatten()]

    # Printing the results
    for review, sentiment in zip(new_reviews, interpreted_predictions):
        print(f"Review: {review}\nPredicted Sentiment: {sentiment}")


    model.save('my_model.keras')  # This saves the trained model

    import pickle

    # Assuming `tokenizer` is your Keras Tokenizer instance
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


#Imports
import pandas as pd
from webdriver_manager.chrome import ChromeDriverManager
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def scrape_review_data():
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    reviews = []

    url = 'https://www.google.com/maps/search/Restaurants/@40.6919479,-74.104705,11.26z/data=!4m2!2m1!6e5?entry=ttu'
    driver.get(url)
    try:
        # Accept cookies
        WebDriverWait(driver,20).until(EC.element_to_be_clickable((By.XPATH,"//span[contains(@class, 'VfPpkd-vQzf8d') and text()='Accept all']"))).click()

        time.sleep(1)

        #scroll down the list so we have more restaurants to scrape
        scroll_duration = 30  # Duration in seconds for which you want to scroll down
        scroll_increment = 1000  # The amount to scroll on each iteration (adjust as needed)

        scrollable_div = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd > div.m6QErb.DxyBCb.kA9KIf.dS8AEf.ecceSd")  # Replace spaces with dots for CSS selector

        start_time = time.time()
        while time.time() - start_time < scroll_duration:
            driver.execute_script('arguments[0].scrollBy(0, arguments[1]);', scrollable_div, scroll_increment)
            time.sleep(0.3)  # Adjust sleep time as needed to control scroll speed

        # scrape the urls of the restaurants
        css_selector = ".Nv2PK.THOPZb.CpccDe a.hfpxzc"  # This targets <a> tags within elements with the specified classes
        elements = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, css_selector)))

        # Extracting the 'href' attribute of each element
        urls = [element.get_attribute('href') for element in elements]

        # Printing out all URLs

        for url in urls:
            driver.get(url)
            time.sleep(1.5)
            # Click on the 'Reviews' button
            button_xpath = "//div[contains(@class, 'RWPxGd')]//button[contains(@class, 'hh2c6') and starts-with(@aria-label, 'Reviews')]"
            button = WebDriverWait(driver, 30).until(EC.element_to_be_clickable((By.XPATH, button_xpath)))
            button.click()

            time.sleep(1)
            #scroll down the list so we have more reviews to scrape
            scroll_duration = 45  # Duration in seconds for which you want to scroll down
            scroll_increment = 1000  # The amount to scroll on each iteration (adjust as needed)

            scrollable_div = driver.find_element(By.CSS_SELECTOR, "div.m6QErb.DxyBCb.kA9KIf.dS8AEf")  # Replace spaces with dots for CSS selector

            start_time = time.time()
            while time.time() - start_time < scroll_duration:
                driver.execute_script('arguments[0].scrollBy(0, arguments[1]);', scrollable_div, scroll_increment)
                time.sleep(0.3)  # Adjust sleep time as needed to control scroll speed

            #scrape the reviews
            review_containers = WebDriverWait(driver, 30).until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.jJc9Ad")))

            for container in review_containers:
                try:
                    # Find the review text
                    review_text_span = container.find_element(By.CSS_SELECTOR, "div.MyEned > span")
                    review_text = review_text_span.text

                    # Find the star rating from the aria-label attribute of the span within the div.DU9Pgb
                    star_rating_span = container.find_element(By.CSS_SELECTOR, "div.DU9Pgb > span")
                    star_rating = star_rating_span.get_attribute("aria-label")

                    reviews.append({"text": review_text, "stars": star_rating})
                except NoSuchElementException:
                    continue
                except Exception as e:
                    print(e)
                    continue

    except Exception as e:
        print(e)

    finally:
        # Save the reviews as a json
        df = pd.DataFrame(reviews)
        df.to_json("reviews.json", orient="records")

    driver.quit()
