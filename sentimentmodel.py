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
