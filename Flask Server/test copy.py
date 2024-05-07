# !pip install nltk
import nltk
import pandas as pd
import numpy as np
import string
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pandas import read_csv
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, LSTM, Dropout, Embedding, Dense, Bidirectional,Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, CategoricalAccuracy, Recall


import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



data = read_csv('train.csv')

# Function to preprocess text data
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove numerical values
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize the text
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    # Join the lemmatized words back into a single string
    preprocessed_text = ' '.join(lemmatized_words)
    return preprocessed_text

# Apply the preprocessing function to the "comment_text" column
data['comment_text'] = data['comment_text'].apply(preprocess_text)


# Create subsets based on toxic and clean comments
column_labels = data.columns.tolist()[2:]
label_counts = data[column_labels].sum().sort_values()


train_toxic = data[data[column_labels].sum(axis=1) > 0]
train_clean = data[data[column_labels].sum(axis=1) == 0]



# Randomly sample 15,000 clean comments
train_clean_sampled = train_clean.sample(n=16225, random_state=42)

# Combine the toxic and sampled clean comments
dataframe = pd.concat([train_toxic, train_clean_sampled], axis=0)

# Shuffle the data to avoid any order bias during training
dataframe = dataframe.sample(frac=1, random_state=42)



X = dataframe['comment_text']
y = dataframe[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]

# Tokenize the input text
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Pad sequences to ensure uniform length
max_seq_length = 100
X_pad = pad_sequences(X_seq, maxlen=max_seq_length)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)



# Build the LSTM model
embedding_dim = 50

modelLSTM = Sequential([
   
    Embedding(max_words, embedding_dim),
    LSTM(50),
    Dense(6, activation='sigmoid')
])
  # Use 6 units for the six categories

modelLSTM.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Compile the model
modelLSTM.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
historyLSTM=modelLSTM.fit(X_train, y_train, epochs=5, batch_size=128, validation_split=0.2)


modelLSTM.save('lstm.h5')
import pickle
pickle.dump(tokenizer,open('tkn.pkl','wb'))

# Custom input
custom_input = "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"

# Tokenize and pad the custom input
# ===================================================
# tokenizer1=Tokenizer(num_words=10000)
with open('tkn.pkl', 'rb') as f:
    tokenizer1 = pickle.load(f)
# ====================================
custom_seq = tokenizer1.texts_to_sequences([custom_input])
custom_pad = pad_sequences(custom_seq, maxlen=max_seq_length)

model1=tf.keras.models.load_model('lstm.h5')
# Make predictions
custom_pred = model1.predict(custom_pad)

# Threshold the predictions (assuming binary classification)
threshold = 0.5
custom_pred_binary = (custom_pred > threshold).astype(int)

# Interpret the predictions
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
predictions_dict = dict(zip(labels, custom_pred_binary[0]))

# Print the predictions
print("Predictions for the custom input:")
for label, prediction in predictions_dict.items():
    print(f"{label}: {'Toxic' if prediction == 1 else 'Not Toxic'}")

print(custom_pred_binary)
# Check if any label contains 1
if 1 in custom_pred_binary:
    print("Toxic")
else:
    print("Non-Toxic")
print('toxicity labels')
# Get the labels which contain value 1
toxic_labels = [labels[i] for i, val in enumerate(custom_pred_binary[0]) if val == 1]

# Print the labels
if toxic_labels:
    print("Toxic Labels:", toxic_labels)
else:
    print("Non-Toxic")

