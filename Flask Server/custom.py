from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle

model1=tf.keras.models.load_model('lstm.h5')

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Custom input
# custom_input = "COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK"
# custom_input = "I like your post"
custom_input = "kill that enemey"

# Tokenize and pad the custom input
with open('tkn.pkl', 'rb') as f:
    tokenizer1 = pickle.load(f)
# tokenizer=Tokenizer(num_words=10000)
custom_seq = tokenizer1.texts_to_sequences([custom_input])
custom_pad = pad_sequences(custom_seq, maxlen=1000)


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


# # Tokenize the preprocessed data
# tokenizer = Tokenizer(num_words=10000)
# tokenizer.fit_on_texts(custom_input)
# X_seq = tokenizer.texts_to_sequences(custom_input)

# # Pad sequences to ensure uniform length
# X_pad = pad_sequences(X_seq, maxlen=1000)

# # Make predictions
# predictions = model1.predict(X_pad)

# # Assuming you want to print the predictions
# print(predictions[0])