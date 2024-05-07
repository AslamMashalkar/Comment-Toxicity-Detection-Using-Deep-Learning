
from flask import Flask, render_template,request,jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app,origins=["http://127.0.0.1:3000","http://localhost:3000"])
model1=tf.keras.models.load_model('lstm.h5')

model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Tokenize and pad the custom input
with open('tkn.pkl', 'rb') as f:
    tokenizer1 = pickle.load(f)

@app.route("/", methods=['GET', 'POST'])
# def index():

    
#     return render_template("index.html")

def analyze_sentiment():
    if request.method == 'POST':
    # if request.method == 'GET':
        req_data = request.get_json()
        comment = req_data['comment']
        # comment = request.form.get('comment')
        # comment = "love  you"
        print("printing custom comment")
        print(comment)
       

       # tokenizer=Tokenizer(num_words=10000)
        custom_seq = tokenizer1.texts_to_sequences([comment])
        custom_pad = pad_sequences(custom_seq, maxlen=1000)


        # Make predictions
        custom_pred = model1.predict(custom_pad)

        # Threshold the predictions (assuming binary classification)
        threshold = 0.5
        custom_pred_binary = (custom_pred > threshold).astype(int)

        # Interpret the predictions
        labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        predictions_dict = dict(zip(labels, custom_pred_binary[0]))

  
        x='' 
        # Check if any label contains 1
        if 1 in custom_pred_binary:
            print("Toxic")
            x='Toxic'
        else:
            print("Non-Toxic")
            x="Non-Toxic"
        print('toxicity labels')
        # Get the labels which contain value 1
        toxic_labels = [labels[i] for i, val in enumerate(custom_pred_binary[0]) if val == 1]

        
        # Print the labels
        if toxic_labels:
            print("Toxic Labels:", toxic_labels)
        else:
            print("Non-Toxic")
        return jsonify(
            isToxic=x,
            labels=toxic_labels
        )

 
if __name__ == "__main__":
    app.run(debug=True,port=5050)