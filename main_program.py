# Import Library from Flask
from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyJSONEncoder
from flasgger import swag_from
from flask import Markup


# Import library or framework for Machine Learning process
import pickle, re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Set Flask's JSON encoder to LazyJSONEncoder
app.json_encoder = LazyJSONEncoder

# Swagger template with standard strings
swagger_template = {
    "info": {
        "title": "Api Documentation for Sentiments Analysis API",
        "version": "1.0.0",
        "description": "Dokumentasi API untuk Sentiment Analysis API",
    },
    "host": "127.0.0.1:5000",  # You can set the host directly here
}
# Swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "docs",
            "route": "/docs.json",
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
}
# Initialize Swagger
swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Define Feature Extraction parameter and Tokenizer class
max_features = 100000
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

# Define label of sentiments
sentiment = ['negative', 'neutral', 'positive']

# Cleansing process
def cleansing(sent):
    # Mengubah kata menjadi huruf kecil semua dengan menggunakan fungsi lower()
    string = sent.lower()
    # Menghapus emoticon dan tanda baca menggunakan "RegEx" dengan script di bawah
    string = re.sub(r'[^a-zA-Z0-9]', ' ', string)
    return string

# LSTM
## Load result of Feature Extraction process from LSTM
file = open("model_of_lstm/x_pad_sequences.pickle",'rb')
feature_file_from_lstm = pickle.load(file)
file.close()

## Load model from LSTM
model_file_from_lstm = load_model('model_of_lstm/model.h5')

# NN
## Load result of Feature Extraction process from NN
file = open("model_of_nn/feature.p",'rb')
tfidf_vec = pickle.load(file)
file.close()

## Load model from NN
model_file_from_nn = pickle.load(open('model_of_nn/model.p', 'rb'))

# Default route when not on swagger
@app.route('/', methods=['GET'])
def home():
    # Define API response
    json_response = {
        'status_code': 200,
        'description': 'Go to <a href="/docs">/docs</a> to see Api Documentation'
    }
    response_data = Markup(json_response)
    return response_data

# Define endpoint for Sentiment Analysis using LSTM
@swag_from("docs/lstm.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    # Get text
    original_text = request.form.get('text')
    # Cleansing
    text = [cleansing(original_text)]
    # Feature extraction
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    # Inference
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data

# Define endpoint for Sentiment Analysis using LSTM from file
@swag_from("docs/lstm_file.yml", methods=['POST'])
@app.route('/lstm-file', methods=['POST'])
def lstm_file():

    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding='latin-1')

    # Get text from file in "List" format
    texts = df.Tweet.to_list()

    # Loop list or original text and predict to model
    text_with_sentiment = []
    for original_text in texts:

        # Cleansing
        text = [cleansing(original_text)]
        # Feature extraction
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        # Inference
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]

        # Predict "text_clean" to the Model. And insert to list "text_with_sentiment".
        text_with_sentiment.append({
            'text': original_text,
            'sentiment': get_sentiment
        })
    
    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data

# Define endpoint for Sentiment Analysis using NN
@swag_from("docs/nn.yml", methods=['POST'])
@app.route('/nn', methods=['POST'])
def cnn():
    # Get text
    original_text = request.form.get('text')
    # Cleansing
    text = cleansing(original_text)
    # Feature extraction
    text_feature = tfidf_vec.transform([text])
    # Inference
    get_sentiment = model_file_from_nn.predict(text_feature)[0]

    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using CNN",
        'data': {
            'text': original_text,
            'sentiment': get_sentiment
        },
    }
    response_data = jsonify(json_response)
    return response_data


# Define endpoint for Sentiment Analysis using NN from file
@swag_from("docs/nn_file.yml", methods=['POST'])
@app.route('/nn-file', methods=['POST'])
def nn_file():

    # Upladed file
    file = request.files.getlist('file')[0]

    # Import file csv ke Pandas
    df = pd.read_csv(file, encoding='latin-1')

    # Get text from file in "List" format
    texts = df.Tweet.to_list()

    # Loop list or original text and predict to model
    text_with_sentiment = []
    for original_text in texts:

        # Cleansing
        text = cleansing(original_text)
        # Feature extraction
        text_feature = tfidf_vec.transform([text])
        # Inference
        get_sentiment = model_file_from_nn.predict(text_feature)[0]

        # Predict "text_clean" to the Model. And insert to list "text_with_sentiment".
        text_with_sentiment.append({
            'text': original_text,
            'sentiment': get_sentiment
        })
    
    # Define API response
    json_response = {
        'status_code': 200,
        'description': "Teks yang sudah diproses",
        'data': text_with_sentiment,
    }
    response_data = jsonify(json_response)
    return response_data

if __name__ == '__main__':
   app.run()