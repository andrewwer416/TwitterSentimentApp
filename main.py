from waitress import serve
import pandas as pd
from tensorflow import keras
import numpy as np
import API_client
import data_preprocess
from flask import Flask, render_template, request
app=Flask(__name__)
#load model
model = keras.models.load_model('my_model')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    query = request.form['search_term']
    twitter_data = API_client.download(search=query)
    seq = []
    for i in range(0, len(twitter_data.data)):
        seq.append({'text': str(twitter_data.data[i])})
    df = pd.DataFrame(seq)
    data_to_pred = data_preprocess.text_preprocess(df)
    prediction = model.predict(data_to_pred)
    prediction[prediction >= 0.5] = int(1)
    prediction[prediction < 0.5] = int(0)
    #print(prediction)
    positive_count = np.count_nonzero(prediction == 1)
    positive_ratio = positive_count / len(twitter_data.data) * 100
    positive_ratio = round(positive_ratio, 2)
    result = (str(positive_ratio) + "% of the past 100 Tweets for " + query + " are positive")
    return render_template('index.html', result=result)


if __name__ == '__main__':
    #app.run(debug=True)
    serve(app, host='0.0.0.0', port=80)
