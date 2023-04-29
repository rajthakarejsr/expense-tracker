import json

from flask import Flask, request, jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text')

    # 2. vectorize
    vector_input = tfidf.transform([text])
    # 3. predict
    result = model.predict(vector_input)[0]

    return jsonify({'target': str(result)})


if __name__ == '__main__':
    app.run(debug=True)
