from flask import Flask, request, jsonify
from fastai.basics import *
from fastai.vision.all import *
from flask_cors import CORS, cross_origin

app = Flask(__name__)

learn = load_learner('trained_model.pkl')


def predict_image(img):
    prediction = learn.predict(PILImage(img))
    if prediction[0] == 'hot_dog':
        return 'hot dog'
    return 'not hot dog'


@app.route('/predict', methods=['POST'])
def predict():
    return predict_image(request.files['image'])


if __name__ == '__main__':
    app.run
