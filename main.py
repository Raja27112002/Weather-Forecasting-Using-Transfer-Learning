from flask import Flask, request, jsonify,render_template
from tensorflow import keras
import tensorflow as tf

import numpy as np
import cv2


app = Flask(__name__)
@app.route('/',methods=['post','get'])
def homepage():
    return render_template('index.html')

model = keras.models.load_model("weathermodel.h5")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image'].read()

    file_bytes = np.fromstring(file, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (256, 256))  # Resize the image if necessary
    img_array = np.expand_dims(img, axis=0)

    prediction = model.predict(img_array)
    class_names = ['ClearSky', 'Cloudy', 'Rain', 'Sunrise']
    prediction = class_names[np.argmax(prediction)]

    # Assuming binary classification
    return render_template('results.html', prediction=prediction)



if __name__ == '__main__':
    app.run(debug=True)
