from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key'

model=load_model("alzimers.h5")


@app.route('/')
def home():
    return render_template('index.html', title='index')

@app.route("/performance")
def performance():
    return render_template("performance.html")

@app.route('/uploadimagepage')
def uploadimagepage():
    return render_template('uploadimage.html', title='upload image')
      

def preprocess(image_path):
    test_image = load_img(image_path, target_size = (224,224))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image/255
    return test_image



@app.route('/uploadimage', methods=['GET', 'POST'])
def uploadimage():
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        filename = 'static/uploads/' + uploaded_file.filename
        uploaded_file.save(filename)


        # Preprocess image
        test_image = preprocess(filename)
        prediction = model.predict(test_image)[0]   # raw prediction array

        class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

        # Find predicted label
        predicted_label = class_names[np.argmax(prediction)]
        display_prob = f"{np.max(prediction) * 100:.2f}%"  # string only for display

        # Extract numeric probabilities for graphs
        Mild_prob = round(float(prediction[0] * 100), 2)
        Moderate_prob = round(float(prediction[1] * 100), 2)
        Nondem_prob = round(float(prediction[2] * 100), 2)
        Very_Mild_prob = round(float(prediction[3] * 100), 2)

        return render_template(
            'uploadimage.html',
            result=predicted_label,
            prob=display_prob,
            Mild_prob=Mild_prob,
            Moderate_prob=Moderate_prob,
            Nondem_prob=Nondem_prob,
            Very_Mild_prob=Very_Mild_prob
        )
    return redirect(url_for('uploadimagepage'))

if __name__ == '__main__':
    app.run(debug=True)
