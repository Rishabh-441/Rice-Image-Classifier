from flask import Flask, render_template, request

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import numpy as np

app = Flask(__name__)
vgg16_model = load_model('final_model.h5')

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        folder = './static/images'
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

        imageFile = request.files['imageFile']
        image_path = "./static/images/" + imageFile.filename
        imageFile.save(image_path)

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        prediction = vgg16_model.predict(img_array)

        result = np.argmax(prediction)

        img_dic = {0: 'Karacadag', 1: 'Basmati', 2: 'Jasmine', 3: 'Arborio', 4: 'Ipsala'}

        max_prob = np.max(prediction)
        total_sum = np.sum(prediction)

        percentage_max_prob = (max_prob / total_sum) * 100
        probability = f"{percentage_max_prob:.2f}"

    return render_template('index.html', image_path=os.path.join('/static/images', imageFile.filename),
                           prediction=img_dic[result], probability=probability)


if __name__ == '__main__':
    app.run(port=3000, debug=True)
