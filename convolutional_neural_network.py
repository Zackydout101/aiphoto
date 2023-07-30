import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from flask import Flask
from flask_restful import Api, Resource

app = Flask(__name__)
api = Api(app)


class Emit(Resource):
    def get(self):
        class_labels = ['cat', 'dog', 'flower']

        # Part 4 - Making a single prediction
        loaded_model = tf.keras.models.load_model('my_model.h5')

        # Load and preprocess the image
        test_image = tf.keras.preprocessing.image.load_img('dataset/single_prediction/tico.jpg', target_size=(64, 64))
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)

        # Make a prediction
        result = loaded_model.predict(test_image/255.0)

        predicted_class_index = np.argmax(result[0])
        
        # Get the predicted class label
        prediction = class_labels[predicted_class_index]
        
        return prediction


api.add_resource(Emit, "/ask")

if __name__ == "__main__":
    app.run(debug=True)