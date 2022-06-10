import imp
from app import app
import os
from flask import request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D, Convolution2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from app.model.model import create_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# call conda activate p374
# set FLASK_APP=server.py
# call flask run --port=5000

target_size = (224,224)

# warm-up
labels = pd.read_csv(os.path.join('app','images_labelling.csv'))
labels = labels.drop_duplicates(subset=['label'])
model = create_model(target_size=target_size)
model.load_weights(os.path.join('app', 'model', 'checkpoint'))

@app.route('/predict', methods=['POST'])
def run_inference():
    
    # save file
    f = request.files['file']
    f.save(os.path.join('app', 'images', '0', 'image.png'))
    
    # use generator to be sure that preprocessing is the same
    pred_datagen = ImageDataGenerator(rescale=1./255)
    pred_generator = pred_datagen.flow_from_directory(
        'app/images',
        target_size=target_size,
        batch_size=1,
        class_mode='categorical')
    res = np.argmax(model.predict(pred_generator))
    del pred_generator, pred_datagen

    #prepare response
    label = str(labels[['label','class_']][labels.label==res].label.values[0])
    class_= str(labels[['label','class_']][labels.label==res].class_.values[0])

    # send it
    return jsonify({"class_":class_, 'label':label}), 200