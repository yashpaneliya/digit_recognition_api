from django.apps import AppConfig
from keras.models import load_model
from django.conf import settings
import pickle
import os


class DigitrecappConfig(AppConfig):
    name = 'digitrecapp'
    # loading model
    model_path = os.path.join(settings.MODELS, 'digitdemo1.h5')
    digitmodel = load_model(model_path)



