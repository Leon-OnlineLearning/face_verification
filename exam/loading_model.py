import numpy as np
from tensorflow import keras
import tensorflow as tf

class Singleton_model:
   __model = None
   @staticmethod 
   def getInstance():
      if Singleton_model.__model == None:
         Singleton_model()
      return Singleton_model.__model

   
   def __init__(self):
      if Singleton_model.__model != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton_model.__model = keras.models.load_model('./resnet50_triplet_loss_2048.h5', custom_objects={'tf': tf},compile=False)
