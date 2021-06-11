import numpy as np
from tensorflow import keras
import tensorflow as tf
from mtcnn import MTCNN

import threading

class Singleton_model:
   __model = None
   @staticmethod 
   def getInstance():
      print("auth model requested")
      print(Singleton_model.__model)
      if Singleton_model.__model == None:
         Singleton_model()
      return Singleton_model.__model

   
   def __init__(self):
      if Singleton_model.__model != None:
         raise Exception("This class is a singleton!")
      else:
         # Singleton_model.__model = keras.models.load_model('./exam/resnet50_triplet_loss_2048.h5', custom_objects={'tf': tf},compile=False)
         Singleton_model.__model = keras.models.load_model('.\\auth_model')



class Singleton_MTCNN:
   __model = None
   @staticmethod 
   def getInstance():
      print("mtcnn requested")
      print(Singleton_MTCNN.__model)
      if Singleton_MTCNN.__model == None:
         Singleton_MTCNN()
      return Singleton_MTCNN.__model

   
   def __init__(self):
      if Singleton_MTCNN.__model != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton_MTCNN.__model = MTCNN()
