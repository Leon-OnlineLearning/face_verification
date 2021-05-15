class Singleton_model:
   __model = None
   @staticmethod 
   def getInstance():
      """ Static access method. """
      if Singleton_model.__model == None:
         Singleton()
      return Singleton.__model
   def __init__(self):
      """ Virtually private constructor. """
      if Singleton_model.__model != None:
         raise Exception("This class is a singleton!")
      else:
         Singleton_model.__model = keras.models.load_model('./resnet50_triplet_loss_2048.h5', custom_objects={'tf': tf},compile=False)
