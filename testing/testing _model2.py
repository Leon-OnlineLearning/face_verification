from LodingData import LodingData
from DataGenerator import DataGenerator
from tensorflow import keras
import tensorflow as tf
import numpy as np
import cv2
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

class thresh_calculation:
    def __init__(self, model_path,data_path):
        self.model=keras.models.load_model(model_path, custom_objects={'tf': tf},compile=False)
        self.data_path=data_path
        acc=self.acc_for_diff_thresh()
        self.plot_acc(acc)
      


    def acc_for_diff_thresh(self):
        acc=[]
        X=DataGenerator(self.data_path,100,2048,True)[0][0]
        for i in np.arange(.4,.9,.01):
            accuracy=self.calculate_acc(i,X)
            acc.append(accuracy)
            print(i," -acc = ",accuracy)
        return acc
    
    def plot_acc(self,acc):
        x=[i for i in np.arange(.4,.9,.01) ]
        y=acc
        indices=np.argmax(acc)
        chosen_thresh=x[indices]
        max_acc=acc[indices]

        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(8,6))
        plt.plot(x,y)
        plt.scatter(chosen_thresh,max_acc)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xlabel('thresh')
        plt.ylabel('accuracy')
        plt.title('thresh_acc-graph',fontsize=12)
        plt.show()
        plt.savefig("/home/mh1644/Desktop/High resoltion.png")
        
        print (chosen_thresh)

    def calculate_acc(self,thresh,X):
        
        count=0
        for i in range(100):
          img1 = tf.convert_to_tensor(cv2.resize(X[0][i],(224,224)))
          img2 = tf.convert_to_tensor(cv2.resize(X[1][i],(224,224)))
          img3 = tf.convert_to_tensor(cv2.resize(X[2][i],(224,224)))
          y1=tf.math.l2_normalize(self.model.predict(np.expand_dims(img1, axis=0)), axis=-1)
          y2=tf.math.l2_normalize(self.model.predict(np.expand_dims(img2, axis=0)), axis=-1)
          y3=tf.math.l2_normalize(self.model.predict(np.expand_dims(img3, axis=0)), axis=-1)
          if self.is_match(y1,y2,thresh)[0]:
            count+=1          
          if not self.is_match(y2,y3,thresh)[0]:
            count+=1
          if not self.is_match(y1,y3,thresh)[0]:
            count+=1

        accuracy=((count)/300)*100
        return accuracy


    def is_match(self,known_embedding, candidate_embedding, thresh):
        # calculate distance between embeddings
        score = tf.norm(known_embedding - candidate_embedding, axis=1)
        if score <= thresh:
            return (True,score)
        else:
            return (False,score)


data_path=r"/media/mh1644/My work/machine learning/deep_learning/computer_vision/face_verification/data/data2"
model_path=r'/media/mh1644/My work/machine learning/deep_learning/computer_vision/face_verification/code/resnet50_triplet_loss_2048.h5'
thresh_calculation(model_path,data_path)