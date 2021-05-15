import numpy as np
import os
from zipfile import ZipFile
import tensorflow as tf
import cv2
import pickle

class LodingData:
    """
    this class for loading_images training and validation images from zip files
    
    """

    def __init__(self, data_path):
        self.path=data_path
        self.training_path=os.path.join(self.path,"training_faces")
        self.validation_path=os.path.join(self.path,"validation_faces")
        self.testing_path=os.path.join(self.path,"testing_faces")
        self.training_pickle_path=os.path.join(self.path,"training.pickle")
        self.val_pickle_path=os.path.join(self.path,"validation.pickle")
        self.test_pickle_path=os.path.join(self.path,"test.pickle")
        self.train_faces={}
        self.val_faces={}
        self.test_faces={}

        if not os.path.exists(self.training_path):
            self.extracting_images(self.training_path)

        if not os.path.exists(self.validation_path):
            self.extracting_images(self.validation_path)
        
        if not os.path.exists(self.testing_path):
            self.extracting_images(self.testing_path)


        if not os.path.exists(self.training_pickle_path):
            self.train_faces=self.dataset_dict(self.training_path)
            self.save_pickle(self.train_faces,"training.pickle")
            print("serialization of training data finished")

        if not os.path.exists(self.val_pickle_path):
            self.val_faces=self.dataset_dict(self.validation_path)
            self.save_pickle(self.val_faces,"validation.pickle")
            print("serialization of validation data finished")
        
        # if not os.path.exists(self.test_pickle_path):
        #     self.val_faces=self.dataset_dict(self.test_pickle_path)
        #     self.save_pickle(self.test_faces,"test.pickle")
        #     print("serialization of testing data finished")
            
        

    def extracting_images(self,path2):

        """
        extracting folders of persons from zip file

        """
        with ZipFile(path2+'.zip','r') as zip:
                print("extracting_training_images.....")
                zip.extractall(self.path)
                print("extracting_finished!")

    
    def dataset_dict(self,path):

        """
        loading images for training and testing
        
        returns:
        array of all images after normalization them 
        """
        dataset={}

        for person in os.listdir(path):
            person_path=os.path.join(path,person)
            dataset[person]=[]
            for imgname in os.listdir(person_path) :
                dataset[person].append(imgname)
        
        return dataset

    def save_pickle(self,arr,file_name):

        """
        insted of loding images more than one time(that take along time)
        loading them one time and save them in pickle file
        """

        with open (os.path.join(self.path,file_name),"wb") as f:
            pickle.dump(arr,f)
    

    

    



