from LodingData import LodingData 
import os
import tensorflow as tf
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    """
    this class for generating triplets that have the size of the batch size for the purpose of training
    instead of generating all images in the same time which consuming the memeory
    
    """

    def __init__(self, data_path,batch_size,empiding_size,training=True):
        self.path=data_path
        self.training_path=os.path.join(self.path,"training_faces")
        self.validation_path=os.path.join(self.path,"validation_faces")
        self.training_pickle_path=os.path.join(self.path,"training.pickle")
        self.val_pickle_path=os.path.join(self.path,"validation.pickle")
        self.train_faces={}
        self.val_faces={}
        self.batch_size=batch_size
        self.empiding_size=empiding_size
        self.training=training

        if training:
            if not os.path.exists(self.training_pickle_path):
                LodingData(self.path)
            
            with open(self.training_pickle_path,"rb") as f:
                self.train_faces=pickle.load(f)

        else:
            if not os.path.exists(self.val_pickle_path):
                LodingData(self.path)
            
            with open(self.val_pickle_path,"rb") as f:
                self.val_faces=pickle.load(f)
        
    
    def __getitem__(self,index):
        anchors=[]
        positives=[]
        negatives=[]
         
        for _ in range(self.batch_size):
            if self.training:
                path=self.training_path
                data=self.train_faces
            else:
                path=self.validation_path
                data=self.val_faces
            
            anchor_ind,positive_ind,negative_idx=self.triplet(data)
            anchors.append(self.load_image(path,list(data)[anchor_ind[0]],data[list(data)[anchor_ind[0]]][anchor_ind[1]]))
            positives.append(self.load_image(path,list(data)[positive_ind[0]],data[list(data)[positive_ind[0]]][positive_ind[1]]))
            negatives.append(self.load_image(path,list(data)[negative_idx[0]],data[list(data)[negative_idx[0]]][negative_idx[1]]))

        anchors=np.array(anchors)
        positives=np.array(positives)
        negatives=np.array(negatives)
        # self.ploting_batching_triplet(anchors,positives,negatives)

        X=[anchors,positives,negatives]
        y=np.zeros((self.batch_size,3*self.empiding_size))
        
        return X,y

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.training:
            data=self.train_faces
        else:
            data=self.val_faces

        return int(np.floor(len(data.keys()) / self.batch_size))


    def load_image(self,path,person,img_name):
        person_path=os.path.join(path,person)
        img_path=os.path.join(person_path,img_name)
        img=cv2.imread(img_path)
        img=cv2.resize(img,(224, 224))
        img_arr=np.asarray(img,dtype=np.float64)
        norm_img=self.preprocess_input(img_arr)

        return(norm_img)

    def triplet(self,data):
        """
        this function finding index for the one single triplet 

        returns: the index of anchor ,positive_image,negative_image
        """
        no_of_persons=len(data.keys())-1

        int_1=random.randint(0,no_of_persons)
        int_2=random.randint(0,no_of_persons) # the precentage of generating integer that equal to the above integer is very very low
        
        while int_1==int_2:
            int_2=random.randint(0,no_of_persons)
        
        int_3=len(list(data)[int_1])-1
        int_4=len(list(data)[int_2])-1


        anchor_ind=(int_1,random.randint(0,int_3))
        positive_ind=(int_1,random.randint(0,int_3))

        while anchor_ind==positive_ind:
            positive_ind=(int_1,random.randint(0,int_3))

        negative_ind=(int_2,random.randint(0,int_4))

        return anchor_ind,positive_ind,negative_ind


    def ploting_batching_triplet(self,anchors,positives,negatives):

        """
        this funct to plot patches to make sure that these patches are as we need

        """
        n=len(anchors)
        _,ax=plt.subplots(n,3)
        for i in range(n):
            ax[i,0].imshow(anchors[i])
            ax[i,1].imshow(positives[i])
            ax[i,2].imshow(negatives[i])
        plt.show()

    def preprocess_input(self,x, data_format=None, version=1):
        """
        this function is taken from keras_vggface as it is this model that we need to rebuild it in tensorflow v2.4 instead of version 1
        """
        x_temp = np.copy(x)
        K = tf.keras.backend
        if data_format is None:
            data_format = K.image_data_format()
        assert data_format in {'channels_last', 'channels_first'}

        if version == 1:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 93.5940
                x_temp[:, 1, :, :] -= 104.7624
                x_temp[:, 2, :, :] -= 129.1863
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 93.5940
                x_temp[..., 1] -= 104.7624
                x_temp[..., 2] -= 129.1863

        elif version == 2:
            if data_format == 'channels_first':
                x_temp = x_temp[:, ::-1, ...]
                x_temp[:, 0, :, :] -= 91.4953
                x_temp[:, 1, :, :] -= 103.8827
                x_temp[:, 2, :, :] -= 131.0912
            else:
                x_temp = x_temp[..., ::-1]
                x_temp[..., 0] -= 91.4953
                x_temp[..., 1] -= 103.8827
                x_temp[..., 2] -= 131.0912
        else:
            raise NotImplementedError

        return x_temp
