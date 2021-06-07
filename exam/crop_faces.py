import os
import cv2
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from preprocess_input import preprocess_input,gamma_correction
from loading_model import Singleton_model ,Singleton_MTCNN



def extract_face(image):
    detector = Singleton_MTCNN.getInstance()
    results = detector.detect_faces(image)
	# extract the bounding box from the first face
    if len(results):
        if results[0]['confidence']>.95:
            x1, y1, width, height = results[0]['box']
            x2, y2 = x1 + width, y1 + height 
            face=image[y1:y2, x1:x2]
            if (face.shape[0]>0 and face.shape[1]>0):
               return(face,1)
    return(False,0)

def embedding_mean_calculating(faces):
    img_arr=np.asarray(faces,dtype=np.float64)
    tens_img= tf.convert_to_tensor(img_arr)
    # model=keras.models.load_model('./resnet50_triplet_loss_2048.h5', custom_objects={'tf': tf},compile=False)
    model=Singleton_model.getInstance()
    embedding=tf.math.l2_normalize(model.predict(tens_img), axis=-1)
    feature=np.mean(embedding, axis=0)
    return(feature)


def crop_faces(video_path):
    print(video_path)
    vs = cv2.VideoCapture(str(video_path))
    read=0
    faces=[]
    # loop over some frames...this time using the threaded stream
    while True:
        # grab the frame from the threaded video stream and resize it
        (grabbed,frame) = vs.read()
        read += 1
        #When the video ends 
        if not grabbed:
            break
        # check to see if we should process this frame
        if read % 15 == 0:
            face,exist=extract_face(frame)
            if exist:
                img=gamma_correction(face)
                img=cv2.resize(img,(224, 224))
                img_arr=np.asarray(img,dtype=np.float64)
                norm_img=preprocess_input(img_arr)
                tens_img= tf.convert_to_tensor(norm_img)
                faces.append(tens_img)
    
    embedding=embedding_mean_calculating(faces)
    return(embedding)

# """
# this function is taken from keras_vggface as it is this model that we need to rebuild it in tensorflow v2.4 instead of version 1
# """
# def preprocess_input(x, data_format=None, version=1):
#     x_temp = np.copy(x)
#     K = tf.keras.backend
#     if data_format is None:
#         data_format = K.image_data_format()
#     assert data_format in {'channels_last', 'channels_first'}
#     if version == 1:
#         if data_format == 'channels_first':
#             x_temp = x_temp[:, ::-1, ...]
#             x_temp[:, 0, :, :] -= 93.5940
#             x_temp[:, 1, :, :] -= 104.7624
#             x_temp[:, 2, :, :] -= 129.1863
#         else:
#             x_temp = x_temp[..., ::-1]
#             x_temp[..., 0] -= 93.5940
#             x_temp[..., 1] -= 104.7624
#             x_temp[..., 2] -= 129.1863
#     elif version == 2:
#         if data_format == 'channels_first':
#             x_temp = x_temp[:, ::-1, ...]
#             x_temp[:, 0, :, :] -= 91.4953
#             x_temp[:, 1, :, :] -= 103.8827
#             x_temp[:, 2, :, :] -= 131.0912
#         else:
#             x_temp = x_temp[..., ::-1]
#             x_temp[..., 0] -= 91.4953
#             x_temp[..., 1] -= 103.8827
#             x_temp[..., 2] -= 131.0912
#     else:
#         raise NotImplementedError
#     return x_temp






