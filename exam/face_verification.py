from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
from preprocess_input import preprocess_input,gamma_correction
from loading_model import Singleton_model,Singleton_MTCNN
import logging

def face_verification(chunk_path,known_embedding):
    detector = Singleton_MTCNN.getInstance()
    vs = cv2.VideoCapture(str(chunk_path))
    read=0 #frame reading counter
    matched=0 #matched counter
    procedded_fram=0 # the number of frame which is procedded

    # loop over some frames
    while True:
        # grab the frame from the threaded video stream 
        (grabbed,frame) = vs.read()
        read += 1

        #When the video ends 
        if not grabbed:
            if procedded_fram:
                match_presentage=(matched)/(procedded_fram)
                if match_presentage>.8:
                    return(True)
            return(False)

        # check to see if we should process this frame
        if read % 15 == 0:
            #detection any face in the frame
            face = detector.detect_faces(frame)

            # if any detection has confidence more than 95%
            if len(face):
                if face[0]['confidence']>.95:
                    x1, y1, width, height = face[0]['box']
                    x2, y2 = x1 + width, y1 + height
                    #extracting the face 
                    face=frame[y1:y2, x1:x2]
                    #insuring that face has shape not adummy face
                    if (face.shape[0]>0 and face.shape[1]>0):
                        candidate_embedding=embedding_calculating(face)
                        flag=is_match(known_embedding,candidate_embedding)
                        procedded_fram+=1
                        if flag:
                            matched+=1
                        
"""     
take an array of refrence embeddings and the currecnt embedding
and return if matched or not
"""
def is_match(known_embedding, candidate_embedding, thresh=.6):
    match=0
    matched=False
    score = tf.norm(known_embedding- candidate_embedding, axis=1)
    logging.info(score)
    print(score)
    if score <= thresh:
        matched=True
    return(matched)

"""
taking an image and return the embedding of theat image
"""
def embedding_calculating(img):
    img=gamma_correction(img)
    img=cv2.resize(img,(224, 224))
    img_arr=np.asarray(img,dtype=np.float64)
    norm_img=preprocess_input(img_arr)
    tens_img= tf.convert_to_tensor(cv2.resize(norm_img,(224,224)))
    model=Singleton_model.getInstance()
    # model=keras.models.load_model('./resnet50_triplet_loss_2048.h5', custom_objects={'tf': tf},compile=False)
    embedding=tf.math.l2_normalize(model.predict(np.expand_dims(tens_img, axis=0)), axis=-1)
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