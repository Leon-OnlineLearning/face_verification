import os
import cv2
import shutil
import numpy as np
from tqdm.notebook import tqdm
from mtcnn import MTCNN



def extract_face(old_path,new_path,required_size=(224, 224)):
    if not os.path.exists(new_path):
        image=cv2.imread(old_path)
        face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        grayImage = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
        detected_faces = face.detectMultiScale3(
        grayImage,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE,
        outputRejectLevels = True
)   
        if len(detected_faces[0]):
            if detected_faces[2][0][0]>5:

                # print(detected_faces[0][0])
                x1, y1, width, height= detected_faces[0][0]
                x2, y2 = x1 + width, y1 + height
                face=image[y1:y2, x1:x2]
                cv2.imwrite(new_path, face)
                print(1)
        else:
	        # detect faces in the image
            detector = Singleton_MTCNN.getInstance()
            results = detector.detect_faces(image)
	        # extract the bounding box from the first face
            # for j in range(len(results)):
            if len(results):
                if results[0]['confidence']>.95:
                    x1, y1, width, height = results[0]['box']
                    x2, y2 = x1 + width, y1 + height 
                    face=image[y1:y2, x1:x2]
                    if (face.shape[0]>0 and face.shape[1]>0):
                        cv2.imwrite(new_path, face)
                        print(2)

dataset_path = r'/media/mh1644/My work/machine learning/deep_learning/computer_vision/face_verification/data/data2/testing_faces'
path = r'/media/mh1644/My work/machine learning/deep_learning/computer_vision/face_verification/data/data2/archive3'

list_of_images = []


for dirname in tqdm(os.listdir(path)):
    image_folder_path = os.path.join(path, dirname)
    os.mkdir(os.path.join(dataset_path, dirname))
    for image in tqdm(os.listdir(image_folder_path), leave=True, position=1):
        image_path = os.path.join(image_folder_path, image)
        save_image = os.path.join(os.path.join(dataset_path, dirname), image)
        extract_face(image_path,save_image)

# path='/media/mh1644/My work/machine learning/deep_learning/computer_vision/livenes_detection/data/archive'
# path2='/media/mh1644/My work/machine learning/deep_learning/computer_vision/livenes_detection/data/dataset/not_live'
# j=0
# for image in tqdm(os.listdir(path), leave=True, position=1):
#     image_path=os.path.join(path,image)
#     save_image=os.path.join(path2,image)
#     extract_face(image_path,save_image,j)
#     j+=1



