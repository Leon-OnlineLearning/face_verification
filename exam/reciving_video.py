import os
from flask import Flask,request,Response,jsonify
from flask_api import status
from werkzeug.utils import secure_filename
from flask import send_from_directory
from crop_faces import crop_faces
import numpy as np
from face_verification import face_verification
 
UPLOAD_FOLDER = './recordings'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/users', methods=['POST'])
def receving_video_before_exam():
    if 'student_video' in request.files:
        file = request.files['student_video']
        if file and file.filename != '':
            if file:
                filename = secure_filename(file.filename)
                if not os.path.exists(app.config['UPLOAD_FOLDER']):
                    os.mkdir(app.config['UPLOAD_FOLDER'])
                datapath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(datapath)
                vector=crop_faces(datapath)

                #return the embedding vactor 
                return jsonify({'embedding':vector.tolist()})

    #if there is no file or bad file recived return 400
    return "Inavalid video type", status.HTTP_400_BAD_REQUEST

@app.route('/exams/<user_id>', methods=[ 'POST'])
def receving_chunks_during_exam(user_id):
    #to make sure that every element in the array is in float type
    embedding = list(map(lambda x: float(x),request.form.values()))
    file = request.files['chunk'] #the video
    if file and file.filename != '':
        filename = secure_filename(file.filename)
        datapath=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(datapath)
        matched=face_verification(datapath,np.asarray(embedding))
        if matched:
            os.remove(datapath)
        return jsonify({'matched':matched})

    #if there is no file or bad file recived return 400
    return "Inavalid video type", status.HTTP_400_BAD_REQUEST

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=50000, debug=True)