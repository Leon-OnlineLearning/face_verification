from loading_model import Singleton_model,Singleton_MTCNN
from flask import Flask,request,Response,jsonify
from crop_faces import extract_face
import numpy as np

app = Flask(__name__)


@app.before_first_request
def initialize():
    Singleton_MTCNN.getInstance()
    print ("Called only once, when the first request comes in")

@app.route('/')
def hello():
    extract_face(np.zeros([500,500,3]))
    return "Hello"

if __name__ == '__main__':
    
    app.run(host='0.0.0.0',port=50000, debug=True)