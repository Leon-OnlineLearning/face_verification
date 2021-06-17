FROM tensorflow/tensorflow:2.4.2

WORKDIR /code

RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y


COPY requirements.txt ./

RUN pip install --no-cache -r requirements.txt


COPY . .


CMD ["python","exam/reciving_vedio.py"]