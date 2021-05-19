FROM python:3.6-slim

WORKDIR /code

RUN apt update
RUN apt-get install ffmpeg libsm6 libxext6  -y


COPY requirements.txt ./

RUN pip install --no-cache -r requirements.txt
RUN pip install --no-cache flask_api
# due to bug https://github.com/Azure-Samples/ms-identity-python-webapp/issues/16
RUN pip install --no-cache Werkzeug==0.16.0 


COPY . .


CMD ["python","exam/reciving_vedio.py"]