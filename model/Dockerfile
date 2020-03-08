FROM tensorflow/tensorflow:2.1.0-gpu-py3
RUN apt-get update && pip install --upgrade pip && apt-get install -y libsm6 libxext6 libxrender-dev
ENV QT_X11_NO_MITSHM=1
CMD bash
