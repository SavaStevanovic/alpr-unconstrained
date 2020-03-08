docker build -t tf21playground .
xhost + 
docker run -e DISPLAY=$DISPLAY -it -v `pwd`/project:/app -v /tmp/.X11-unix:/tmp/.X11-unix -p 6006:6006 --gpus all tf21playground
