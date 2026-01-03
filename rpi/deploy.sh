# script to move the code to raspberry pi and run the service

# copy the code to raspberry pi
scp -r ../rpi/ rpi5:/home/rpi5/bird-detection/

# run the service
# ssh root@192.168.1.100 "cd /root/bird-detection && ./run.sh"