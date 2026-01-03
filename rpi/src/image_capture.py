import time
import os
import argparse
import json
from picamera2 import Picamera2, Preview
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to the config file")
    args = parser.parse_args()
    return args


def main(args):
    with open(args.config, "r") as f:
        config = json.load(f)
        
    sensor_mode = config["mode_selected"]
    image_folder = config["SSD_image_folder"]
    number_of_images = config["number_of_images"]
    number_of_imgs_per_second = config["number_of_imgs_per_second"]
    
    os.makedirs(image_folder, exist_ok=True)
    picam2 = Picamera2()
    modes = picam2.sensor_modes
    picam2.sensor_mode = modes[sensor_mode]
    capture_config = picam2.create_still_configuration(raw={'format':modes[sensor_mode]['format'], 
                                                        'bit_depth':modes[sensor_mode]['bit_depth'], 
                                                        'fps':modes[sensor_mode]['fps']})
    picam2.configure(capture_config)
    picam2.start()
    
    
    # get the current time
    # create a folder with the current time, YYYYMMDD_HHMMSS
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(image_folder, current_time)
    os.makedirs(folder_name, exist_ok=True)
    
    # capture the images
    for i in range(number_of_images):
        curr_time = time.time()
        # get the current time and then add the image number to it
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        picam2.capture_file(f"{folder_name}/{current_time}_{i:04d}.jpeg")
        end_time = time.time()
        time.sleep(max(0, 1/number_of_imgs_per_second - (end_time - curr_time)))
    
    picam2.stop()
    
    
    
    

if __name__ == "__main__":
    args = parse_args()
    main(args)