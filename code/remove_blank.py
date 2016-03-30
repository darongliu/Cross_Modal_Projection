"""
remove blank from image dir name
"""
import os

image_dir = "../images"

all_dir = os.listdir(image_dir)

for dir in all_dir :
    os.rename(os.path.join(image_dir,dir),os.path.join(image_dir,dir.strip()))
