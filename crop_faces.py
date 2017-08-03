# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:43:28 2017

@author: luluc
Crop the faces, rescale and convert to ray scale
"""

import scipy
from PIL import Image
import re
from pathlib import Path
from os import walk, remove

act = ['Gerard Butler', 'Michael Vartan',  'Angie Harmon','Daniel Radcliffe', 'Lorraine Bracco', 'Peri Gilpin',]



#151,55,329,233
#img = scipy.misc.imread("uncropped_faces/Angie Harmon_34.jpg")
#cropped = img[55:233, 151:329]
#croppedimg = Image.fromarray(cropped, 'RGB')
#croppedimg.save("test.jpg")
            
# Read list of actor / actresses to work with
with open('subset_actors_genderTest.txt') as f:
    lines = f.readlines()
    
#Crop faces
currentName = None
i=0
#go through lines and get crop coordinates. 
#if the file corresponding to the current line exists, crop it and save into a cropped_faces folder.
for line in lines:
    l = re.split(r'\t+', line.rstrip('\t'))
    name = l[0]
#    if name in act:
    if name != currentName:
            currentName = name
            i=0
#        uncropped_path_name = "uncropped_faces/"+name+"_"+i.__str__()+".jpg"
#        cropped_path_name = "cropped_faces/"+name+"_"+i.__str__()+".jpg"       
    uncropped_path_name = "genderTest/male/"+name+"_"+i.__str__()+".png"
    cropped_path_name = uncropped_path_name
    my_file = Path(uncropped_path_name)
    if my_file.is_file():
        #print(uncropped_path_name)
        face_coord_line = l[4]
        face_coord = [x.strip() for x in face_coord_line.split(',')]
        x1 = int(face_coord[0])
        y1 = int(face_coord[1])
        x2 = int(face_coord[2])
        y2 = int(face_coord[3])
        try:
            img = scipy.misc.imread(uncropped_path_name)
            cropped = img[y1:y2, x1:x2]
            croppedimg = Image.fromarray(cropped, 'RGB')
            croppedimg.save(cropped_path_name)
        except ValueError as e:
            print(name+i.__str__()+": "+e.__str__())
        #croppedimg.show()
    i += 1
   
    
#go through each file of the cropped_faces folder and resize + convert to graysacle (need to go from jpg to png)
f = []
for (dirpath, dirnames, filenames) in walk("genderTest/male"):
    f.extend(filenames)
    break
for file in f:
    #get file_name without jpg extension
    l = [x for x in map(str.strip, file.split('.')) if x]
    file_name = l[0]
    cropped_path_name = "genderTest/male/"+file
    resized_path_name = cropped_path_name
    img = Image.open(cropped_path_name)
    img = img.resize((32, 32), Image.ANTIALIAS)
    img = img.convert('LA')
    #save as png
    img.save(resized_path_name)