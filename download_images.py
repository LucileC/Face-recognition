import requests
import re
from requests.exceptions import ConnectionError
import os
import imghdr


act = ['Gerard Butler', 'Michael Vartan',  'Angie Harmon','Daniel Radcliffe', 'Lorraine Bracco', 'Peri Gilpin',]



with open('subset_actors_gendertest.txt') as f:
    lines = f.readlines()
currentName = None
i=0
for line in lines:
    l = re.split(r'\t+', line.rstrip('\t'))
    name = l[0]
#    if name in act:
    if name != currentName:
            currentName = name
            i=0
#                if name == 'Angie Harmon':
#                    i = 226
    url = l[3]
    print(name + i.__str__())
    try:
        img_data = requests.get(url).content
    except ConnectionError as e:    
        print(e)
        img_data = None
    #img_name = "uncropped_faces/"+name+"_"+i.__str__()+'.jpg'
    img_name = "genderTest/male/"+name+"_"+i.__str__()+'.png'
    with open(img_name, 'wb') as handler:
        if img_data:
            handler.write(img_data)
    if imghdr.what(img_name) == None:
        os.remove(img_name)
    i += 1
        
        
