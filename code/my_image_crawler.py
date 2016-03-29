import sys
import os
import urllib
from apiclient.discovery import build
import time

root_file = "./image_data"

all_triplet = []
#relation play
#left = ["people" , "dog" , "cat"]
#right = ["basketball" , "soccer" , "frisbee"]
#for l in left :
#    for r in right :
#        all_triplet.append([l , "play" , r])

#relation eat
#all_triplet = all_triplet + [["dog" , "eat" , "dog food"],["cat" , "eat" , "cat food"] , ["people" , "eat" , "burger"]]

#relation group
#all_triplet = all_triplet + [["dog" , "group" , "dog"],["cat" , "group" , "cat"] , ["people" , "group" , "people"]]

#relation study
left = ["people" , "dog" , "cat"]
right = ["book" , "computer"]
for l in left :
    for r in right :
        all_triplet.append([l , "study" , r])


service = build("customsearch", "v1",
               developerKey="AIzaSyAjiNb5uQCCxqPe3vFl1ZRVZcxiGF7cvOk")

for tri in all_triplet :
    name = tri[0] + " " + tri[1] + " " + tri[2]
    print "start crawling" , name
    count = 1
    for i in range(3) :
        print "round" , str(i+1)
        res = service.cse().list(
            q=name,
            cx='000109622348442705360:f_fblmv4vyu',
            searchType='image',
            num=10,
            imgType='clipart',
            safe= 'off',
            start=count
        ).execute()

        image_dir = os.path.join(root_file,name)
        if not os.path.exists(image_dir) : 
            os.makedirs(image_dir)

        for item in res['items']:
            image_name = item['title']
            url = item['link']
            print "downloading: " , image_name , " " , url
            try:
                urllib.urlretrieve(url, os.path.join(image_dir,image_name))
            except :
                print "url retrieve error"

        count = count + 10



