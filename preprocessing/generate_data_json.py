import os
import json
import random

train = []
test = []
test_id = ['4','5','7']

data_folders = ['/dataset/jana2012/train','/dataset/jana2012/test','/dataset/jana2012/validation']

clean_names = lambda x: [i for i in x if i[0] != '.']
for folder in data_folders:
    imgnames = sorted(clean_names(os.listdir(folder)))
    for imgname in imgnames:
        if imgname[0] in test_id:
            if folder == '/dataset/jana2012/test':  #?
                test.append(os.path.join(folder, imgname))
        else:
            train.append(os.path.join(folder, imgname))

random.shuffle(train)
N = len(train)
valid = train[int(N*0.9):]
train = train[:int(N*0.9)]

with open('train.json', 'w') as json_file:
    json.dump(train, json_file)
with open('validation.json', 'w') as json_file:
    json.dump(valid, json_file)
with open('test.json', 'w') as json_file:
    json.dump(test, json_file)
# import pdb;pdb.set_trace()
