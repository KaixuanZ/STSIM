import os
import json
import random

test_ids = [['0','2'], ['1','3'], ['5','9'], ['6','7'], ['4','8']]

for idx in range(5):
    train = []
    valid = []
    test = []
    test_id = test_ids[idx]
    output_dir = '/dataset/jana2012/5-fold_cv'
    v=str(idx)
    data_folders = ['/dataset/jana2012/train','/dataset/jana2012/test','/dataset/jana2012/validation']

    clean_names = lambda x: [i for i in x if i[0] != '.']
    for folder in data_folders:
        imgnames = sorted(clean_names(os.listdir(folder)))
        for imgname in imgnames:
            if imgname[0] in test_id:
                if folder == '/dataset/jana2012/test':  #?
                    test.append(os.path.join(folder, imgname))
            elif 'valid' in folder:
                valid.append(os.path.join(folder, imgname))
            else:
                train.append(os.path.join(folder, imgname))


    # random.shuffle(train)
    # N = len(train)
    # valid = train[int(N*0.9):]
    # train = train[:int(N*0.9)]

    with open(os.path.join(output_dir, 'train_'+v+'.json'), 'w') as json_file:
        json.dump(train, json_file)
    with open(os.path.join(output_dir, 'validation_'+v+'.json'), 'w') as json_file:
        json.dump(valid, json_file)
    with open(os.path.join(output_dir, 'test_'+v+'.json'), 'w') as json_file:
        json.dump(test, json_file)
# import pdb;pdb.set_trace()
