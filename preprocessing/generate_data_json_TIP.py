import os
import json
import random

# test_ids = [['0','2','17','12'], ['1','3','11','13','14'], ['5','9','16','10','19'], ['6','7','18','15'], ['4','8','20','21']]
# test_ids = [['5','2','17','10','15'], ['1','3','20','13'], ['0','8','16','21'], ['6','9','11','12','19'], ['4','7','18','14']]

# 0.807 0.809 0.634
# test_ids = [['18', '4', '3', '20', '16'], ['14', '15', '19', '0', '21'], ['9', '17', '10', '12'], ['1', '11', '7', '13'], ['6', '8', '5', '2']]

#
# test_ids = [['17', '13', '8', '2'], ['3', '5', '7', '16'], ['15', '6', '9', '1'], ['10', '11', '20', '12'], ['0', '4', '18', '21']]
# test_ids = [['14', '13', '8', '3'], ['2', '5', '7', '16'], ['15', '6', '9', '1'], ['10', '11', '20', '12'], ['0', '4', '18', '21']]
# test_ids = [['14', '13', '20', '3'], ['2', '5', '7', '16'], ['15', '6', '9', '1'], ['10', '11', '8', '12'], ['0', '4', '18', '21']] # 0.85 PLCC for STSIM-VGG
# test_ids = [['17', '13', '20', '3'], ['2', '5', '7', '16'], ['15', '6', '9', '1'], ['10', '11', '8', '12'], ['0', '4', '18', '21']] 0.76 PLCC for STSIM-Mf

test_ids = [['15', '5', '7', '1'], ['13', '20', '10', '9'], ['2', '14', '11', '12'], ['3', '4', '6', '18'], ['0', '17', '21', '8']]     # 0.823 PLCC for STSIM-Mf
# import pdb;pdb.set_trace()
# test_ids = [ str(i) for i in range(22)]
# test_ids.remove('14')
# test_ids.remove('19')
# random.shuffle(test_ids)
# test_ids = [test_ids[:4], test_ids[4:8], test_ids[8:12], test_ids[12:16], test_ids[16:]]
# test_ids = [test_ids[:5], test_ids[5:10], test_ids[10:14], test_ids[14:18], test_ids[18:]]
print(test_ids)
for idx in range(5):
    train = []
    valid = []
    test = []
    test_id = test_ids[idx]
    output_dir = '/dataset/new_data_TIP_all/5-fold_cv_v2'
    v=str(idx)
    data_folders = ['/dataset/new_data_TIP_all/train','/dataset/new_data_TIP_all/test','/dataset/new_data_TIP_all/validation']

    clean_names = lambda x: [i for i in x if i[0] != '.']
    for folder in data_folders:
        imgnames = sorted(clean_names(os.listdir(folder)))
        for imgname in imgnames:
            if imgname.split('_')[0] in test_id:
                if folder == '/dataset/new_data_TIP_all/test':  #?
                    test.append(os.path.join(folder, imgname))
            elif 'valid' in folder:
                valid.append(os.path.join(folder, imgname))
            else:
                train.append(os.path.join(folder, imgname))


    # random.shuffle(train)
    # N = len(train)
    # valid = train[int(N*0.9):]
    # train = train[:int(N*0.9)]
    print(test_id)
    print(len(train))   # N textures * D distortions * (10+1) random seeds, the extra one is from testset (texture not in testset)
    print(len(valid))   # N textures * D distortions * 2 random seeds
    print(len(test))    # len(test_id) textures * D distortions * 2 random seeds
    with open(os.path.join(output_dir, 'train_'+v+'.json'), 'w') as json_file:
        json.dump(train, json_file)
    with open(os.path.join(output_dir, 'validation_'+v+'.json'), 'w') as json_file:
        json.dump(valid, json_file)
    with open(os.path.join(output_dir, 'test_'+v+'.json'), 'w') as json_file:
        json.dump(test, json_file)
    # import pdb;pdb.set_trace()
