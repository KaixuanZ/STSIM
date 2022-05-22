import argparse
import os
import json
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
# from utils.dataset_macro import Dataset # for STSIM
from utils.dataset import Dataset # for VGG
from PIL import Image, ImageOps
from metrics.STSIM import *
from metrics.STSIM_VGG import *

def save_res(query, gallery, q_idx, g_idx, output_path):
    imgs = torch.cat([query[q_idx], gallery[g_idx]])

    save_image(imgs, output_path)

@torch.no_grad()
def extract_features_SCF(data_dir, data_split, device, batch_size=100):
    pred = []
    # model = Metric(filter='SCF', device=device)
    testset = Dataset(data_dir=data_dir, data_split=data_split)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    for X_test in tqdm(test_loader):
        if data_split == 'test':
            pred.append(X_test.to(device))
        if data_split == 'train':
            pred.append(X_test.to(device))
    return torch.cat(pred)

@torch.no_grad()
def extract_features_VGG(data_dir, data_split, batch_size, device):
    pred = []
    count = 0
    model = STSIM_VGG(dim=[5900, 10]).to(device)
    testset = Dataset(data_dir=data_dir, data_split=data_split)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    for X_test in tqdm(test_loader):
        X_test = X_test.reshape(-1,3,256,256)
        X_test = X_test.to(device)
        feats = model.forward_once(X_test)
        if data_split == 'train':
            pred.append(feats.reshape(-1,9,5900))
        if data_split == 'test':
            pred.append(feats)
        count += 1

        # if count%2000==0:
        #     torch.save(torch.cat(pred).T, os.path.join('features_generated900K',str(count).zfill(6) + '.pt'))
        #     pred = []

    return torch.cat(pred)

def retrieval(model, gallery, query):
    preds = []
    for i in tqdm(range(query.shape[0])):
        pred = model(gallery, query[i].repeat(gallery.shape[0], 1))
        preds.append(pred)
    preds = torch.stack(preds)
    preds = preds.squeeze(-1)
    return preds.T

def viz(gallary_json, query_json, gallary_idx, query_idx, output_dir = 'tmp'):
    with open(gallary_json) as f:
        gallary_path = json.load(f)
    with open(query_json) as f:
        query_path = json.load(f)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    img_gs = []
    img_qs = []
    count = 0
    for i in range(len(gallary_idx)):
        if count==10:
            grid = make_grid(img_gs+img_qs, 10)
            save_image(grid, os.path.join(output_dir, str(i).zfill(6) + '.png'))
            img_gs = []
            img_qs = []
            count = 0
        img_g = Image.open(gallary_path[gallary_idx[i]])
        img_g = transforms.ToTensor()(img_g)
        C,H,W = img_g.shape
        img_g = img_g[:,H//2-128:H//2+128, W//2-128:W//2+128]
        img_gs.append(img_g)
        img_q = Image.open(query_path[query_idx[i]])
        img_q = transforms.ToTensor()(img_q)
        img_qs.append(img_q)
        count+=1
        # import pdb;
        # pdb.set_trace()
        # save_image(torch.stack([img_g,img_q]), os.path.join(output_dir, str(i).zfill(6)+'.png'))
    grid = make_grid(img_gs + img_qs, count)
    save_image(grid, os.path.join(output_dir, str(i).zfill(6) + '.png'))

@torch.no_grad()
def test_retrieval(config):
    import json
    with open(config) as f:
        config = json.load(f)
        print(config)
    output_dir = 'res_STSIM_SCF_macro_trained'
    # output_dir = 'res_STSIM_VGG_macro'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # testset = Dataset(data_dir='/dataset/MacroTextures500', mode='test')
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=500)
    # query = next(iter(test_loader))
    # testset = Dataset(data_dir='/dataset/MacroSyn5000', mode='test')
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=5000)
    # gallery = next(iter(test_loader))

    # data_query = extract_features_SCF('/dataset/MacroTextures3K', 'train', device, 100)
    # data_gallery = extract_features_SCF('/dataset/MacroSyn30000', 'test', device, 1000)
    # torch.save(data_query, 'data/MacroTextures3K_SCF.pt')
    # torch.save(data_gallery, 'data/MacroSyn30K_SCF.pt')
    # data_query = extract_features_VGG('/dataset/MacroTextures3K', data_split='train', batch_size=5, device=device)
    # torch.save(data_query, 'data/MacroTextures3K_VGG.pt')
    # data_gallery = extract_features_VGG('/dataset/MacroSyn30000', data_split='test', batch_size=60,device=device)
    # torch.save(data_gallery, 'data/MacroSyn30k_VGG.pt')
    # extract_features_VGG('/dataset/generated900K', data_split='test', batch_size=1, device=device)
    # import pdb;
    # pdb.set_trace()

    original = torch.load('/dataset/MacroFeatures/MacroTextures3K_VGG.pt')
    original = original[:,4,:]
    synthesized = torch.load('/dataset/MacroFeatures/MacroSyn30K_VGG.pt')
    # synthesized = synthesized[:,0,:]  # SCF
    synthesized = synthesized[:,:]  # VGG


    model = STSIM_M([5900, 10], device=device).to(device)
    model.load_state_dict(torch.load('weights/STSIM_macro_VGG_05222022/epoch_0070.pt'))
    preds = retrieval(model, original.float(), synthesized.float())

    # gallary_idx2, query_idx2 = torch.where(preds>10)
    # gallary_idx1, query_idx1 = torch.where(preds<3)
    # viz('/dataset/MacroFeatures/MacroTextures3K.json', '/dataset/MacroFeatures/MacroSyn30000.json', gallary_idx1, query_idx1, output_dir='threshold3')
    #
    # gallary_idx1, query_idx1 = torch.where(preds<4)
    # viz('/dataset/MacroFeatures/MacroTextures3K.json', '/dataset/MacroFeatures/MacroSyn30000.json', gallary_idx1[::5], query_idx1[::5], output_dir='threshold4')
    #
    # gallary_idx1, query_idx1 = torch.where(preds<4.5)
    # viz('/dataset/MacroFeatures/MacroTextures3K.json', '/dataset/MacroFeatures/MacroSyn30000.json', gallary_idx1[::25], query_idx1[::25], output_dir='threshold4-5')

    # gallary_idx1, query_idx1 = torch.where(preds<5)
    # viz('/dataset/MacroFeatures/MacroTextures3K.json', '/dataset/MacroFeatures/MacroSyn30000.json', gallary_idx1[::25], query_idx1[::100], output_dir='threshold5')

    import pdb;
    pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='weights/STSIM_macro_12282021/config.json', help="path to data config file")
    opt = parser.parse_args()
    print(opt.config)
    test_retrieval(opt.config)
