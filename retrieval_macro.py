import argparse
import os
import json
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from utils.dataset import Dataset
from PIL import Image, ImageOps
from metrics.STSIM import *
from metrics.STSIM_VGG import *

def save_res(query, gallery, q_idx, g_idx, output_path):
    imgs = torch.cat([query[q_idx], gallery[g_idx]])

    save_image(imgs, output_path)

@torch.no_grad()
def extract_features_SCF(data_dir, data_split, device, weights=None):
    pred = []
    model = Metric(filter='SCF', device=device)
    testset = Dataset(data_dir=data_dir, data_split=data_split)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=50)
    for X_test in tqdm(test_loader):
        X_test = X_test.to(device)
        pred.append(model.STSIM_color(X_test))
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
        if count%2000==0:
            torch.save(torch.cat(pred).T, os.path.join('features_generated900K',str(count).zfill(6) + '.pt'))
            pred = []
        # import pdb;
        # pdb.set_trace()
    # return torch.cat(pred).T    # [gallery size, query size]

def retrieval(model, gallery, query):
    preds = []
    for i in tqdm(range(query.shape[0])):
        pred = model(gallery, query[i].repeat(gallery.shape[0], 1))
        preds.append(pred)
    preds = torch.stack(preds)
    preds = preds.squeeze(-1)
    return preds.T

def viz(gallary_json, query_json, gallary_idx, query_idx):
    with open(gallary_json) as f:
        gallary_path = json.load(f)
    with open(query_json) as f:
        query_path = json.load(f)

    if not os.path.isdir('tmp'):
        os.mkdir('tmp')
    img_gs = []
    img_qs = []
    count = 0
    for i in range(len(gallary_idx)):
        if count==10:
            grid = make_grid(img_gs+img_qs, 10)
            save_image(grid, os.path.join('tmp', str(i).zfill(6) + '.png'))
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
    # save_image(torch.stack([img_g,img_q]), os.path.join('tmp', str(i).zfill(6)+'.png'))

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

    # data_query = extract_features_SCF('/dataset/MacroTextures3K', device)
    # data_gallery = extract_features_SCF('/dataset/MacroSyn30000', device)
    # data_query = extract_features_VGG('/dataset/generated900K', data_split='train', batch_size=12, device=device)
    # torch.save(data_query, 'data/MacroTextures3K_VGG.pt')
    # data_gallery = extract_features_VGG('/dataset/generated900K', data_split='test', batch_size=1 ,device=device)
    # torch.save(data_gallery, 'data/MacroSyn900k_VGG.pt')
    extract_features_VGG('/dataset/generated900K', data_split='test', batch_size=1, device=device)
'''
    original = torch.load('data/MacroTextures3K_VGG.pt')
    original = original[:,4,:]
    synthesized = torch.load('data/MacroSyn30000_VGG.pt')

    # model = STSIM_M([82 * 3, 10], device=device).to(device)
    model = STSIM_M([5900, 10], device=device).to(device)
    model.load_state_dict(torch.load('weights/STSIM_macro_05052022/epoch_0180.pt'))
    preds = retrieval(model, original, synthesized)
    gallary_idx1, query_idx1 = torch.where(preds<2)
    gallary_idx2, query_idx2 = torch.where(preds>10)
    import pdb;
    pdb.set_trace()
    viz('data/MacroTextures3K.json', 'data/MacroSyn30000.json', gallary_idx1, query_idx1)
    # viz('data/MacroTextures3K.json', 'data/MacroSyn30000.json', gallary_idx2, query_idx2)
    # for ref_idx in range(len(preds)):
    #     val_min, idx_min = torch.topk(preds[ref_idx].reshape(-1), 5, largest=False)
    #     save_res(query, gallery, torch.tensor([ref_idx]).to(device), idx_min, os.path.join(output_dir,str(ref_idx).zfill(4)+'.png'))
    import pdb;
    pdb.set_trace()
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='weights/STSIM_macro_12282021/config.json', help="path to data config file")
    opt = parser.parse_args()
    print(opt.config)
    test_retrieval(opt.config)
