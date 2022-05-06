import argparse
import os

from torchvision.utils import save_image
from tqdm import tqdm
from utils.dataset import Dataset

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
        # import pdb;
        # pdb.set_trace()
    return torch.cat(pred)

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
    # data_query = extract_features_VGG('/dataset/MacroTextures3K', data_split='train', batch_size=12, device=device)
    # torch.save(data_query, 'data/MacroTextures3K_VGG.pt')
    data_gallery = extract_features_VGG('/dataset/MacroSyn30000', data_split='test', batch_size=50 ,device=device)
    torch.save(data_gallery, 'data/MacroSyn30000_VGG.pt')
    import pdb;
    pdb.set_trace()
    preds = []
    model = STSIM_M([82 * 3, 10], device=device).to(device)
    model.load_state_dict(torch.load('weights/STSIM_macro_02212022/epoch_0180.pt'))
    for i in tqdm(range(100)):
        pred = model(data_gallery, data_query[i].repeat(data_gallery.shape[0], 1))
        # pred = data_gallery - data_query[i].repeat(data_gallery.shape[0], 1)
        pred = (pred**2).sum(1)
        preds.append(pred)


    for ref_idx in range(len(preds)):
        val_min, idx_min = torch.topk(preds[ref_idx].reshape(-1), 5, largest=False)
        save_res(query, gallery, torch.tensor([ref_idx]).to(device), idx_min, os.path.join(output_dir,str(ref_idx).zfill(4)+'.png'))
    import pdb;
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='weights/STSIM_macro_12282021/config.json', help="path to data config file")
    opt = parser.parse_args()
    print(opt.config)
    test_retrieval(opt.config)
