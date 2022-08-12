# %%
from tkinter import N
from typing import Dict
import torch
import torchvision.transforms as transforms
from argparse import Namespace
import os
import pickle
from cassle.methods import METHODS
from cassle.distillers import DISTILLERS
from cassle.args.setup import parse_args_pretrain
from torchvision.datasets import CIFAR100
from cassle.utils.checkpointer import Checkpointer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from cassle.utils.classification_dataloader import prepare_data as prepare_data_classification
from tqdm import tqdm
from cassle.utils.pretrain_dataloader import (
    prepare_dataloader,
    prepare_datasets,
    prepare_multicrop_transform,
    prepare_n_crop_transform,
    prepare_transform,
    split_dataset,
)

class MyCacheFeat():
    def __init__(self):
        if not os.path.exists('./.caches'):
            os.makedirs('./.caches')
        self.inner_cache = {}

    def __getitem__(self, idx):
        if idx not in self.inner_cache:
            retr = pickle.load(open(f'./.caches/{idx}.pkl', 'rb'))
            self.inner_cache[idx] = retr
        return self.inner_cache[idx]

    def __setitem__(self, idx, value):
        self.inner_cache[idx] = value

    def persist(self):
        for idx, value in self.inner_cache.items():
            basedir = os.path.dirname(f'./.caches/{idx}.pkl')
            os.makedirs(basedir, exist_ok=True)
            pickle.dump(value, open(f'./.caches/{idx}.pkl', 'wb'))
        self.inner_cache = {}

    def __contains__(self, idx):
        return idx in self.inner_cache or os.path.exists(f'./.caches/{idx}.pkl')

# def yield_metrics(path):
#     astring = Namespace(
#         pretrained_model=path,
#         method='barlow_twins',
#         batch_size=256,
#         distiller='decorrelative',
#         encoder='resnet18',
#         # ---
#         proj_hidden_dim=2048,
#         output_dim=2048,
#         scale_loss=0.1,
#         # ---
#         dataset='cifar100',
#         data_dir='./data',
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.2,
#         hue=0.1,
#         split_strategy='class',
#         task_idx=4,
#         optimizer='sgd'
#     )
#     astring = [f'--{a}={str(b)}' for a, b in astring.__dict__.items() if b is not None]
#     # print(astring)
#     args = parse_args_pretrain(astring)
#     # print(args)
#     args.multiple_trainloader_mode = "min_size"
#     args.online_eval_batch_size = int(args.batch_size) if args.dataset == "cifar100" else None

#     transform = prepare_transform(
#                     args.dataset, multicrop=args.multicrop, **args.transform_kwargs
#                 )
#     task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)
#     online_eval_transform = transform[-1] if isinstance(transform, list) else transform
#     train_dataset, online_eval_dataset = prepare_datasets(
#         args.dataset,
#         task_transform=task_transform,
#         online_eval_transform=online_eval_transform,
#         data_dir=args.data_dir,
#         train_dir=args.train_dir,
#         no_labels=args.no_labels,
#     )

#     test_transform = transforms.Compose(
#         [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
#     )
#     test_dataset = CIFAR100('./data/cifar100/val/', train=False, transform=test_transform, download=True)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

#     MethodClass = METHODS[args.method]
#     if args.distiller:
#         MethodClass = DISTILLERS[args.distiller](MethodClass)

#     model = MethodClass(**args.__dict__, tasks=5)

#     print(f"Loading previous task checkpoint {args.pretrained_model}...")
#     state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
#     model.load_state_dict(state_dict, strict=False)
    
#     # cache and compute all features

#     need_compute = False
#     if not os.path.exists('.cachefeat.pkl'):
#         need_compute = True
#         cachefeat = {}
#     else:
#         cachefeat = pickle.load(open('.cachefeat.pkl', 'rb'))
#         if args.pretrained_model not in cachefeat:
#             need_compute = True

#     model.cuda()

#     if need_compute:
#         with torch.no_grad():
#             model.eval()
#             print("Computing features...")
#             cachefeat[args.pretrained_model] = {'feats': [], 'z': [], 'y': []}
#             for i, (x, y) in enumerate(tqdm(test_loader)):
#                 x = x.cuda()
#                 y = y.cuda()
#                 r = model(x)
#                 cachefeat[args.pretrained_model]['feats'].append(r['feats'].detach().cpu().numpy())
#                 cachefeat[args.pretrained_model]['z'].append(r['z'].detach().cpu().numpy())
#                 cachefeat[args.pretrained_model]['y'].append(y.detach().cpu().numpy())
#             cachefeat[args.pretrained_model]['feats'] = np.concatenate(cachefeat[args.pretrained_model]['feats'])
#             cachefeat[args.pretrained_model]['z'] = np.concatenate(cachefeat[args.pretrained_model]['z'])
#             cachefeat[args.pretrained_model]['y'] = np.concatenate(cachefeat[args.pretrained_model]['y'])    
#             pickle.dump(cachefeat, open('.cachefeat.pkl', 'wb'))
#             print("Done.")


#     from sklearn.manifold import TSNE
#     import matplotlib.pyplot as plt

#     subsamp = np.random.random(size=cachefeat[args.pretrained_model]['feats'].shape[0]) < 0.5
#     subsamp = subsamp & (cachefeat[args.pretrained_model]['y'] >= 90)

#     for s in ['feats', 'z']:
#         plt.figure()
#         pca = TSNE(n_components=2)
#         dd = pca.fit_transform(cachefeat[args.pretrained_model][s][ subsamp])
        
#         for cl in np.unique(cachefeat[args.pretrained_model]['y'][subsamp]):
#             nu_class_mask = cachefeat[args.pretrained_model]['y'][subsamp] == cl
#             plt.scatter(dd[:, 0][nu_class_mask], 
#                     dd[:, 1][nu_class_mask],
#                     marker='^', color='C'+str(cl), s=10, alpha=1)
            
#     def kmm(feats, classes):
#         from sklearn.cluster import KMeans
#         k = np.unique(classes).shape[0]
#         kmeans = torch.tensor(KMeans(n_clusters=k, random_state=0).fit_predict(feats))
#         ents = []
#         for i in kmeans.unique():
#             _ , conf = np.unique(classes[kmeans == i], return_counts=True)
#             # compute entropy
#             probs = conf / conf.sum()
#             entropy = -(probs * np.log(probs)).sum()
#             ents.append(entropy)
#         return np.mean(ents)

#     allkm = []
#     for s in ['feats', 'z']:
#         for t in range(5):
#             fs = cachefeat[args.pretrained_model][s][cachefeat[args.pretrained_model]['y'] // 20 == t]
#             ys = cachefeat[args.pretrained_model]['y'][cachefeat[args.pretrained_model]['y'] // 20 == t]
#             v = kmm(fs, ys)
#             print(f"{path} KMM {s} - task {t}: {v}")
#             allkm.append(v)
#     plt.figure()
#     plt.bar(range(len(allkm)), allkm)
#     return allkm
# # %%
# akm_cas = yield_metrics(os.popen('cat experiments/stoa_cas/last_checkpoint.txt').read().splitlines()[0])
# akm_ft = yield_metrics(os.popen('cat experiments/stoa_ft/last_checkpoint.txt').read().splitlines()[0])
# # %%
# from matplotlib import pyplot as plt
# plt.bar(range(5), akm_cas[5:])
# plt.bar(range(5,10), akm_ft[5:])
# %%

def allp_metrics(folder):
    paths = sorted(sum([[a + '/' + x for x in c if '.ckpt' in x] for a, b, c in os.walk(folder, topdown=True)], []), key=lambda x: x.split('-task')[1])
    allkm = []
    for jjjj, path in enumerate(tqdm(paths)):
        astring = Namespace(
            pretrained_model=path,
            method='barlow_twins',
            batch_size=256,
            distiller='decorrelative',
            encoder='resnet18',
            # ---
            proj_hidden_dim=2048,
            output_dim=2048,
            scale_loss=0.1,
            # ---
            dataset='cifar100',
            data_dir='./data',
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            split_strategy='class',
            task_idx=4,
            optimizer='sgd'
        )
        astring = [f'--{a}={str(b)}' for a, b in astring.__dict__.items() if b is not None]
        # print(astring)
        args = parse_args_pretrain(astring)
        # print(args)
        args.multiple_trainloader_mode = "min_size"
        args.online_eval_batch_size = int(args.batch_size) if args.dataset == "cifar100" else None

        need_compute = False
        cachefeat = MyCacheFeat()
        if args.pretrained_model not in cachefeat:
            need_compute = True
        if need_compute:
            transform = prepare_transform(
                            args.dataset, multicrop=args.multicrop, **args.transform_kwargs
                        )
            task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)
            online_eval_transform = transform[-1] if isinstance(transform, list) else transform
            train_dataset, online_eval_dataset = prepare_datasets(
                args.dataset,
                task_transform=task_transform,
                online_eval_transform=online_eval_transform,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                no_labels=args.no_labels,
            )

            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
            )
            test_dataset = CIFAR100('./data/cifar100/val/', train=False, transform=test_transform, download=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            MethodClass = METHODS[args.method]
            if args.distiller:
                MethodClass = DISTILLERS[args.distiller](MethodClass)

            model = MethodClass(**args.__dict__, tasks=5)

            if '-ep=499-' not in args.pretrained_model:
                continue

            print(f"Loading previous task checkpoint {args.pretrained_model}...")
            try:
                state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
            except:
                continue
            model.load_state_dict(state_dict, strict=False)
            
            # cache and compute all features

            # model.cuda()

            with torch.no_grad():
                model.eval()
                print("Computing features...")
                cachefeat[args.pretrained_model] = {'feats': [], 'z': [], 'y': []}
                for i, (x, y) in enumerate(tqdm(test_loader)):
                    # x = x.cuda()
                    # y = y.cuda()
                    r = model(x)
                    cachefeat[args.pretrained_model]['feats'].append(r['feats'].detach().cpu().numpy())
                    cachefeat[args.pretrained_model]['z'].append(r['z'].detach().cpu().numpy())
                    cachefeat[args.pretrained_model]['y'].append(y.detach().cpu().numpy())
                cachefeat[args.pretrained_model]['feats'] = np.concatenate(cachefeat[args.pretrained_model]['feats'])
                cachefeat[args.pretrained_model]['z'] = np.concatenate(cachefeat[args.pretrained_model]['z'])
                cachefeat[args.pretrained_model]['y'] = np.concatenate(cachefeat[args.pretrained_model]['y'])    
                cachefeat.persist()
                print("Done.")
    
        def kmm(feats, classes):
            from sklearn.cluster import KMeans
            k = np.unique(classes).shape[0]
            kmeans = torch.tensor(KMeans(n_clusters=k, random_state=0).fit_predict(feats))
            ents = []
            for i in kmeans.unique():
                _ , conf = np.unique(classes[kmeans == i], return_counts=True)
                # compute entropy
                probs = conf / conf.sum()
                entropy = -(probs * np.log(probs)).sum()
                ents.append(entropy)
            return np.mean(ents)

        allkm.append([])
        for s in ['z']:
            for t in tqdm(range(5), leave=False):
                fs = cachefeat[args.pretrained_model][s][cachefeat[args.pretrained_model]['y'] // 20 == t]
                ys = cachefeat[args.pretrained_model]['y'][cachefeat[args.pretrained_model]['y'] // 20 == t]
                v = kmm(fs, ys)
                allkm[-1].append(v)
    return allkm
# %%
# quad_cas = allp_metrics('experiments/stoa_cas')
# quad_ft  = allp_metrics('experiments/stoa_ft')
# quad_joi  = allp_metrics('experiments/stoa_joi')
# quad_r  = allp_metrics('experiments/stoa_r')
# quad_r5k  = allp_metrics('experiments/stoa_r5k')
# # %%
# import seaborn as sns
# alld = np.concatenate([np.array(quad_cas), np.ones((5,1))*float('nan'), np.array(quad_ft)], 1)
# sns.heatmap(alld, cmap='Reds', annot=True, fmt='.2f')
# from matplotlib import pyplot as plt
# plt.figure()
# sns.heatmap(np.array(quad_joi), cmap='Greens', annot=True, fmt='.2f')
# print(quad_r)
# print(quad_r5k)
# %%
### KNN

def allp_knn(folder):
    paths = sorted(sum([[a + '/' + x for x in c if '.ckpt' in x] for a, b, c in os.walk(folder, topdown=True)], []), key=lambda x: x.split('-task')[1])
    allknn = []
    for jjjj, path in enumerate(tqdm(paths)):
        astring = Namespace(
            pretrained_model=path,
            method='barlow_twins',
            batch_size=256,
            distiller='decorrelative',
            encoder='resnet18',
            # ---
            proj_hidden_dim=2048,
            output_dim=2048,
            scale_loss=0.1,
            # ---
            dataset='cifar100',
            data_dir='./data',
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
            split_strategy='class',
            task_idx=4,
            optimizer='sgd'
        )
        astring = [f'--{a}={str(b)}' for a, b in astring.__dict__.items() if b is not None]
        # print(astring)
        args = parse_args_pretrain(astring)
        # print(args)
        args.multiple_trainloader_mode = "min_size"
        args.online_eval_batch_size = int(args.batch_size) if args.dataset == "cifar100" else None

        need_compute = False
        cachefeat = MyCacheFeat()
        if args.pretrained_model not in cachefeat:
            need_compute = True
        if need_compute:
            transform = prepare_transform(
                            args.dataset, multicrop=args.multicrop, **args.transform_kwargs
                        )
            task_transform = prepare_n_crop_transform(transform, num_crops=args.num_crops)
            online_eval_transform = transform[-1] if isinstance(transform, list) else transform
            train_dataset, online_eval_dataset = prepare_datasets(
                args.dataset,
                task_transform=task_transform,
                online_eval_transform=online_eval_transform,
                data_dir=args.data_dir,
                train_dir=args.train_dir,
                no_labels=args.no_labels,
            )

            test_transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
            )
            test_dataset = CIFAR100('./data/cifar100/val/', train=False, transform=test_transform, download=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
            train_dataset = CIFAR100('./data/cifar100/val/', train=True, transform=test_transform, download=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

            MethodClass = METHODS[args.method]
            if args.distiller:
                MethodClass = DISTILLERS[args.distiller](MethodClass)

            model = MethodClass(**args.__dict__, tasks=5)

            print(f"Loading previous task checkpoint {args.pretrained_model}...")
            state_dict = torch.load(args.pretrained_model, map_location="cpu")["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            
            # cache and compute all features

            # model.cuda()

            with torch.no_grad():
                model.eval()
                print("Computing features...")
                cachefeat[args.pretrained_model] = {'feats': [], 'z': [], 'y': [], 'tr_feats': [], 'tr_z': [], 'tr_y': []}
                for i, (x, y) in enumerate(tqdm(test_loader)):
                    # x = x.cuda()
                    # y = y.cuda()
                    r = model(x)
                    cachefeat[args.pretrained_model]['feats'].append(r['feats'].detach().cpu().numpy())
                    cachefeat[args.pretrained_model]['z'].append(r['z'].detach().cpu().numpy())
                    cachefeat[args.pretrained_model]['y'].append(y.detach().cpu().numpy())
                cachefeat[args.pretrained_model]['feats'] = np.concatenate(cachefeat[args.pretrained_model]['feats'])
                cachefeat[args.pretrained_model]['z'] = np.concatenate(cachefeat[args.pretrained_model]['z'])
                cachefeat[args.pretrained_model]['y'] = np.concatenate(cachefeat[args.pretrained_model]['y'])    

                for i, (x, y) in enumerate(tqdm(train_loader)):
                    # x = x.cuda()
                    # y = y.cuda()
                    r = model(x)
                    cachefeat[args.pretrained_model]['tr_feats'].append(r['feats'].detach().cpu().numpy())
                    cachefeat[args.pretrained_model]['tr_z'].append(r['z'].detach().cpu().numpy())
                    cachefeat[args.pretrained_model]['tr_y'].append(y.detach().cpu().numpy())
                cachefeat[args.pretrained_model]['tr_feats'] = np.concatenate(cachefeat[args.pretrained_model]['tr_feats'])
                cachefeat[args.pretrained_model]['tr_z'] = np.concatenate(cachefeat[args.pretrained_model]['tr_z'])
                cachefeat[args.pretrained_model]['tr_y'] = np.concatenate(cachefeat[args.pretrained_model]['tr_y'])    
                cachefeat.persist()
                
                print("Done.")

        from sklearn.neighbors import KNeighborsClassifier
        allknn.append([])
        for s in ['z']:
            for t in tqdm(range(5), leave=False):
                fs = cachefeat[args.pretrained_model]['tr_'+s][cachefeat[args.pretrained_model]['tr_y'] // 20 == t]
                ys = cachefeat[args.pretrained_model]['tr_y'][cachefeat[args.pretrained_model]['tr_y'] // 20 == t]
                knn = KNeighborsClassifier(n_neighbors=5)
                knn.fit(fs, ys)
                tfs = cachefeat[args.pretrained_model][s][cachefeat[args.pretrained_model]['y'] // 20 == t]
                tys = cachefeat[args.pretrained_model]['y'][cachefeat[args.pretrained_model]['y'] // 20 == t]
                allknn[-1].append(knn.score(tfs, tys))
            # class-il
            knn = KNeighborsClassifier(n_neighbors=5)    
            fs = cachefeat[args.pretrained_model]['tr_'+s][cachefeat[args.pretrained_model]['tr_y'] // 20 <= jjjj]
            ys = cachefeat[args.pretrained_model]['tr_y'][cachefeat[args.pretrained_model]['tr_y'] // 20 <= jjjj]
            knn.fit(fs, ys)
            tfs = cachefeat[args.pretrained_model][s][cachefeat[args.pretrained_model]['y'] // 20 <= jjjj]
            tys = cachefeat[args.pretrained_model]['y'][cachefeat[args.pretrained_model]['y'] // 20 <= jjjj]
            allknn[-1].append(knn.score(tfs, tys))
    return allknn
# %%
k_cas = allp_knn('experiments/stoa_cas')
k_ft  = allp_knn('experiments/stoa_ft')
k_joi = allp_knn('experiments/stoa_joi')
k_r  = allp_knn('experiments/stoa_r')
k_r5k  = allp_knn('experiments/stoa_r5k')
# %%
import seaborn as sns
alld = np.concatenate([np.array(k_cas), np.ones((5,1))*float('nan'), np.array(k_ft)], 1)
sns.heatmap(alld, cmap='Greens', annot=True, fmt='.2f')
from matplotlib import pyplot as plt
plt.figure()
sns.heatmap(np.array(k_joi), cmap='Greens', annot=True, fmt='.2f')

print(k_r)
print(k_r5k)
# %%
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

def linear_eval(path, epochs=10):
    cachefeat = MyCacheFeat()
    assert path in cachefeat

    _, feats_shape = cachefeat[path]['feats'].shape
    linear = nn.Linear(feats_shape, 100).cuda()
    optimizer = optim.Adam(linear.parameters(), lr=1e-3)

    class FeatDataset(torch.utils.data.Dataset):
        def __init__(self, feats, y):
            self.feats = feats
            self.y = y
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            return self.feats[idx], self.y[idx]

    feat_tr_dataset = FeatDataset(cachefeat[path]['tr_feats'], cachefeat[path]['tr_y'])
    feat_tr_loader = torch.utils.data.DataLoader(feat_tr_dataset, batch_size=128, shuffle=True, num_workers=4)

    feat_te_dataset = FeatDataset(cachefeat[path]['feats'], cachefeat[path]['y'])
    feat_te_loader = torch.utils.data.DataLoader(feat_te_dataset, batch_size=128, shuffle=False, num_workers=4)

    accuracies = []
    for e in tqdm(range(epochs)):
        linear.train()
        for i, (x, y) in enumerate(feat_tr_loader):
            x = x.cuda()
            y = y.cuda()
            linear.zero_grad()
            r = linear(x)
            loss = F.cross_entropy(r, y)
            loss.backward()
            optimizer.step()
        linear.eval()
        with torch.no_grad():
            correct = 0
            for i, (x, y) in enumerate(feat_te_loader):
                x = x.cuda()
                y = y.cuda()
                r = linear(x)
                correct += (r.argmax(1) == y).sum().item()
            accuracies.append(correct / len(feat_te_dataset))
    return accuracies
# %%
accs = {}
for folder in ['experiments/stoa_cas', 'experiments/stoa_ft', 'experiments/stoa_joi', 'experiments/stoa_r', 'experiments/stoa_r5k']:
    paths = sorted(sum([[a + '/' + x for x in c if '.ckpt' in x] for a, b, c in os.walk(folder, topdown=True)], []), key=lambda x: x.split('-task')[1])
    for i, path in enumerate(paths):
        accs[(folder, i)] = linear_eval(path, epochs=20)

# %%
import matplotlib.pyplot as plt
for i in range(5):
    for j, folder in enumerate(['experiments/stoa_cas', 'experiments/stoa_ft', 'experiments/stoa_joi', 'experiments/stoa_r', 'experiments/stoa_r5k']):
        if (folder, i) in accs:
            plt.scatter(i, np.max(accs[(folder, i)]), color='C%d'%j)
plt.legend(['CAS', 'FT', 'JOI'])
# %%
def eigengaps(path):
    cachefeat = MyCacheFeat()
    assert path in cachefeat

    N, feats_shape = cachefeat[path]['feats'].shape
    adjmat = np.zeros((N, N))
    batch_size = 1024
    for i in tqdm(range(0, N, batch_size)):
        for j in range(0, N, batch_size):
            if i > j:
                continue
            a = cachefeat[path]['feats'][i:i+batch_size]
            b = cachefeat[path]['feats'][j:j+batch_size]
            dist = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            # print(dist.shape, N, i, j, a.shape, b.shape)
            adjmat[i:i+batch_size, j:j+batch_size] = dist
            adjmat[j:j+batch_size, i:i+batch_size] = dist.T
            
    # eigengap
    D = np.diag(np.sum(adjmat, 1))
    L = D - adjmat
    eigvals = np.linalg.eigvalsh(L)
    return eigvals

# %%

eigvals = {}
for folder in [ 'experiments/stoa_joi', 'experiments/stoa_r', 'experiments/stoa_r5k']: #'experiments/stoa_cas', 'experiments/stoa_ft',
    paths = sorted(sum([[a + '/' + x for x in c if '.ckpt' in x] for a, b, c in os.walk(folder, topdown=True)], []), key=lambda x: x.split('-task')[1])
    paths = paths[-1:]
    for i, path in enumerate(paths):
        ee = eigengaps(path)
        eigvals[(folder, i)] = ee
# %%
for e in eigvals:
    ee = eigvals[e]
    plt.plot((ee[2:] - ee[1:-1])[:30], label=e)
    print(e, np.argmax((ee[2:] - ee[1:-1])[:2000]))
plt.legend()
# %%
# Plot clusters
import matplotlib
cachefeat = MyCacheFeat()
subsamp = 0.5
for folder in [ 'experiments/stoa_joi', 'experiments/stoa_r', 'experiments/stoa_r5k']: #'experiments/stoa_cas', 'experiments/stoa_ft',
    paths = sorted(sum([[a + '/' + x for x in c if '.ckpt' in x] for a, b, c in os.walk(folder, topdown=True)], []), key=lambda x: x.split('-task')[1])
    paths = paths[-1:]
    # just last task
    plt.figure()
    dd = cachefeat[path]['feats']
    yy = cachefeat[path]['y']
    dd = dd[yy // 20 == 4]
    yy = yy[yy // 20 == 4]
    # tsne
    from sklearn.manifold import TSNE
    tsne = TSNE(2)
    dd = tsne.fit_transform(dd)
    mask = np.random.random(len(dd)) < subsamp
    dd = dd[mask]
    yy = yy[mask]
    cmap = matplotlib.cm.get_cmap('viridis')
    for i in range(len(dd)):
        plt.scatter(dd[i, 0], dd[i, 1], color=cmap(((yy[i]-80)/20)), s=0.5)

    # dd = cachefeat[path]['feats']
    # yy = cachefeat[path]['y']
    # tsne = TSNE(2)
    # dd = tsne.fit_transform(dd)
    # mask = np.random.random(len(dd)) < subsamp
    # dd = dd[mask]
    # yy = yy[mask]
    # cmap = matplotlib.cm.get_cmap('viridis')
    # for i in range(len(dd)):
    #     plt.scatter(dd[i, 0], dd[i, 1], color=cmap(((yy[i])/100)), s=0.5)

# %%
