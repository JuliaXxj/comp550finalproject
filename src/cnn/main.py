# -*- coding: utf-8 -*-
"""
@author: 
        - Ardalan Mehrani <ardalan77400@gmail.com>
@brief:
"""
import json
import os
import torch
import lmdb
import argparse
import numpy as np
import torch.nn as nn
from sklearn import metrics

from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from src.datasets import load_datasets
from src.cnn.net import CNN
from src.cnn.lib import CharVectorizer, get_metrics

# multiprocessing workaround
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))


def get_args():
    parser = argparse.ArgumentParser(
        """Character-level Convolutional Networks for Text Classification (https://arxiv.org/pdf/1509.01626.pdf)""")
    parser.add_argument("--dataset", type=str, default='ag_news')
    parser.add_argument("--model_folder", type=str, default="models/cnn/ag_news")
    parser.add_argument("--data_folder", type=str, default="datasets/ag_news/cnn")
    parser.add_argument("--config", type=str, default="small", choices=['small', 'big'])
    parser.add_argument("--maxlen", type=int, default=1014)
    parser.add_argument("--batch_size", type=int, default=128, help="number of example read by the gpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--solver", type=str, default="sgd", help="'agd' or 'adam'")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_halve_interval", type=float, default=10,
                        help="Number of iterations before halving learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Number of iterations before halving learning rate")
    parser.add_argument("--snapshot_interval", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gpuid", type=int, default=-1, help="select gpu (-1 if cpu)")
    parser.add_argument("--nthreads", type=int, default=4)
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--log_msg", type=str, default="")
    parser.add_argument("--no_stress", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--add_space", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--special_character", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--spell_check", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--lemma", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--stem", default=False, type=lambda x: (str(x).lower() == 'true'))
    args = parser.parse_args()
    return args


class TupleLoader(Dataset):

    def __init__(self, path="", maxlen=None):
        self.path = path
        self.maxlen = maxlen

        self.env = lmdb.open(path, max_readers=opt.nthreads, readonly=True, lock=False, readahead=False, meminit=False)
        self.txn = self.env.begin(write=False)

    def __len__(self):
        return self.list_from_bytes(self.txn.get('nsamples'.encode()))[0]

    def list_from_bytes(self, string, dtype=np.int):
        return np.frombuffer(string, dtype=dtype)

    def __getitem__(self, i):
        xtxt = self.list_from_bytes(self.txn.get(('txt-%09d' % i).encode()), int)
        lab = self.list_from_bytes(self.txn.get(('lab-%09d' % i).encode()), int)[0]

        # padding array with 0 to maxlen if needed:
        if self.maxlen:
            xtxt = np.pad(xtxt, (0, self.maxlen - len(xtxt)), 'constant')

        return xtxt, lab


def train(epoch, net, dataset, device, msg="val/test", optimize=False, optimizer=None, scheduler=None, criterion=None):
    net.train() if optimize else net.eval()

    epoch_loss = 0
    nclasses = len(list(net.parameters())[-1])
    cm = np.zeros((nclasses, nclasses), dtype=int)

    with tqdm(total=len(dataset), desc="Epoch {} - {}".format(epoch, msg)) as pbar:
        for iteration, (tx, ty) in enumerate(dataset):

            data = (tx, ty)
            data = [x.to(device) for x in data]

            if optimize:
                optimizer.zero_grad()

            out = net(data[0])
            ty_prob = F.softmax(out, 1)  # probabilites

            # metrics
            y_true = data[1].detach().cpu().numpy()
            y_pred = ty_prob.max(1)[1].cpu().numpy()

            cm += metrics.confusion_matrix(y_true, y_pred, labels=range(nclasses))
            dic_metrics = get_metrics(cm, list_metrics)

            loss = criterion(out, data[1])
            epoch_loss += loss.item()
            dic_metrics['logloss'] = epoch_loss / (iteration + 1)

            if optimize:
                loss.backward()
                optimizer.step()
                dic_metrics['lr'] = optimizer.state_dict()['param_groups'][0]['lr']

            pbar.update(1)
            pbar.set_postfix(dic_metrics)

    if scheduler:
        scheduler.step()
    return dic_metrics


def predict(net, dataset, device, msg="prediction"):
    net.eval()

    y_probs, y_trues = [], []

    for iteration, (tx, ty) in tqdm(enumerate(dataset), total=len(dataset), desc="{}".format(msg)):
        data = (tx, ty)
        data = [x.to(device) for x in data]
        out = net(data[0])
        ty_prob = F.softmax(out, 1)  # probabilites
        y_probs.append(ty_prob.detach().cpu().numpy())
        y_trues.append(data[1].detach().cpu().numpy())

    return np.concatenate(y_probs, 0), np.concatenate(y_trues, 0).reshape(-1, 1)


def save(net, path):
    """
    Saves a model's state and it's embedding dic by piggybacking torch's save function
    """
    # state = {
    #     'epoch': epoch + initial_epoch,
    #     'state_dict': net.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'loss': avg_loss
    # }
    dict_m = net.state_dict()
    torch.save(dict_m, path)


if __name__ == "__main__":

    opt = get_args()
    print("parameters: {}".format(vars(opt)))

    os.makedirs(opt.model_folder, exist_ok=True)
    os.makedirs(opt.data_folder, exist_ok=True)

    dataset = load_datasets(names=[opt.dataset])[0]
    dataset_name = dataset.__class__.__name__
    n_classes = dataset.n_classes
    print("dataset: {}, n_classes: {}".format(dataset_name, n_classes))

    tr_path = "{}/train.lmdb".format(opt.data_folder)
    te_path = "{}/test.lmdb".format(opt.data_folder)

    # check if datasets exis
    all_exist = False if (os.path.exists(tr_path) and os.path.exists(te_path)) else False

    vectorizer = CharVectorizer(maxlen=opt.maxlen, no_stress=opt.no_stress, add_space=opt.add_space,
                                special_character=opt.special_character, spell_check=opt.spell_check, lemma=opt.lemma,
                                stem=opt.stem)
    phoneme = vectorizer.phoneme_dict
    input_dim = len(phoneme) + 1  # dim = n chars + 1
    print(input_dim)

    # assert phoneme is made of unique characters
    if len(phoneme) != len(set(phoneme)):
        dic = {}
        for c in phoneme:
            if c in dic:
                dic[c] += 1
            else:
                dic[c] = 1
        out = [k for k in dic if dic[k] > 1]
        print("{} are duplicates phonemes".format(out))
        raise ValueError

    if not all_exist:
        print("Creating datasets")
        tr_sentences = [txt for txt, lab in tqdm(dataset.load_train_data(), desc="counting train samples")]
        te_sentences = [txt for txt, lab in tqdm(dataset.load_test_data(), desc="counting test samples")]

        n_tr_samples = len(tr_sentences)
        n_te_samples = len(te_sentences)
        del tr_sentences
        del te_sentences

        print("[{}/{}] train/test samples".format(n_tr_samples, n_te_samples))

        ###################
        # transform train #
        ###################
        with lmdb.open(tr_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(
                        tqdm(dataset.load_train_data(), desc="transform train...", total=n_tr_samples)):
                    xtxt = vectorizer.transform([sentence])[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i

                    txn.put(lab_key.encode(), np.array([lab]).tobytes())
                    txn.put(txt_key.encode(), np.array(xtxt).tobytes())

                txn.put('nsamples'.encode(), np.array([i + 1]).tobytes())

        ##################
        # transform test #
        ##################
        with lmdb.open(te_path, map_size=1099511627776) as env:
            with env.begin(write=True) as txn:
                for i, (sentence, label) in enumerate(
                        tqdm(dataset.load_test_data(), desc="transform test...", total=n_te_samples)):
                    xtxt = vectorizer.transform([sentence])[0]
                    lab = label

                    txt_key = 'txt-%09d' % i
                    lab_key = 'lab-%09d' % i

                    txn.put(lab_key.encode(), np.array([lab]).tobytes())
                    txn.put(txt_key.encode(), np.array(xtxt).tobytes())

                txn.put('nsamples'.encode(), np.array([i + 1]).tobytes())


    def tuple_batch(l, maxlen=opt.maxlen, input_dim=input_dim):

        X, Y = zip(*l)
        ty = torch.Tensor(np.array(Y)).long()
        tx = torch.LongTensor(np.array(X)).unsqueeze(2)  # [batch_size, maxlen, 1]

        # one-hot encoding
        batch_t = torch.zeros(len(X), maxlen, input_dim).float()  # [batch_size, maxlen, n caracters + 1]
        batch_t.scatter_(2, tx, 1.0)  # [batch_size, maxlen, n caracters + 1] filled with '1'
        batch_t = batch_t[:, :,
                  1:]  # slice the first vocabulary dimension so that blank and unknown tokens have a vector of zeros (it's a trick)
        return batch_t, ty


    tr_loader = DataLoader(TupleLoader(tr_path, maxlen=opt.maxlen), batch_size=opt.batch_size, shuffle=True,
                           num_workers=opt.nthreads, collate_fn=tuple_batch, pin_memory=True)
    te_loader = DataLoader(TupleLoader(te_path, maxlen=opt.maxlen), batch_size=opt.batch_size, shuffle=False,
                           num_workers=opt.nthreads, collate_fn=tuple_batch, pin_memory=False)

    # select cpu or gpu
    device = torch.device("cuda:{}".format(opt.gpuid) if opt.gpuid >= 0 else "cpu")
    list_metrics = ['accuracy']

    print("Creating model...")
    if opt.config == 'small':
        net = CNN(n_classes=n_classes, input_length=opt.maxlen, input_dim=len(phoneme), n_conv_filters=256,
                  n_fc_neurons=1024)
    elif opt.config == 'big':
        net = CNN(n_classes=n_classes, input_length=opt.maxlen, input_dim=len(phoneme), n_conv_filters=1024,
                  n_fc_neurons=2048)

    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)

    assert opt.solver in ['sgd', 'adam']
    if opt.solver == 'sgd':
        print(" - optimizer: sgd")
        optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=opt.momentum)
    elif opt.solver == 'adam':
        print(" - optimizer: adam")
        optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)

    scheduler = None
    if opt.lr_halve_interval and opt.lr_halve_interval > 0:
        print(" - lr scheduler: {}".format(opt.lr_halve_interval))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_halve_interval, gamma=opt.gamma, last_epoch=-1)

    # log={}
    log_file = './log_{}_{}.txt'.format(opt.dataset, opt.log_msg)
    with open(log_file, 'w') as fp:
        # fp.write("Dataset: {}.".format(opt.dataset))
        fp.write("parameters: {}\n\n".format(vars(opt)))
    for epoch in range(1, opt.epochs + 1):
        epoch_log = train(epoch, net, tr_loader, device, msg="training", optimize=True, optimizer=optimizer,
                          scheduler=scheduler, criterion=criterion)
        val_epoch_log = train(epoch, net, te_loader, device, msg="testing ", criterion=criterion)

        # log[epoch] = epoch_log
        with open(log_file, 'a') as fp:
            fp.write("Epoch: {}. accuracy: {}. logloss: {}. lr: {}\n".format(epoch, epoch_log['accuracy'],
                                                                             epoch_log['logloss'], epoch_log['lr']))
            fp.write("Epoch: {}. val_accuracy: {}. val_logloss: {}.\n\n".format(epoch, val_epoch_log['accuracy'],
                                                                                val_epoch_log['logloss']))
        if (epoch % opt.snapshot_interval == 0) and (epoch > 0):
            path = "{}/model_epoch_{}_{}".format(opt.model_folder, epoch, opt.model_name)
            print("snapshot of model saved as {}".format(path))
            save(net, path=path)

    if opt.epochs > 0:
        # with open('./log_{}.txt'.format(opt.log_msg), 'w') as fp:
        #     json.dump(log, fp, indent=4)
        path = "{}/final_model_epoch_{}_{}".format(opt.model_folder, opt.epochs, opt.model_name)
        print("snapshot of model saved as {}".format(path))
        state = {
            'config': opt.config,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'max_len': opt.maxlen
        }
        dict_m = net.state_dict()
        # torch.save(state, path)
        save(net, path=path)
