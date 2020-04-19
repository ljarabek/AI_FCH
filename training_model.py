import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import *
from torch.utils.tensorboard import SummaryWriter
from data.dataset import PET_CT_Dataset, master_list, m_list_settings
from torch.utils.data import DataLoader
from resnets.densenet import densenet121
from resnets.resnet import resnet10, resnet50
from time import time
import torch.nn.functional as F
from data.sampling import sample_by_label
import json
import random
import pickle
from models.my_models import MyModel
import numpy as np


# model_used = resnet10(num_classes=5, activation="softmax")
# model_used = MyModel


class Run():
    def __init__(self, modeln="MyModel", val_length=10, batch_size=2, classifications_file="classifications.pkl",
                 learning_rate=3e-2):

        # SAMPLE FOR VALIDATION AND TEST SETS
        self.val_length = val_length  # , self.test_length = 20, 0
        self.batch_size = batch_size
        self.classifications_file = classifications_file
        self.lr = learning_rate

        sample = random.sample(range(0, len(master_list)), k=self.val_length)  # + self.test_length
        # sample  = sample_by_label(master_list, val_size=self.val_length, n_min=2)
        # self.test_list = [e for i, e in enumerate(master_list) if i in sample[self.val_length:]]
        self.val_list = [e for i, e in enumerate(master_list) if i in sample]
        self.train_list = [e for i, e in enumerate(master_list) if i not in sample]

        print("train length: %s \t val length: %s \t test length: " % (
            len(self.train_list), len(self.val_list)))  # , len(self.te)))

        self.train_dataset = PET_CT_Dataset(self.train_list)
        self.val_dataset = PET_CT_Dataset(self.val_list)
        # self.test_dataset = PET_CT_Dataset(self.test_list)

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False,
                                     drop_last=False)
        # self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False)

        self.writer = SummaryWriter()
        self.modeln = modeln
        self.model = self._init_model(model_name=self.modeln)
        self.model = self.model.to(device)

        self.loss_ce = nn.BCELoss()
        # self.loss_ce = nn.BCEWithLogitsLoss()  # naj bi se BCE loss uporabljal z sigmoidom, ne pa z softmax!!

        # oba optimizera sta kr cool :)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-3,
                                         momentum=0.9)  # works better?
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)  # weight_decay=5e-3, momentum=0.9)
        self.global_step = 0
        self.val_top_loss = 1e5
        self.train_top_loss = 1e5

    def _init_model(self, model_name):
        if model_name == "MyModel":
            return MyModel(num_classes=5)
        if model_name == 'resnet10':
            return resnet10(num_classes=5, activation="softmax")
        else:
            return None

    def epoch_train(self):
        self.model = self.model.train()
        epoch_loss = 0
        for ct, pet, merged, label, _ in self.train_loader:
            self.optimizer.zero_grad()
            inp = torch.Tensor(merged.float())
            inp = inp.to(device)  # no schema error!!
            label = label.to(device)
            otpt = self.model(inp)
            loss = self.loss_ce(otpt, label)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.sum().detach().cpu()
        epoch_loss /= len(self.train_list)
        self.writer.add_scalar("train_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def epoch_val(self):
        self.model = self.model.eval()
        epoch_loss = 0
        log_txt = ""
        for ct, pet, merged, label, _ in self.val_loader:
            inp = torch.Tensor(merged.float())
            inp = inp.to(device)
            label = label.to(device)
            otpt = self.model(inp)
            loss = self.loss_ce(otpt, label)
            epoch_loss += loss.sum().detach().cpu()
            # otpt = F.sigmoid(otpt)

            log_txt += f'truth: \t{str(label.detach().cpu().numpy())} output: \t{str(otpt.detach().cpu().numpy())}\n'
        epoch_loss /= len(self.val_list)
        self.writer.add_text("val_", text_string=log_txt, global_step=self.global_step)
        self.writer.add_scalar("val_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def evaluate_classification(self):
        self.model = torch.load(os.path.join(self.writer.log_dir, "best_val.pth"))
        label_list = m_list_settings['encoding'][1]
        try:
            with open(self.classifications_file, "rb") as f:
                classifications = pickle.load(f)
        except:
            classifications = dict()
            classifications['val_loss'] = list()  # se itak požene na koncu, ko je že zoptimiziran..
            classifications['model_version'] = list()
            classifications['truth'] = dict()
            classifications['pred'] = dict()
            classifications['CT_dirs'] = list()
            classifications['PET_dirs'] = list()
            for l in label_list:
                classifications['truth'][l] = list()
                classifications['pred'][l] = list()

        self.model = self.model.eval()
        val_loss = 0
        for ct, pet, merged, label, entry in self.val_loader:
            inp = torch.Tensor(merged.float())
            inp = inp.to(device)
            label = label.to(device)
            otpt = self.model(inp)
            loss = self.loss_ce(otpt, label)
            val_loss += loss.sum().detach().cpu()
            otpt = otpt.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            for b in range(otpt.shape[0]):  # for example in batch
                classifications['model_version'].append(self.writer.log_dir)
                classifications['CT_dirs'].append(entry['CT_dir'])
                classifications['PET_dirs'].append(entry['PET_dir'])
                for il, l in enumerate(label_list):
                    classifications['truth'][l].append(label[b, il])
                    classifications['pred'][l].append(otpt[b, il])
        classifications['val_loss'].append(val_loss)

        with open(self.classifications_file, "wb") as f:
            pickle.dump(classifications, f)

    def train(self, no_epochs=10):

        for i in range(no_epochs):
            t0 = time()
            self.global_step += 1
            tr = self.epoch_train()
            val = self.epoch_val()
            self.writer.add_scalars(main_tag="losses", tag_scalar_dict={'train_loss': tr, "val_loss": val},
                                    global_step=self.global_step)
            if val < self.val_top_loss:
                torch.save(self.model, os.path.join(self.writer.log_dir, "best_val.pth"))
                self.val_top_loss = val
                print("saved_top_model_val")
            if tr < self.train_top_loss:
                torch.save(self.model, os.path.join(self.writer.log_dir, "best_tr.pth"))
                self.train_top_loss = tr
                print("saved_top_model_tr")

            print(f"STEP: {i} TRAINLOSS: {tr} VALLOSS {val} dt {time() - t0}")
        self.writer.close()

from pprint import pprint
if __name__ == "__main__":
    # run = Run()
    # run.train(50)
    # run.evaluate_classification()
    # TODO: data_augmentation!! isti slice v batchu prikazujejo bajno različne anatomije!!!
    # TODO: nared da med trainingom vsakih npr. 10 batchov 5 batchov validira - dogaja se, da je po prvem batchu minimaln val_loss!!

    cross_validation_fold = 12

    space = np.logspace(-1.5, -5, num=10)
    print(space)
    for model in ['MyModel', 'resnet10']:
        for ids, s in enumerate(space):
            d = {
                'modeln': model,
                'val_length': 10,
                'batch_size': 2,
                'classifications_file': "classifications_%s_%s.pkl" % (model,ids),
                'lr': s
            }
            pprint(d)
            for i in range(cross_validation_fold):

                run = Run(modeln=d['modeln'], val_length=d['val_length'], batch_size=d['batch_size'],
                          classifications_file=d['classifications_file'], learning_rate=d['lr'])
                run.train(20)
                run.evaluate_classification()
                with open(os.path.join(run.writer.log_dir, "settings.json"), "w") as f:
                    json.dump(d, f)

    """run = Run()
    #run.train(25)
    run.model = torch.load("./runs/Apr19_06-37-40_leon-desktop/best_val.pth")
    for ct, pet, merged, label, _ in run.val_loader:
        print(_)
        inp = torch.Tensor(merged.float())
        inp = inp.to(device)
        label = label.to(device)
        otpt = run.model(inp)
        loss = run.loss_ce(otpt, label)
        break
    #print(run.epoch_val())
        #run.evaluate_classification()"""
