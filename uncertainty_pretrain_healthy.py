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
from datetime import datetime
from time import time
import torch.nn.functional as F
from data.sampling import sample_by_label
import json
import random
import pickle
from argparse import ArgumentParser
from models.unet import UNet3D
# models.unet.U Net_alternative ALTERNATIVE HAS SIGMOID ACTIVATION!!!

from models import unet_128i
from models.my_models import MyModel
import numpy as np
from multi_slice_viewer.multi_slice_viewer import seg_viewer, multi_slice_viewer


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
        master_list_ = [d for d in master_list if d['histo_lokacija'] == "healthy"]  # we dont use this selects healthy!!
        print(len(master_list_))
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

        self.writer = SummaryWriter(log_dir="run_test/%s" % datetime.now().strftime(
            "%m%d%Y_%H:%M:%S"))  # TODO: dodaj tle notr folder z enkodiranim časom...

        self.modeln = modeln
        self.model = self._init_model("LeonE___")  # TODO self._init_model(model_name=self.modeln)
        self.model = self.model.to(device)

        self.loss_ce = nn.MSELoss()
        self.loss = self.INN_loss
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=5e-3,
                                         momentum=0.9)  # works better?
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-3)  # weight_decay=5e-3, momentum=0.9)
        self.global_step = 0
        self.val_top_loss = 1e5
        self.train_top_loss = 1e5

    def INN_loss(self, otpt, real, tightness=0.1,
                 mean_loss_weight=0.1, base_loss_scalar = 1):  # 3 channel output - low, mid, max;; NCHWD format
        # real format NHWD (C=1)!
        low = otpt[:, 0]
        mid = otpt[:, 1]
        high = otpt[:, 2]

        mid = torch.unsqueeze(mid, dim=1).to(device)
        zero = torch.zeros_like(real).to(device)
        # tightness = torch.tensor(tightness).to(device)
        # mean_loss_weight = torch.tensor(mean_loss_weight).to(device)
        # a = torch.max(torch.sub(real, high).to(device), other=zero).to(device)

        loss = torch.pow(torch.max(real - high, other=zero).to(device), exponent=2).to(device) + \
               torch.pow(torch.max(low - real, zero), 2)
        loss *= base_loss_scalar
        #print("Lol")
        #print(loss.mean())
        loss += tightness * (high - low)
        #print(loss.mean())
        loss += mean_loss_weight * self.loss_ce(mid.double(), real.double())
        #print(loss.mean())
        loss = loss.mean()
        return loss

    def _init_model(self, model_name):
        if model_name.lower() == "mymodel":
            return MyModel(num_classes=5)
        if model_name.lower() == 'resnet10':
            return resnet10(num_classes=5, activation="softmax")
        if model_name.lower() == 'interval_nn':
            return UNet3D(in_channel=2, n_classes=6)
        if model_name.lower() == "leone___":
            return unet_128i.Simply3DUnet(num_in_channels=1, num_out_channels=3, depth=3, init_feature_size=32, bn=True)
        else:
            return None

    def forward(self, *inputs):
        ct, pet, merged, label, entry = inputs
        inp = torch.Tensor(ct.float())
        inp = inp.to(device)  # no schema error!!
        label = label.to(device)
        target = torch.Tensor(pet.float()).to(device)
        self.model = self.model.to(device)
        otpt = self.model(inp)
        # loss = self.loss_ce(otpt, label)
        # pet = pet.to(device)
        # loss = self.loss_ce(otpt.double(), target.double())
        loss = self.loss(otpt.double(), target.double())
        # self.writer.add_graph(model=self.model, input_to_model=inp, verbose=True)
        return loss, otpt

    def epoch_train(self):
        self.model = self.model.train()
        epoch_loss = 0
        for ct, pet, merged, label, _ in self.train_loader:
            self.optimizer.zero_grad()
            loss, otpt = self.forward(ct, pet, merged, label, _)
            loss.backward()
            self.optimizer.step()
            print(loss)
            epoch_loss += loss.sum().detach().cpu()
        epoch_loss /= len(self.train_list)
        self.writer.add_scalar("train_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def epoch_val(self):
        self.model = self.model.eval()
        epoch_loss = 0
        log_txt = ""
        for ct, pet, merged, label, _ in self.val_loader:
            loss, otpt = self.forward(ct, pet, merged, label, _)
            epoch_loss += loss.sum().detach().cpu()
            log_txt += f'truth: \t{str(label.detach().cpu().numpy())} output: \t{str(otpt.detach().cpu().numpy())}\n'
        epoch_loss /= len(self.val_list)
        # self.writer.add_text("val_", text_string=log_txt, global_step=self.global_step)
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
            loss, otpt = self.forward(ct, pet, merged, label, entry)
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

    args = ArgumentParser()

    args.add_argument("--model_name", type=str, default="interval_nn")  # interval_nn, mymodel, resnet10
    args.add_argument("--val_len", type=int, default=10)  # val set size
    args.add_argument("--batch_size", type=int, default=2)  # batch size
    args.add_argument("--classifications_file", type=str, default="classifications.pkl")  # where to save results

    args.parse_args()
    space = np.logspace(-1.5, -5, num=10)
    print(space)

    run = torch.load("test_run.pth")
    run.model = torch.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/run_test/10122020_16:07:17/best_val.pth")
    for ct, pet, merged, label, _ in run.val_loader:
        print(_['ime'],_['priimek'], _['histo_lokacija'], _['histologija'], _['SGD/MGD'], _['CT_dir'])
        inp = torch.Tensor(ct.float())
        inp = inp.to(device)
        target = torch.Tensor(pet.float()).to(device)
        otpt = run.model(inp)
        loss = run.loss_ce(otpt, target)
        seg_viewer(ct[0, 0].cpu().detach().numpy(),
                   F.relu((pet[0, 0].cpu().detach() - otpt[0, 2].cpu().detach())))
        seg_viewer(ct[0, 0].cpu().detach().numpy(),pet[0, 0].cpu().detach().numpy(), cmap_ = "jet")
    # run = Run(modeln="UNet")
    # ##run.epoch_train()
    # torch.save(run.model, "_test.pth")
    # run.train(25)
    # torch.save(run.model, "_test.pth")
    # torch.save(run, "test_run.pth")

    #run.evaluate_classification()
