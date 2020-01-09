import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device_ids = [0]
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from constants import *
from torch.utils.tensorboard import SummaryWriter
from data.dataset import PET_CT, master_list
from torch.utils.data import DataLoader
from resnets.densenet import densenet121
from resnets.resnet import resnet10
from time import time

# model = densenet121(num_init_features=32)#.double()
class Run():
    def __init__(self):

        self.model = resnet10()
        self.model = self.model.to(device)
        self.train_list = master_list[5:-5]
        self.val_list = master_list[:5]
        for a in master_list[-5:]:
            self.val_list.append(a)

        self.train_dataset = PET_CT(self.train_list)
        self.val_dataset = PET_CT(self.val_list)
        self.train_loader = DataLoader(self.train_dataset, batch_size=2, num_workers=4, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=2, num_workers=4, shuffle=False)

        self.writer = SummaryWriter()

        self.loss_ce = nn.BCELoss()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=3e-2, weight_decay=5e-3, momentum=0.9)

        self.global_step = 0

    def epoch_train(self):
        self.model = self.model.train()
        epoch_loss = 0
        for ct, pet, merged, label in self.train_loader:
            #print("lol")
            self.optimizer.zero_grad()
            inp = torch.Tensor(merged.float())
            inp = inp.to(device)  # no schema error!!
            label = label.to(device)
            otpt = self.model(inp)
            loss = self.loss_ce(otpt, label)
            # self.writer.add_scalar("train_loss", loss.sum())
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.sum().detach().cpu()
        self.writer.add_scalar("train_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def epoch_val(self):
        self.model = self.model.eval()
        epoch_loss = 0
        log_txt = ""
        for ct, pet, merged, label in self.val_loader:
            inp = torch.Tensor(merged.float())
            inp = inp.to(device)
            label = label.to(device)
            otpt = self.model(inp)
            loss = self.loss_ce(otpt, label)
            epoch_loss += loss.sum().detach().cpu()

            log_txt += f'truth: \t{str(label.detach().cpu().numpy())} output: \t{str(otpt.detach().cpu().numpy())}\n'
        self.writer.add_text("val_", text_string=log_txt, global_step=self.global_step)
        self.writer.add_scalar("val_loss", epoch_loss, global_step=self.global_step)
        return epoch_loss

    def train(self, no_epochs=10):
        val_top_loss = 1e5
        train_top_loss = 1e5
        for i in range(no_epochs):
            t0 = time()
            self.global_step += 1
            tr = self.epoch_train()
            val = self.epoch_val()

            if val<val_top_loss:
                torch.save(self.model,os.path.join(self.writer.log_dir,"best_val.pth"))
                val_top_loss=val
                print("saved_top_model_val")
            if tr < train_top_loss:
                torch.save(self.model, os.path.join(self.writer.log_dir, "best_tr.pth"))
                train_top_loss=tr
                print("saved_top_model_tr")

            print(f"STEP: {i} TRAINLOSS: {tr} VALLOSS {val} dt {time()-t0}")
        self.writer.close()


if __name__ == "__main__":
    run = Run()
    run.train(50)
