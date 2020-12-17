from uncertainty_pretrain_healthy import *
from tqdm import tqdm

run = torch.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/runs_unceirtanty_model/test_run.pth")
run.model = torch.load("/media/leon/2tbssd/PRESERNOVA/AI_FCH/runs_unceirtanty_model/best_val.pth")

pickles_dir = "/media/leon/2tbssd/PRESERNOVA/AI_FCH/runs_unceirtanty_model/pickles/"
# os.makedirs(pickles_dir, exist_ok=False)

## This makes pickles containing CTs, PETS, as well as INN outputs.

for ct, pet, merged, label, _ in tqdm(run.val_loader):
    wanted_info = ['ime', 'priimek', 'histo_lokacija', 'histologija', 'SGD/MGD', 'CT_dir', 'PET_dir']
    inp = torch.Tensor(ct.float())
    inp = inp.to(device)
    target = torch.Tensor(pet.float()).to(device)
    otpt = run.model(inp)
    loss = run.loss_ce(otpt, target)
    #print(ct.size())

    for i in range(2):
        pkl = dict()
        lst = [_[info][i] for info in wanted_info]

        pkl['info'] = lst
        pkl['CT'] = ct[i, 0].detach().cpu().numpy()
        pkl['PET'] = pet[i, 0].detach().cpu().numpy()
        pkl['output'] = otpt[i].detach().cpu().numpy()
        unique_id = str(hash(str(lst)))[1:]
        with open(os.path.join(pickles_dir, unique_id), "wb") as f:
            pickle.dump(pkl, f)

# copypasted for train_loader:

for ct, pet, merged, label, _ in tqdm(run.train_loader):
    wanted_info = ['ime', 'priimek', 'histo_lokacija', 'histologija', 'SGD/MGD', 'CT_dir', 'PET_dir']
    inp = torch.Tensor(ct.float())
    inp = inp.to(device)
    target = torch.Tensor(pet.float()).to(device)
    otpt = run.model(inp)
    loss = run.loss_ce(otpt, target)
    #print(ct.size())

    for i in range(2):
        pkl = dict()
        lst = [_[info][i] for info in wanted_info]

        pkl['info'] = lst
        pkl['CT'] = ct[i, 0].detach().cpu().numpy()
        pkl['PET'] = pet[i, 0].detach().cpu().numpy()
        pkl['output'] = otpt[i].detach().cpu().numpy()
        unique_id = str(hash(str(lst)))[1:]
        with open(os.path.join(pickles_dir, unique_id), "wb") as f:
            pickle.dump(pkl, f)
