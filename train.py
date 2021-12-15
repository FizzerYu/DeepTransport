from args import ArgParser
from utils import set_global_seed
import logging
import os
import numpy as np
from tqdm import tqdm
from dataloader import preProcessData, prepareData, DataSet
from torch.utils.data import DataLoader
import torch
from model import DeepTransport

args = ArgParser()
set_global_seed(args.seed)


os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

datainfo = preProcessData(args)
train = datainfo.train
test  = datainfo.test
logging.info("Preparing train data")
traindata = prepareData(datainfo.train, args.p)
testdata = prepareData(datainfo.test, args.p)
logging.info("Successfull preparing train data")

traindataset = DataSet(traindata, datainfo.radius_upStreamGraph, datainfo.radius_downStreamGraph, args.radius, args.maxlen)
train_dataloader = DataLoader(traindataset, batch_size=args.bs, shuffle=True, num_workers=args.nworks)

testdataset = DataSet(testdata, datainfo.radius_upStreamGraph, datainfo.radius_downStreamGraph, args.radius, args.maxlen)
test_dataloader = DataLoader(testdataset, batch_size=args.bs, shuffle=False, num_workers=args.nworks)

model = DeepTransport()
model = model.cuda()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_all_losses = []
valid_all_losses = []
losses = []
total_y_pred_train = []
total_y_truth_train = []
model.train()

logging.info("Training")

for epoch in range(args.epochs):
    ############################################### train
    losses = []
    model.train()
    with tqdm(total=len(train_dataloader), leave=True, desc=f"Train at Epoch {epoch} ==>") as pbar:
        for idx, ( pred, up, down, label ) in enumerate( train_dataloader ):
            optimizer.zero_grad()
            up, down, pred, label = up.float(), down.float(), pred.float(), label.unsqueeze(-1).float()
            up, down, pred, label = up.cuda(), down.cuda(), pred.cuda(), label.cuda()
            out = model(  up, down, pred )
#         total_y_pred.append(y_pred.cpu().detach().numpy())
#         total_y_truth.append(labels.cpu().detach().numpy())
            loss = criterion(label, out)
            loss.backward()
            optimizer.step()
            pbar.update(1)
            pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss.item())})
            losses.append(loss.item())
        epoch_loss = np.mean( losses )
        train_all_losses.append(epoch_loss)
        
    ################################################ test
    losses = []
    model.eval()
    with torch.no_grad():
        for idx, ( pred, up, down, label ) in enumerate( test_dataloader ):
            up, down, pred, label = up.float(), down.float(), pred.float(), label.unsqueeze(-1).float()
            up, down, pred, label = up.cuda(), down.cuda(), pred.cuda(), label.cuda()
            out = model(  up, down, pred )
            loss = criterion(label, out)
            losses.append(loss.item())
        epoch_loss = np.mean( losses )
        valid_all_losses.append(epoch_loss)
        
    mess = f"Epoch #{epoch}/{args.epochs}\tTrain Loss: {train_all_losses[-1]:.3f}\tValid Loss: {valid_all_losses[-1]:.3f}\n\n"
    tqdm.write(mess)
