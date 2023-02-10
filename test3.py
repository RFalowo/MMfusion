from midfusion import Multiencoder
import torch
import  torch.nn as nn
import numpy as np
from torch.utils.data.dataloader import DataLoader
from src.Dataloader import AISTDataset
import wandb

# wandb.init(project="Multimdoal_LSTM", entity="comma")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 8
}


audio_dir = '../all_music_wav'
dance_dir = '../keypoints3d'
train_set, val_set = torch.utils.data.random_split(AISTDataset(dance_dir,audio_dir,seq_length=16), [0.8, 0.2])
train_dataloader = DataLoader(train_set,wandb.config["batch_size"], shuffle = False )

seq_len = 10
'''dis be tuple'''
n_features = (8,20)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
multiLSTM = Multiencoder(64,8,20,64,64,32,32)
multiLSTM = multiLSTM.to(device)
params = multiLSTM.parameters()

def train(model, trainset, valset, n_epochs, lr):
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)
  criterion = nn.MSELoss().to(device)
  history = dict(train=[], val=[])

  for epoch in range(1,n_epochs + 1):
    model = model.train()
    train_losses = []

    for item in trainset:
      ((dX,dY),(mX,mY)) = item
      for idx in range(len(dX)):
        optimizer.zero_grad()
        '''is it just Y or Y[idx]'''
        seq_true = (torch.from_numpy(dY[idx]).to(torch.float32).to(device),torch.from_numpy(mY[idx]).to(torch.float32).to(device))
        seq_pred = model(torch.from_numpy(dX[idx]), torch.from_numpy(mX[idx]))
        loss1 = criterion(seq_pred[0],seq_true[0])
        loss2 = criterion(seq_pred[1], seq_true[1])
        loss = loss1+loss2
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # wandb.log({"train loss": loss, "epoch": epoch})
        # wandb.watch(model)

    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for item in valset:
        ((dX, dY), (mX, mY)) = item
        seq_true = (
        torch.from_numpy(dY).to(torch.float32).to(device), torch.from_numpy(mY).to(torch.float32).to(device))
        seq_pred = model(torch.from_numpy(dX[idx]), torch.from_numpy(mX[idx]))
        try:
          loss = criterion(seq_pred, seq_true)
        except AttributeError as e:
          print('wtf mate')
        val_losses.append(loss.item())
        # wandb.log({"val loss": loss})
        # wandb.watch(model)
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    history['train'].append(train_loss)
    history['val'].append(val_loss)
    torch.save({
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'loss': loss,
    }, "../model.pt")
    if val_loss < best_loss:
      best_loss = val_loss
      # best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

  return model.eval(), history

model = train(
  multiLSTM,
  train_set,
  val_set,
  n_epochs=wandb.config["epochs"],
  lr=wandb.config["learning_rate"]
)