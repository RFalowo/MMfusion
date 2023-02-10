from src.lstmae3 import Autoencoder
import torch
import torch.nn as nn
import copy
import numpy as np
from torch.utils.data.dataloader import DataLoader
from src.Dataloader import AISTDataset
import wandb

wandb.init(project="Dance_Forecast", entity="comma")

wandb.config = {
  "learning_rate": 0.001,
  "epochs": 100,
  "batch_size": 8
}

audio_dir = '../all_music_wav'
dance_dir = '../keypoints3d'
train_set, val_set = torch.utils.data.random_split(AISTDataset(dance_dir, audio_dir), [0.8, 0.2])
train_dataloader = DataLoader(train_set,wandb.config["batch_size"], shuffle = False )






seq_len = 10
n_features = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder( n_features,128,64)
model = model.to(device)





def train_model(model, train_dataset, val_dataset, n_epochs, learning_rate):
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  criterion = nn.MSELoss().to(device)
  history = dict(train=[], val=[])
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 10000.0
  for epoch in range(1, n_epochs + 1):
    model = model.train()
    train_losses = []

    for dp in train_dataset:
      X, Y = dp
      for idx in range(len(X)):
        optimizer.zero_grad()
        seq_true = torch.from_numpy(Y[idx]).to(torch.float32).to(device)
        seq_pred = model((torch.from_numpy(X[idx])))
        loss = criterion(seq_pred, seq_true)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        wandb.log({"loss": loss})
        wandb.watch(model)
    val_losses = []
    model = model.eval()
    with torch.no_grad():
      for seq_true in val_dataset:
        seq_true = torch.from_numpy(seq_true).to(device)
        seq_pred = model(seq_true)
        loss = criterion(seq_pred, seq_true)
        val_losses.append(loss.item())
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
      best_model_wts = copy.deepcopy(model.state_dict())
    print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
  model.load_state_dict(best_model_wts)
  return model.eval(), history




model, history = train_model(
  model,
  train_set,
  val_set,
  n_epochs=wandb.config["epochs"],
  learning_rate=wandb.config["learning_rate"]
)
