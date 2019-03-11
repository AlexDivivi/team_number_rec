import torch
import torch.nn as nn


class Train:
    def __init__(self, preObj, *, learning_rate=1e-4, epoch=10):
        self.preObj = preObj
        self.device = torch.device("cpu")
        self.dtype = torch.float
        self.X_train, self.X_val, self.y_train, self.y_val = self.make_tenosors(self.preObj, self.device, self.dtype)
        self.model = self.init_model(self.X_train.shape[1], 10)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        self.loss_hist_train, self.loss_hist_val, self.trained_model = self.train_model(
            self.model, epoch, self.optimizer, self.criterion, self.X_train, self.y_train, self.X_val, self.y_val)

    @classmethod
    def make_tenosors(cls, preObj, device, dtype):
        X_train = torch.as_tensor(preObj.splits['X_train'], device=device, dtype=dtype)
        X_val = torch.as_tensor(preObj.splits['X_val'], device=device, dtype=dtype)
        y_train = torch.as_tensor(preObj.splits['y_train'], device=device, dtype=torch.long)
        y_val = torch.as_tensor(preObj.splits['y_val'], device=device, dtype=torch.long)
        return X_train, X_val, y_train, y_val

    @classmethod
    def init_model(cls, D_in, D_out):
        model = torch.nn.Sequential(
            torch.nn.Linear(D_in, D_out))
        return model

    @classmethod
    def train_model(cls, model, epoch, optimizer, criterion, x, y, X_val, y_val):
        loss_hist = []
        loss_hist_val = []
        for t in range(epoch):
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            val_outputs = model(X_val)
            loss_v = criterion(val_outputs, y_val)

            if t % 1000 == 0:
                loss_hist.append(loss.item())
                loss_hist_val.append(loss_v.item())
                print(t, loss.item(), loss_v.item())
        torch.save(model, '../../model/model.pt')
        return loss_hist, loss_hist_val,  model
