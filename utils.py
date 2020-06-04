import torch
from torch.utils.data import Dataset

## train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

## test data    
class testData(Dataset):
    
    def __init__(self, X_data):
        self.X_data = X_data
        
    def __getitem__(self, index):
        return self.X_data[index]
        
    def __len__ (self):
        return len(self.X_data)


def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train(model, criterion, optimizer, train_loader, params):

    loss_data = []
    acc_data = []

    # pos_weight = torch.Tensor([(549 / 342)]) #handle the unbalanced data
    # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # optimizer = optim.Adam(model.parameters())

    model.train()

    for e in range(1, params['epochs']+1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch, y_batch
            optimizer.zero_grad()
            
            y_pred = model(X_batch)
            
            loss = criterion(y_pred, y_batch)
            acc = binary_acc(y_pred, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
        loss_data.append(epoch_loss/len(train_loader))
        acc_data.append(epoch_acc/len(train_loader))
            

        print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f}'
              + f' | Acc: {epoch_acc/len(train_loader):.3f}')

    return model, loss_data, acc_data

def gen_predictions(model, test_loader):
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch
            y_test_pred = model(X_batch)
            y_test_pred = torch.sigmoid(y_test_pred)
            y_pred_tag = torch.round(y_test_pred)
            # y_pred_list.append(y_pred_tag.cpu().numpy())
            y_pred_list.append(y_pred_tag.numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    return y_pred_list