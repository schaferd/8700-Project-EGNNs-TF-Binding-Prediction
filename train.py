import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network




class Train():
    
    def __init__(self,param_dict):
        self.num_tokens = num_tokens
        self.dim = dim
        self.depth = depth
        self.num_neatest_neighbors = num_nearest_neighbors
        self.norm_coords = norm_coords
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2

        self.model = None
        self.criterion = None
        self.optimizer = None
        

    def get_trained_model(self,train_loader, test_loader):
        self.model = get_model()
        self.criterion = nn.MSELoss() #CrossEntropy, RMSD
        self.model.apply(self.init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.l2)

        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            train_loss = self.train_iteration(train_loader)
            test_loss = self.test_iteration(test_loader)
            train_losses.append(train_losses)
            test_losses.append(test_losses)

        self.create_loss_plot(train_losses,test_losses)

    def create_loss_plot(self,train_losses,test_losses):
        raise ValueError()

    def init_weights(self,m):
        raise ValueError()

    def train_iteration(train_loader):
        loss = 0
        counter += 1
        for feats, coords, labels in train_loader:
            counter += 1
            features_out,coords_out = self.model(feats, coords)
            train_loss = self.criterion(features_out.to(device),labels.float().to(device))

            train_loss.backward()
            self.optimizer.step()
            loss += train_loss.detach().item()
        loss = loss/counter
        return loss

    def test_iteration(test_loader):
        loss = 0
        counter = 0
        for feats, coords, labels in test_loader:
            counter += 1
            features_out, coords_out = self.model(feats,coords)
            test_loss = self.criterion(features_out.to(device),labels.float().to(device))

            loss += test_loss.item()
        loss = loss/counter 
        return loss
    
    def get_model(self,num_tokens,dim,depth,num_nearest_neighbors,norm_coords):
        net = EGNN_Network(
                num_tokens = num_tokens,
                dim = dim,
                depth = depth,
                num_nearest_neighbors = num_nearest_neighbors,
                norm_coords = norm_coords,
            )
        return net


if __name__ == "__main__":

epochs = 100
batch_size = 16
lr = 1e-4
l2 = 1e-2
num_tokens = 2
depth = 5
num_nearest_neighbors = 20
norm_coords = True


params = {
        "epochs":epochs,
        "batch_size":batch_size,
        "lr":lr,
        "l2":l2,
        "num_tokens":num_tokens,
        "depth":depth,
        "num_nearest_neighbors",:num_nearest_neighbors,
        "norm_coords":norm_coords
        }


