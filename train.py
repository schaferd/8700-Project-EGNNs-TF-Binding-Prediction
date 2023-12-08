import torch
import torch.nn as nn
from model import EGNN_Model

class TFBindPredDataset(Dataset):
    def __init__(self,feats,coords,pwms):
        self.feats = feats
        self.coords = coords
        self.pwms = pwms

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self,idx):
        feats = torch.tensor(self.feats[idx]).float()
        coords = torch.tensor(self.coords[idx]).float()
        pwms = torch.tensor(self.pwms[idx]).float()
        return feats, coords, pwms




class Train():

    def __init__(self,param_dict):
        self.pwm_length = param_dict['pwm_length']
        self.hidden_features = param_dict['hidden_features']
        self.input_features = param_dict['input_features']
        self.epochs = param_dict['epochs']
        self.lr = param_dict['lr']
        self.l2 = param_dict['l2']

        self.model = None
        self.criterion = None
        self.optimizer = None


    def get_trained_model(self,train_loader, test_loader):
        self.model = EGNN_Model(self.input_features,self.hidden_features,self.pwm_length)
        self.criterion = nn.MSELoss() #CrossEntropy, RMSD
        self.model.apply(self.init_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.lr,weight_decay=self.l2)

        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            train_loss = self.train_iteration(train_loader)
            test_loss = self.test_iteration(test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

        self.create_loss_plot(train_losses,test_losses)
        pd.to_pickle(test_pred, "test_pred"+str(fold)+".pkl")
        pd.to_pickle(test_labels, "test_labels"+str(fold)+".pkl")
        pd.to_pickle(train_loss,"train_loss"+str(fold)+".pkl")
        pd.to_pickle(test_loss,"test_loss"+str(fold)+".pkl")

    def create_loss_plot(self,train_losses,test_losses):
        fig,ax = plt.subplots()
        ax.plot(np.arange(self.epochs),train_losses)
        ax.plot(np.arange(self.epochs),test_losses)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Epochs")
        fig.savefig(name)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def train_iteration(train_loader):
        loss = 0
        counter = 0
        for feats, coords, labels in train_loader:
            counter += 1
            pred_pwm = self.model(feats, coords)
            train_loss = self.criterion(pred_pwm,labels)

            train_loss.backward()
            self.optimizer.step()
            loss += train_loss.detach().item()
        loss = loss/counter
        return loss

    def test_iteration(test_loader):
        loss = 0
        counter = 0
        test_pred = []
        test_labels = []
        for feats, coords, labels in test_loader:
            counter += 1
            pred_pwm = self.model(feats,coords)
            test_loss = self.criterion(pred_pwm,labels)
            test_labels = test_labels + labels.tolist()
            test_pred = test_pred + out.tolist()

            loss += test_loss.item()
        loss = loss/counter
        return loss,test_pred,test_labels

    def get_model(self,num_tokens,dim,depth,num_nearest_neighbors,norm_coords):
        net = EGNN_Network(
                num_tokens = num_tokens,
                dim = dim,
                depth = depth,
                num_nearest_neighbors = num_nearest_neighbors,
                norm_coords = norm_coords,
            )
        return net

    def cross_validation(self,folds):
        for fold in range(folds):
            train_feats, train_coords, train_pwms, test_feats, test_coords, test_pwms = self.train_test_split_multiple()

            train_dataset = TFBindPredDataset(train_feats,train_coords,train_pwms)
            test_dataset = TFBindPredDataset(test_feats,test_coords,test_pwms)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size,shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size,shuffle=False)

            self.get_trained_model(train_loader,test_loader,fold)

    def train_test_split_multiple(self, test_size=0.3, random_state=None):
        feats = self.data['feats']
        coords = self.data['coords']
        pwms = self.data['pwms']

        if not 0.0 < test_size < 1.0:
            raise ValueError(f"test_size should be between 0.0 and 1.0, but got {test_size}")

        feats = np.array(feats)
        coords = np.array(coords)
        pwms = np.array(pwms)

        num_samples = len(pwms)
        indices = np.arange(num_samples)

        if random_state is not None:
            np.random.seed(random_state)

        np.random.shuffle(indices)
        test_size = int(test_size * num_samples)
        test_indices, train_indices = indices[:test_size], indices[test_size:]

        train_feats = feats[train_indices]
        train_coords = coords[train_indices]
        train_pwms = pwms[train_indices]

        test_feats = feats[test_indices]
        test_coords = coords[test_indices]
        test_pwms = pwms[test_indices]

        return train_feats, train_coords, train_pwms, test_feats, test_coords, test_pwms

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




