import torch
from torch import nn

class EGNN_Model(nn.Module):
    def __init__(self,input_features,hidden_features,pwm_length):
        super(EGNN_Model,self).__init__()
        self.egnn1 = EGNN(dim=input_features,update_coords=False,num_nearest_neighbors=3)
        self.egnn2 = EGNN(dim=input_features,update_coords=False,num_nearest_neighbors=3)
        self.egnn3 = EGNN(dim=input_features,update_coords=False,num_nearest_neighbors=3)
        self.lin1 = nn.Linear(hidden_features, (1,pwm_length))

    def forward(self, feats, coords):
        feats, coords = self.egnn1(feats, coords)
        feats, coords = self.egnn2(feats, coords)
        feats, coords = self.egnn3(feats, coords)
        pooling_result = torch.mean(feats, dim=0, keepdim=True)
        output = self.lin1(pooling_result)
        return output


