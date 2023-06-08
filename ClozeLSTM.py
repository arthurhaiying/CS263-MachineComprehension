from torch import nn
import torch.nn.functional as Fun
import torch


class ClozeLSTM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_size = config['model']['embed_size']
        self.vocab_size = config['model']['vocab_size']
        self.hidden_size = config['model']['hidden_size']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, 
                            dropout=self.dropout, num_layers=self.num_layers)
        self.latent_size = 2*self.num_layers*self.hidden_size
        self.output_size = self.vocab_size
        middle_size = (self.latent_size+self.output_size)//2
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_size, middle_size),
            nn.ReLU(),
            nn.Linear(middle_size, self.output_size)
        )
        
    def forward(self, X):
        # input: (N, L)
        X = self.embedding(X) # shape (N, L, E)
        X = X.swapaxes(0, 1) # shape (L, N, E)
        O, (H, C) = self.lstm(X) # H, C: (num_layers, N, H)
        H = H.swapaxes(0, 1) # H: (N, num_layers, H)
        C = C.swapaxes(0, 1) # H: (N, num_layers, H)
        H = H.flatten(start_dim=1)
        C = C.flatten(start_dim=1)
        HC = torch.concat((H, C), dim=1) # HC: (N, 2*num_layers*H)
        output = self.mlp(HC) # output: (N, V)
        return output
    
    def likelihood(self, X, YP, YN):
        # input: (N, L)
        # YP: (N,)
        # YN: (N,4)
        output = self.forward(X) # output: (N, V)
        prob = Fun.log_softmax(output, dim=-1) # prob: (N, V)
        batch_size = prob.size[0]

        lld = torch.zeros(batch_size, requires_grad=True)
        for i in range(batch_size):
            lldP = prob[i][YP[i]] 
            lldN = prob[i][YN[i]]
            lld[i] += lldP - torch.sum(lldN)

        lld = -1*torch.mean(lld)
        return lld

            

        

        
        

        
    