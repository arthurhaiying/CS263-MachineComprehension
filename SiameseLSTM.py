from torch import nn
from torch.autograd import Variable
import torch



class LSTMEncoder(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, batch_size, vocab_size):
        super(LSTMEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers)

    def initHiddenCell(self):
        rand_hidden = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        rand_cell = Variable(torch.randn(self.num_layers, self.batch_size, self.hidden_size))
        return rand_hidden, rand_cell

    def forward(self, input, h0, c0):
        input = self.embedding(input)
        input = torch.swapaxes(input, 0, 1) # seqlen axis first
        output, _ = self.lstm(input, (h0, c0))
        return output[-1] # output of last time step


class Siamese_lstm(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers, batch_size, vocab_size):
        super(Siamese_lstm, self).__init__()

        self.encoder = LSTMEncoder(embed_size, hidden_size, num_layers, batch_size, vocab_size)
        self.input_dim = 5 * self.encoder.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim//2),
            nn.ReLU(),
            nn.Linear(self.input_dim//2, 1)
        )
        

    def forward(self, X1, X2):

        # init hidden, cell
        h1, c1 = self.encoder.initHiddenCell()
        h2, c2 = self.encoder.initHiddenCell()

        # input one by one
        o1 = self.encoder(X1, h1, c1)
        o2 = self.encoder(X2, h2, c2)
        # print(f"o1: {o1.size()} o2: {o2.size()}") 
        # utilize these two encoded vectors
        features = torch.cat((o1, o2, torch.abs(o1-o2), o1*o2, (o1+o2)/2.0), -1)
        # features = [v1, v2, |v1-v2|, v1*v2, (v1+v2)/2]
        output = self.classifier(features)
        score = nn.functional.sigmoid(output)
        score = score.squeeze()
        return score
    


