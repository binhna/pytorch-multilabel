import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DNNModel(nn.Module):
    def __init__(self, ninp, nout, dropout=0.5):
        super(DNNModel, self).__init__()

        self.lin = nn.Linear(ninp, 400)
        self.l1 = nn.Linear(400, 50)
        self.lout = nn.Linear(50, nout)

        # self.drop = dropout

    def forward(self, x):
        tmp = F.sigmoid(self.lin(x))
        out = F.sigmoid(self.l1(tmp))
        out = F.softmax(self.lout(out))

        # tmp = F.dropout(F.relu(self.lth(tmp)), self.drop)
        threshold = 0#F.sigmoid(self.lthout(tmp))

        return out, threshold

