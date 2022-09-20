from torch import nn

class Factorization(nn.Module):
    def __init__(self, num_out=192, num_hidden=1024, num_prj=192):
        super(Factorization, self).__init__()

        self.fc1 = nn.Linear(num_out, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_out)
        self.bn2 = nn.BatchNorm1d(num_out)

        self.fc_spk = nn.Linear(num_out, num_prj)
        self.fc_dev = nn.Linear(num_out, num_prj)
        self.bn_out = nn.BatchNorm1d(num_out)

        self.relu = nn.ReLU()

    def forward(self, x):

        x_dev = self.fc1(x)
        x_dev = self.bn1(x_dev)
        x_dev = self.relu(x_dev)

        x_dev = self.fc2(x_dev)
        x_dev = self.bn2(x_dev)        
        x_dev = self.relu(x_dev)

        x_spk = x + x_dev

        return x_spk, x_dev

def MainModel(num_out=192):
    model = Factorization(num_out=num_out, num_hidden=1024, num_prj=num_out)
    return model
