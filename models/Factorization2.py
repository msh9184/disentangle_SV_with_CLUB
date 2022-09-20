from torch import nn

class Factorization2(nn.Module):
    def __init__(self, num_out=192, num_hidden=1024, num_prj=192):
        super(Factorization2, self).__init__()

        self.fc1 = nn.Linear(num_out, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden)

        self.fc_spk = nn.Linear(num_hidden, num_prj)
        self.bn_spk = nn.BatchNorm1d(num_prj)

        self.fc_dev = nn.Linear(num_hidden, num_prj)
        self.bn_dev = nn.BatchNorm1d(num_prj)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_spk = self.fc_spk(x)
        x_spk = self.bn_spk(x_spk)

        x_dev = self.fc_dev(x)
        x_dev = self.bn_dev(x_dev)

        return x_spk, x_dev

def MainModel(num_out=192):
    model = Factorization2(num_out=num_out, num_hidden=1024, num_prj=num_out)
    return model
