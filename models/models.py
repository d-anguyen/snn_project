import snntorch as snn
import torch
import torch.nn as nn


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")


# Note that the network will output spike outputs of each layers (starting from the first hidden layer)
# and the membrane potential of the last layer. This way it matches with the count_regions function
class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # Initialize layers of widths 2-3
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2, 2, bias=True)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# a simple SNN of architecture 2-3-2
class SNN(nn.Module):
    def __init__(self, num_steps=2, dynamic_input=False):
        super().__init__()
        self.num_steps = num_steps
        self.dynamic_input = dynamic_input
        
        # Initialize layers of widths 2-3-2
        self.fc1 = nn.Linear(2, 2, bias = True)
        self.fc1.weight.data.uniform_(0, 2)
        self.lif1 = snn.Leaky(beta=0.5,learn_threshold=True,learn_beta=True)
        
        self.fc2 = nn.Linear(2, 2, bias = True)
        self.fc2.weight.data.uniform_(-1,3)
        self.lif2 = snn.Leaky(beta=0.5,learn_beta=True,learn_threshold=True)
        
        #self.init_mem1 = 2*torch.rand(2)
        #self.init_mem2 = 2*torch.rand(2)
        
    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        #mem1 = self.init_mem1
        #mem2 = self.init_mem2
        
        # Record all layers
        spk1_rec = []
        mem1_rec = []
        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            if self.dynamic_input:
                cur1 = self.fc1(x[:,step,:])
            else:
                cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        
        return torch.stack(spk1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
