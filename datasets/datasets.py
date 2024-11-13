from torch.utils.data.dataset import Dataset
import numpy as np

# generate data
# TODO: think about having some epsilon separation
class Linear_ToyDataset(Dataset):
    def __init__(self, size=1024, seed=3):
        super(Linear_ToyDataset, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.__vals = []
        self.__cs = []
        self.class_names = ['above', 'below']
        for i in range(size):
        # later think about keeping the dataset balanced            
            x,y = 2*self.rng.rand(2)-np.array([1.0,1.0]) # normalize to [-1,1]
            c = int(x+y>0)
            self.__vals.append(np.array([x, y],dtype=float))
            self.__cs.append(c)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        return sample

    def __len__(self):
        return len(self.__cs)

class ReLU_ToyDataset(Dataset):
    def __init__(self, size=1024, seed=3):
        super(ReLU_ToyDataset, self).__init__()

        self.rng = np.random.RandomState(seed)
        self.__vals = []
        self.__cs = []
        self.class_names = ['above', 'below']
        for i in range(size):           
            x,y = 2*self.rng.rand(2)-np.array([1.0,1.0]) #normalize to [-1,1]
            c = int(y>np.maximum(0,x))
            self.__vals.append(np.array([x, y],dtype=float))
            self.__cs.append(c)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        return sample

    def __len__(self):
        return len(self.__cs)

class Dynamic_ToyDataset(Dataset):
    # A toy dataset with T>=2 time steps, each corresponds to a coordinate of a 2d data
    # size: batch_size x num_steps x spatial coordinates
    def __init__(self, size=1024, num_steps=2, seed=3):
        super(Dynamic_ToyDataset, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.__vals = []
        self.__cs = []
        self.class_names = ['above', 'below']
        for i in range(size):
        # later think about keeping the dataset balanced            
            x,y = 2*self.rng.rand(2)-np.array([1.0,1.0]) # normalize to [-1,1]
            c = int(x+y>0)
            data = [[x,0]]
            for i in range(num_steps-2):
                data.append([0,0])
            data.append([0,y])
            self.__vals.append(np.array(data,dtype=float))
            self.__cs.append(c)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        return sample

    def __len__(self):
        return len(self.__cs)
    
class XOR_ToyDataset(Dataset):
    def __init__(self, size=1024, seed=3):
        super(XOR_ToyDataset, self).__init__()
        self.rng = np.random.RandomState(seed)
        self.__vals = []
        self.__cs = []
        self.class_names = ['x', 'o']
        xs = np.array([[1.0,1.0],[-1.0,-1.0], [1.0,-1.0], [-1.0,1.0]])
        cs = [1,1,0,0]
        
        for i in range(size):
        # later think about keeping the dataset balanced            
            index = self.rng.randint(4)    
            x = xs[index]   
            c = cs[index]
            
            #noise = self.rng.rand(2) #in [0,1)
            
            self.__vals.append(x+0.8*(self.rng.rand(2)-1/2))
            self.__cs.append(c)

    def __getitem__(self, index):
        sample = (self.__vals[index].copy(), self.__cs[index])
        return sample

    def __len__(self):
        return len(self.__cs)
    