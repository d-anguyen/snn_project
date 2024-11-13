import torch

import numpy as np
import matplotlib.pyplot as plt


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")



# Counting the number of regions made by an SNN as the mapping from the input to the last spiking layer
# Note that this is not applicable to ANNs
def count_regions(net, num=200):
    print("Counting the number of regions made by a random initialized network of the following architecture ")
    print(net)
    print("with weight matrices")
    print(net.fc1.weight.data)
    print("with " +str(net.num_steps) +" time steps")
    list_outputs = []
    matrix = [] # list of datapoints with color index
    
        
    for x in np.linspace(-5,5,num):
        for y in np.linspace(-5,5,num):
            data= torch.tensor([x,y],dtype=torch.float).to(device)
            spk2_rec, mem2_rec = net(data)
            spk_outputs = spk2_rec.bool().int().detach().numpy()

            #if spk_outputs is not in the list of outputs, then put it there
            if len(list_outputs) == 0:
                list_outputs.append(spk_outputs)    
            if np.any(np.all(spk_outputs == np.array(list_outputs), axis=-1)) == False:
                list_outputs.append(spk_outputs)

            #coloring the input space
            index = ((list_outputs == spk_outputs).all(axis=-1)).nonzero()[0][0]
            matrix.append([x,y,index])
            #plt.plot(x,y, marker = 'o', markersize = 2.7, color = cmap(index) )
    color_matrix = np.array(matrix)
    
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.scatter(color_matrix[:,0], color_matrix[:,1], c=color_matrix[:,2], s=2.8)
    plt.show()
    num_regions = len(list_outputs)
       
    print("We find " + str(num_regions) + " regions according to the spike outputs")
    print(list_outputs)
