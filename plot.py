import torch

import numpy as np
import matplotlib.pyplot as plt


dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")



# count and visualize the regions made by a SNN at several hidden layers, 
# whose indices are given in layer_indices (starting from 0)
# let s=box_size, we do grid search in the square [-s,s]^2
def count_regions(net, num=200, box_size = 1, layer_indices = [0], path = './images/', trained=False): 
    if not trained:
        print("Counting the number of regions made by a random initialized network of the following architecture ")
    else:
        print("Counting the number of regions made by a trained network of the following architecture ")
    print(net)
    print("with weight matrices", net.fc1.weight.data, " and biases ", net.fc1.bias.data)
    print(" with " +str(net.num_steps) +" time steps")
    
    
    for layer in layer_indices:  
        list_outputs = []
        matrix = [] # list of datapoints with color index
        for x in np.linspace(-box_size,box_size,num):
            for y in np.linspace(-box_size,box_size,num):
                index = None
                data= torch.tensor([x,y],dtype=torch.float).to(device)
                spk2_rec = net(data)[layer]
                #print(spk2_rec,mem2_rec)
                spk_outputs = spk2_rec.bool().int().detach().cpu().numpy()

                if len(list_outputs)==0:
                    list_outputs.append(spk_outputs)

                for i, array in enumerate(list_outputs):
                    if np.array_equal(spk_outputs,array):
                        index = i
                        break
                if index == None:
                    list_outputs.append(spk_outputs)
                    index = len(list_outputs)

                #coloring the input space
                matrix.append([x,y,index])
                #plt.plot(x,y, marker = 'o', markersize = 2.7, color = cmap(index) )
        color_matrix = np.array(matrix)
        
        fig = plt.figure(figsize=(3, 3))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.scatter(color_matrix[:,0], color_matrix[:,1], c=color_matrix[:,2], s=2.5)
        if not trained:
            plt.title('Input space partition before training')
            fig.savefig(path+'regions_layer_' +str(layer+1)+'_pretrained.png', bbox_inches='tight')
        else:
            plt.title('Input space partition after training')
            fig.savefig(path+'regions_layer_' +str(layer+1)+'_trained.png', bbox_inches='tight')
        plt.show()
        num_regions = len(list_outputs)
        plt.close()
        print("We find " + str(num_regions) + " regions according to the spike outputs of layer" + str(layer+1))
        #print(list_outputs)

# Visualize the 2d datapoints from the toy datasets with predicted/target labels
def get_plot(net,dataloader,label='prediction',dataset='relu', path = './images/', trained = False):
    xs = np.empty(0)
    ys = np.empty(0)
    cs = np.empty(0)
    for data, targets in dataloader:
        if dataset=='dynamic':
            xs = np.append(xs, np.sum(data[:,0].numpy(), axis=1 ))
            ys = np.append(ys, np.sum(data[:,1].numpy(), axis=1 ))
        else:
            xs = np.append(xs, data[:,0].numpy())
            ys = np.append(ys, data[:,1].numpy())
        if label == 'prediction':
            spk1_rec, spk2_rec, mem2_rec = net(data.to(torch.float).to(device))
            _, idx = spk2_rec.cpu().sum(dim=0).max(1)
        elif label =='target':
            idx = targets
        cs = np.append(cs, idx.numpy())


    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    color = ['r', 'b']

    plt.scatter(xs[cs==0],ys[cs==0], c = color[0], s = 0.2)
    plt.scatter(xs[cs==1],ys[cs==1], c = color[1], s = 0.2)
    
    #draw the decision boundaries (if possible)
    if dataset == 'linear' or dataset=='dynamic':
        plt.plot([-1,1], [1,-1], 'y', linewidth=0.5) 
    elif dataset == 'relu':
        plt.plot([-1,0,1], [0,0,1], 'y', linewidth=0.5)
    
    if label == 'prediction':
        if trained:
            plt.title('Prediction after training')
            fig.savefig(path+'prediction_'+dataset+'_dataset'+'_after_training.png', bbox_inches='tight')
        else: 
            plt.title('Prediction before training')
            fig.savefig(path+'prediction_'+dataset+'_dataset'+'_before_training.png', bbox_inches='tight')
    else:
        plt.title('Target visualization')    
        fig.savefig(path+'Visualization_'+dataset+'_dataset'+'.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
def get_plot_dynamic(net,dataloader,label='prediction',dataset='dynamic', path = './images/', trained=False):
    xs = np.empty(0)
    ys = np.empty(0)
    cs = np.empty(0)
    for data, targets in dataloader:
        xs = np.append(xs, np.sum(data[:,0].numpy(), axis=1 ))
        ys = np.append(ys, np.sum(data[:,1].numpy(), axis=1 ))
        if label == 'prediction':
            spk1_rec, spk2_rec, mem2_rec = net(data.to(torch.float).to(device))
            _, idx = spk2_rec.cpu().sum(dim=0).max(1)
        elif label =='target':
            idx = targets
        cs = np.append(cs, idx.numpy())


    fig = plt.figure(figsize=(3, 3))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    color = ['r', 'b']

    plt.scatter(xs[cs==0],ys[cs==0], c = color[0], s = 0.2)
    plt.scatter(xs[cs==1],ys[cs==1], c = color[1], s = 0.2)
    if dataset == 'linear':
        plt.plot([-1,1], [1,-1], 'y', linewidth=0.5) 
    elif dataset == 'relu':
        plt.plot([-1,0,1], [0,0,1], 'y', linewidth=0.5)
    
    if label == 'prediction':
        if trained:
            plt.title('Prediction after training')
        else:
            plt.title('Prediction before training')
    else:
        plt.title('Target visualization')
    fig.savefig(path+'_'+label+'dataset_'+dataset+'.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
