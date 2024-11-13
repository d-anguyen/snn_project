import torch
import snntorch.functional as SF
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")
path = './images/'



def train(net, num_steps, train_loader, test_loader, num_epochs = 1000, output='mem'):
    if output=='mem':
        loss = nn.CrossEntropyLoss()
    elif output=='spike':
        loss = SF.ce_count_loss()
    #loss = SF.ce_rate_loss() 
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    train_loss_hist, test_loss_hist = [], []
    # Outer training loop        
    for epoch in tqdm(range(num_epochs)):
        
        # Minibatch training loop
        for data, targets in iter(train_loader):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
    
            # forward pass
            net.train()
            spk1_rec, spk_rec, mem_rec = net(data)
            
            if output=='mem':
            # initialize the loss & sum over time
                loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], targets)
            else: 
            #output=='spike'
                loss_val = loss(spk_rec, targets)
    
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            
        if (epoch%10 == 0) or (epoch == num_epochs-1):
            train_loss, test_loss= print_accuracy(net, train_loader, 
            test_loader, epoch=epoch, num_steps=num_steps,output=output)
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
    return train_loss_hist, test_loss_hist

# Plot Loss
def plot_learning_curve(train_loss_hist, test_loss_hist, path=path):
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(train_loss_hist)
    plt.plot(test_loss_hist)
    plt.title("Loss curves")
    plt.legend(["Train Loss", "Test Loss"])
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    fig.savefig(path+'learning_curve.png', bbox_inches='tight')
    plt.show()
    plt.close()
    
    
    
# compute and show the total train/test loss/accuracy for the whole dataset in a given epoch
def print_accuracy(net, train_loader, test_loader, epoch, num_steps=2, output='mem'):
    if output=='mem':
        loss = nn.CrossEntropyLoss()
    elif output=='spike':
        loss = SF.ce_count_loss()
        
    # stores the total train/test loss in each epoch
    train_loss, test_loss = 0.0, 0.0 
    # store the number of accurate predictions per minibatch
    train_acc, train_total, test_acc, test_total = 0, 0, 0, 0 
    
    with torch.no_grad():
        for data, targets in iter(train_loader):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
            
            net.eval()
            spk1_rec, spk_rec, mem_rec = net(data)
            
            if output=='mem':
                train_loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    train_loss_val += loss(mem_rec[step], targets)
            else:
                train_loss_val = loss(spk_rec, targets)
            
            # compute the total train loss for plotting training curve
            batch_size = targets.size(0)
            train_loss += train_loss_val.item()*batch_size
            train_acc += SF.accuracy_rate(spk_rec, targets) * batch_size
            train_total += batch_size
            
        for test_data, test_targets in iter(test_loader):
            test_data = test_data.to(dtype=torch.float).to(device)
            test_targets = test_targets.to(device)
            
            net.eval()
            test_spk1, test_spk, test_mem = net(test_data)
            
            if output=='mem':
                test_loss_val = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss_val += loss(test_mem[step], test_targets)
            else:
                test_loss_val = loss(test_spk, test_targets)
                
            batch_size = test_targets.size(0)
            test_loss += test_loss_val.item() *batch_size
            test_acc += SF.accuracy_rate(test_spk, test_targets) * batch_size
            test_total += batch_size
        # Print train/test loss/accuracy
        #train_loss_hist.append(train_loss/train_total)
        #test_loss_hist.append(test_loss/test_total)
        
        train_loss /= train_total
        test_loss/= test_total
        print(f"Epoch {epoch}")
        print(f"Train loss: {train_loss:.2f}")
        print(f"Test loss: {test_loss:.2f}")
        print(f"Train accuracy: {train_acc/train_total *100:.2f} %")
        print(f"Test accuracy: {test_acc/test_total *100:.2f} %")
    return train_loss, test_loss


def train_ann(net, train_loader, test_loader, num_epochs = 1000):
    loss = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

    train_loss_hist, test_loss_hist = [], []
    # Outer training loop        
    for epoch in tqdm(range(num_epochs), desc='ANN training epoch'):
        
        # Minibatch training loop
        for data, targets in iter(train_loader):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
    
            # forward pass
            net.train()
            out = net(data)
            loss_val = loss(out, targets)
            
            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            
            
        if (epoch%10 == 0) or (epoch == num_epochs-1):
            train_loss, test_loss= print_accuracy_ann(net, train_loader, test_loader, epoch=epoch)
        train_loss_hist.append(train_loss)
        test_loss_hist.append(test_loss)
    return train_loss_hist, test_loss_hist

def print_accuracy_ann(net, train_loader, test_loader, epoch):
    # stores the total train/test loss in each epoch
    train_loss, test_loss = 0.0, 0.0 
    # store the number of accurate predictions per minibatch
    train_acc, train_total, test_acc, test_total = 0, 0, 0, 0 
    loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, targets in iter(train_loader):
            data = data.to(dtype=torch.float).to(device)
            targets = targets.to(device)
            
            net.eval()
            out = net(data)
            train_loss_val = loss(out, targets)
            
            # compute the total train loss for plotting training curve
            batch_size = targets.size(0)
            train_loss += train_loss_val.item()*batch_size
            _, predicted = torch.max(out.data, 1)
            train_acc += (predicted == targets).sum().item()
            train_total += batch_size
            
        for test_data, test_targets in iter(test_loader):
            test_data = test_data.to(dtype=torch.float).to(device)
            test_targets = test_targets.to(device)
            
            net.eval()
            test_out = net(test_data)
            test_loss_val = loss(test_out, test_targets)
            
                
            batch_size = test_targets.size(0)
            test_loss += test_loss_val.item() *batch_size
            _, predicted = torch.max(test_out.data, 1)
            test_acc += (predicted == test_targets).sum().item()
            test_total += batch_size
        # Print train/test loss/accuracy
        #train_loss_hist.append(train_loss/train_total)
        #test_loss_hist.append(test_loss/test_total)
        
        train_loss /= train_total
        test_loss/= test_total
        print(f"Epoch {epoch}")
        print(f"Train loss: {train_loss:.2f}")
        print(f"Test loss: {test_loss:.2f}")
        print(f"Train accuracy: {train_acc/train_total *100:.2f} %")
        print(f"Test accuracy: {test_acc/test_total *100:.2f} %")
    return train_loss, test_loss