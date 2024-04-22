import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil

import timeit

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    
    start = timeit.default_timer()
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    stop = timeit.default_timer()
    time = stop - start
    
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5, time

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss, accuracy1, accuracy5, time = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, time]]
    
    
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, accuracy1, accuracy5, time = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1, time]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'time']
    return pd.DataFrame(rows, columns=columns)

# Initialize variables to store max memory usage
max_cpu_usage = 0
max_gpu_allocated = 0
max_gpu_cached = 0

def update_max_memory_usage():
    global max_cpu_usage, max_gpu_allocated, max_gpu_cached
    
    # Update max CPU memory usage
    current_cpu_usage = psutil.virtual_memory().percent
    max_cpu_usage = max(max_cpu_usage, current_cpu_usage)

    # Check if CUDA is available and update GPU memory usage
    if torch.cuda.is_available():
        current_gpu_allocated = torch.cuda.memory_allocated() / 1e9  # in GB
        current_gpu_cached = torch.cuda.memory_reserved() / 1e9      # in GB
        max_gpu_allocated = max(max_gpu_allocated, current_gpu_allocated)
        max_gpu_cached = max(max_gpu_cached, current_gpu_cached)

def print_max_memory_usage():
    print(f"Max CPU memory usage: {max_cpu_usage}% of total")
    if torch.cuda.is_available():
        print(f"Max GPU memory allocated: {max_gpu_allocated:.2f} GB")
        print(f"Max GPU memory cached: {max_gpu_cached:.2f} GB")

def train_eval_loop_memory(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss, accuracy1, accuracy5, time = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1, time]]
    
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, accuracy1, accuracy5, time = eval(model, loss, test_loader, device, verbose)
        
        # Update memory usage after each epoch
        update_max_memory_usage()
        
        row = [train_loss, test_loss, accuracy1, time]
        scheduler.step()
        rows.append(row)

    columns = ['train_loss', 'test_loss', 'top1_accuracy', 'time']
    return pd.DataFrame(rows, columns=columns), {'cpu': max_cpu_usage, 'gpu_allocated': max_gpu_allocated, 'gpu_cached': max_gpu_cached}
