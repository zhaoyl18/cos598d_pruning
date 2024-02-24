import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)


    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)

    ## Prune ##
    print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    sparsity = 10**(-float(args.compression))
    prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

    
    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                  test_loader, device, args.post_epochs, args.verbose) 

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    prune_result = metrics.summary(model, 
                                   pruner.scores,
                                   metrics.flop(model, input_shape, device),
                                   lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.cm as cm

    weights = []  # List to hold the weights arrays
    layer_names = []  # List to hold the names of the layers

    # Iterate through the model's modules and parameters
    for name, module in model.named_modules():
        # Check if the module is a convolutional layer
        if hasattr(module, 'conv'):
            conv_weights = module.conv.weight.data.cpu().numpy()  # Extract the convolutional weights
            weights.append(conv_weights.flatten())  # Flatten and append the weights
            layer_names.append(f'{name}')  # Append the layer name for weights

        # Check if the module is the fully connected layer by the attribute 'fc'
        if hasattr(module, 'fc'):
            fc_weights = module.fc.weight.data.cpu().numpy()  # Extract the fully connected layer weights
            weights.append(fc_weights.flatten())  # Flatten and append the weights
            layer_names.append(f'fc')  # Append the layer name for weights

    num_layers = len(layer_names)
    color_map = cm.viridis  # Access the colormap directly without using get_cmap
    colors = [color_map(i / num_layers) for i in range(num_layers)]

    # Prepare the figure and axes
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Number of bins for the histogram
    bins = 50
    # Positions for each layer along the y-axis
    layer_positions = np.arange(len(layer_names))

    # Iterate through the layers and weights
    for i, (layer_weights, layer_name) in enumerate(zip(weights, layer_names)):
        # Compute the histogram
        hist, edges = np.histogram(layer_weights, bins=bins)
        # # Normalize the histogram
        # hist = hist / np.max(hist)  # Normalize to the highest peak
        # The x and y coordinates of each bar
        xpos, ypos = np.meshgrid(edges[:-1] + 0.25, [layer_positions[i]], indexing="ij")
        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # The size of each bar
        dx = dy = np.ones_like(zpos) * (edges[1] - edges[0])  # Width of bars
        dz = hist  # Height of bars (normalized histogram values)

        # Plot the bars
        ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors[i], zsort='average', alpha=0.7)

    # Set the y-ticks and y-tick labels to the layer names
    adjusted_layer_positions = layer_positions + dy[0]*2  # Add half the depth to shift labels to center

    # Set the adjusted y-ticks and y-tick labels to the layer names
    ax.set_yticks(adjusted_layer_positions)
    ax.set_yticklabels(layer_names, rotation=-45, ha='left', va='center')


    # Set labels for the other axes
    ax.set_xlabel('Weight values')
    ax.set_ylabel('Layers', labelpad=25) 
    ax.set_zlabel('Count')

    # Adjust the viewing angle if necessary
    ax.view_init(elev=20, azim=-35)

    plt.savefig(f'weight_hist_{args.pruner}.png')