import os
import json
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


def prepare_data(batch_size=4, num_workers=2, train_sample_size=None, test_sample_size=None):
    # Define transformations for the training set
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize images to 32x32
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(
            (32, 32),
            scale=(0.8, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize images
    ])

    # Load the training dataset
    trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=train_transform)

    if train_sample_size is not None:
        # Randomly sample a subset of the training set
        indices = torch.randperm(len(trainset))[:train_sample_size].tolist()
        trainset = torch.utils.data.Subset(trainset, indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Define transformations for the test set
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load the testing dataset
    testset = torchvision.datasets.ImageFolder(root='./data/test', transform=test_transform)

    if test_sample_size is not None:
        # Randomly sample a subset of the test set
        indices = torch.randperm(len(testset))[:test_sample_size].tolist()
        testset = torch.utils.data.Subset(testset, indices)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Get the class names
    classes = ['cats', 'dogs']

    return trainloader, testloader, classes

def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)

def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    # Save the model
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)
