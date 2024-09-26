import json, os, math
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
from torch.optim.adamw import AdamW
import torchvision.transforms as transforms
from torch import nn, optim
from patch.patchEmbeddings import ViTForClassfication

from trainer.trainer import Trainer
from utils.utils import prepare_data

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


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f'model_{epoch}.pt')
    torch.save(model.state_dict(), cpfile)
    torch.save(model, cpfile)


def load_experiment(experiment_name, checkpoint_name="model_final.pt", base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    # Load the model
    model = ViTForClassfication(config)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies


def visualize_images():
    trainset = torchvision.datasets.ImageFolder(root='./data/train', transform=transforms.ToTensor()) # Cambiar el path por el nuevo 
    classes = trainset.classes

    # Seleccionar 30 muestras aleatoriamente
    indices = torch.randperm(len(trainset))[:30].tolist()
    images = [np.asarray(trainset[i][0].permute(1, 2, 0)) for i in indices]  # Convertir el tensor en array de imagen
    labels = [trainset[i][1] for i in indices]

    # Visualizar las imágenes utilizando matplotlib
    fig = plt.figure(figsize=(10, 10))
    for i in range(30):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]])


@torch.no_grad()
def visualize_attention(model, output=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.

    """
    model.eval()
    # Load random images
    num_images = 30
    # Load raw images without any transformations
    testset_raw = torchvision.datasets.ImageFolder(root='./data/test', transform=None)
    classes = testset_raw.classes
    indices = torch.randperm(len(testset_raw))[:num_images].tolist()
    raw_images = [testset_raw[i][0] for i in indices]
    labels = [testset_raw[i][1] for i in indices]
    
    # Apply transformations to get tensors for model input
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    images = [test_transform(img) for img in raw_images]
    images = torch.stack(images).to(device)
    model = model.to(device)  
    # Alternativa: asegurar que todas las imágenes se conviertan a Tensor
    images = []
    for image in raw_images:
        pil_image = Image.fromarray(image.astype('uint8'))
        tensor_image = test_transform(pil_image)
        images.append(tensor_image)
    
    # Check that images are now tensors
    if all(isinstance(img, torch.Tensor) for img in images):
        print("All images successfully converted to tensors.")
    
    # Apilar las imágenes en un solo tensor
    images = torch.stack(images)
    # Move the images to the device
    images = images.to(device)
    model = model.to(device)
    # Get the attention maps from the last block
    logits, attention_maps = model(images, output_attentions=True)
    # Get the predictions
    predictions = torch.argmax(logits, dim=1)
    # Concatenate the attention maps from all blocks
    attention_maps = torch.cat(attention_maps, dim=1)
    # select only the attention maps of the CLS token
    attention_maps = attention_maps[:, :, 0, 1:]
    # Then average the attention maps of the CLS token over all the heads
    attention_maps = attention_maps.mean(dim=1)
    # Reshape the attention maps to a square
    num_patches = attention_maps.size(-1)
    size = int(math.sqrt(num_patches))
    attention_maps = attention_maps.view(-1, size, size)
    # Resize the map to the size of the image
    attention_maps = attention_maps.unsqueeze(1)
    attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
    attention_maps = attention_maps.squeeze(1)

    # Convert raw images to numpy arrays for visualization
    raw_images_np = [np.array(img.resize((32, 32))) for img in raw_images]

    # Plot the images and the attention maps
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        # Show the ground truth and the prediction
        gt = classes[labels[i]]
        pred = classes[predictions[i]]
        ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
    if output is not None:
        plt.savefig(output)
    plt.show()





# Training Vit

exp_name = "vit_experiment"
batch_size = 32
epochs = 10
lr = 1e-2  #@param {type: "number"}
save_model_every = 0 #@param {type: "integer"}

device = "cuda" if torch.cuda.is_available() else "cpu"

config = {
    "patch_size": 4,  # Input image size: 32x32 -> 8x8 patches
    "hidden_size": 48,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 4 * 48, # 4 * hidden_size
    "hidden_dropout_prob": 0.0,
    "attention_probs_dropout_prob": 0.0,
    "initializer_range": 0.02,
    "image_size": 32,
    "num_classes": 2, # num_classes of CIFAR10
    "num_channels": 3,
    "qkv_bias": True,
    "use_faster_attention": True,
}


def train():
    # Training parameters
    save_model_every_n_epochs = save_model_every 
    print("Saved model")
    # Load the CIFAR10 dataset
    trainloader, testloader, _ = prepare_data(batch_size=batch_size)
    # Create the model, optimizer, loss function and trainer
    model = ViTForClassfication(config)
    print("Configs")
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    print("Training")
    trainer = Trainer(model, optimizer, loss_fn, exp_name, device=device)
    trainer.train(trainloader, testloader, epochs, config, save_model_every_n_epochs=save_model_every_n_epochs)


def main():
    config, model, train_losses, test_losses, accuracies = load_experiment(f"{exp_name}/")
    visualize_attention(model, "atteton_visualize.png")

if __name__ == "__main__":
    main()
