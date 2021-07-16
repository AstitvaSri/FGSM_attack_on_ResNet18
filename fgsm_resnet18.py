from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import os
import copy

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

root = 'FGSM'

data_dir = root+'/hymenoptera_data'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
              

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.imshow(inp)
    plt.show()

def attack_model(model, num_images=10, epsilon=0.05, debug=False, visualize=False):
    # load model weights
    model.load_state_dict(torch.load('/home/astitva/Workspace/PyTorch_Projects/TransferLearning/model_weights.pth'))
    model.eval()
    images_so_far = 0
    accuracy = 0.0
    perturbed_samples = []

    for (input, label) in dataloaders['train']: # use 'val' if you want to test on validation set
        if images_so_far==num_images:
            break
        images_so_far += 1

        input = input.to(device)
        label = label.to(device)

        input.requires_grad = True

        output = model(input)
        output = F.softmax(output,dim=1)
        _, init_pred = torch.max(output, 1)

        if init_pred.item() != label.item():
            continue

        # Calculate the loss
        loss = F.nll_loss(output, label)
        model.zero_grad()
        loss.backward()

        # Collect datagrad
        inp_grad = input.grad.data

        # Call FGSM Attack
        perturbed_input = fgsm_attack(input, epsilon, inp_grad)
        output = model(perturbed_input)
        _, pred = torch.max(output, 1)

        if pred.item() == label.item():
            accuracy += 1
        
        else:
            perturbed_samples.append((perturbed_input, label.item(),pred.item()))

        if debug:
            print("Actual:",label.item())
            print("Model Prediction:",init_pred.item())
            print("Fooled Model Prediction:",pred.item())
            print("")

        if visualize:
            plt.axis('off')
            plt.title('Real: {}     Predicted: {}'.format(class_names[label.item()],class_names[pred.item()]))
            inp = perturbed_input.cpu().data[0]
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            plt.show()
    
    final_accuracy = accuracy/images_so_far

    print("Total images: ",images_so_far)

    return final_accuracy, perturbed_samples
    

# feature extractor
model = torchvision.models.resnet18()
# fully connected layer
# (Parameters of newly constructed modules have requires_grad=True by default)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model_conv = model.to(device)

final_accuracy, perturbed_samples = attack_model(model, num_images=200, epsilon=0.07, visualize=False)

print("Accuracy of the model on perturbed samples: ", np.round(final_accuracy*100,2),'%')    


# plotting few perturbed samples on mxn grid
idx = 0
m = 3
n = 3
fig,ax = plt.subplots(m,n)
for i in range(m):
    for j in range(n):
        (pert,gt,pred) = perturbed_samples[idx]
        ax[i][j].axis('off')
        ax[i][j].set_title('Real: {}     Predicted: {}'.format(class_names[gt],class_names[pred]),fontsize=12)
        inp = pert.cpu().data[0]
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        ax[i][j].imshow(inp)
        idx+=1
plt.subplots_adjust(hspace=0.5)
plt.show()
