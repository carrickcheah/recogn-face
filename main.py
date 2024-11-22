
import numpy as np 
import matplotlib.pyplot as plt
import torch 

from torchvision.datasets import ImageFolder
from torchvision import transforms as T 
# Predict facial expression with pytorch

# Task 1: Download dataset ,add dependency, and configuration
# Bash '
# git clone https://github.com/parth1620/Facial-Expression-Dataset.git
# poetry add git+https://github.com/albumentations-team/albumentations timm opencv-contrib-python

TRAIN_IMG_FOLDER_PATH = './Facial-Expression-Dataset/train'
VALID_IMG_FOLDER_PATH = './Facial-Expression-Dataset/validation'

LR = 0.001
BATCH_SIZE = 32
EPOCHS = 2

DEVICE = 'cuda'
MODEL_NAME = 'efficientnet_b0'

# Task 2: Load dataset

train_augs = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=(-20, +20)),
    T.ToTensor()  # PIL / numpy array -> torch tensor -> (h, w, c) -> (c, h, w)
])

valid_augs = T.Compose([
    T.ToTensor()
])

trainset = ImageFolder(TRAIN_IMG_FOLDER_PATH, transform=train_augs)
validset = ImageFolder(VALID_IMG_FOLDER_PATH, transform=valid_augs)


print(f"Total no. of examples in trainset : {len(trainset)}")
print(f"Total no. of examples in validset : {len(validset)}")

print(trainset.class_to_idx)

image, label = trainset[41]

# # Visualize the image
# plt.imshow(image.permute(1, 2, 0))  # Converts (C, H, W) to (H, W, C) for display
# plt.title(label)
# plt.show()



# Task 3: Load dataset into Batches
from torch.utils.data import DataLoader

trainloader = DataLoader(trainset,
                         batch_size=BATCH_SIZE,
                         shuffle=True)

validloader = DataLoader(validset,
                         batch_size=BATCH_SIZE)

print(f"Total no. of batches in trainloader : {len(trainloader)}")
print(f"Total no. of batches in validloader : {len(validloader)}")

# Retrieve one batch
for images, labels in trainloader:
    break;

print(f"One image batch shape : {images.shape}")
print(f"One label batch shape : {labels.shape}")


# Task 4: Create Model

import timm 
from torch import nn 

# Define the model class
class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        # Use the EfficientNet model from timm
        self.eff_net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=7)

    def forward(self, images, labels=None):
        # Get logits from the model
        logits = self.eff_net(images)
        
        # If labels are provided, calculate loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return logits, loss
        
        return logits

# Instantiate the model and move it to the desired device
model = FaceModel()
model.to(DEVICE);



# Task 5: Create Train and Eval Function
from tqdm import tqdm 

def multiclass_accuracy(y_pred,y_true):
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))


def train_fn(model, dataloader, optimizer, current_epo):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    tk = tqdm(dataloader, desc="EPOCH" + "[TRAIN]" + str(current_epo + 1) + "/" + str(EPOCHS))

    for t, data in enumerate(tk):
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        logits, loss = model(images, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += multiclass_accuracy(logits, labels)
        tk.set_postfix({'loss': '%6f' % float(total_loss / (t + 1)), 'acc': '%6f' % float(total_acc / (t + 1)),})

    return total_loss / len(dataloader), total_acc / len(dataloader)



def eval_fn(model, dataloader, current_epo):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    tk = tqdm(dataloader, desc="EPOCH" + "[VALID]" + str(current_epo + 1) + "/" + str(EPOCHS))

    for t, data in enumerate(tk):
        images, labels = data
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        logits, loss = model(images, labels)

        total_loss += loss.item()
        total_acc += multiclass_accuracy(logits, labels)
        tk.set_postfix({'loss': '%6f' % float(total_loss / (t + 1)), 'acc': '%6f' % float(total_acc / (t + 1)),})

    return total_loss / len(dataloader), total_acc / len(dataloader)



# Task 6:# Create Training Loop

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_valid_loss = np.inf

for i in range(EPOCHS):
    train_loss, train_acc = train_fn(model, trainloader, optimizer, i)
    valid_loss, valid_acc = eval_fn(model, validloader, i)

    if valid_loss < best_valid_loss:
        torch.save(model.state_dict(), 'best-weights.pt')
        print("SAVED-BEST-WEIGHTS")
        best_valid_loss = valid_loss




# Task 7:# Inference

def view_classify(img, ps):
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    ps = ps.data.cpu().numpy().squeeze()
    img = img.numpy().transpose(1,2,0)
   
    fig, (ax1, ax2) = plt.subplots(figsize=(5,9), ncols=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    return None

print(next(model.parameters()).device)
