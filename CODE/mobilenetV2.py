from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvBNActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                             activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(
        self,
        plant_types: List[str],
        disease_names: List[str],
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
    ) -> None:
        super(MobileNetV2, self).__init__()

        self.num_plant_types = len(plant_types)
        self.num_diseases = len(disease_names)

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = self._make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = self._make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvBNActivation(3, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, 1280),
            nn.ReLU6(inplace=True),
            nn.Linear(1280, self.num_plant_types + self.num_diseases),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _make_divisible(self, v: float, divisor: int, min_value: Optional[int] = None) -> int:
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        # Split the output into plant types and diseases
        plant_output, disease_output = torch.split(x, [self.num_plant_types, self.num_diseases], dim=1)
        return plant_output, disease_output

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self._forward_impl(x)



import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score
import os
from PIL import Image
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, hamming_loss, mean_squared_error, roc_curve, auc, accuracy_score
from scipy.special import softmax
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, hamming_loss, mean_squared_error, roc_curve, auc, accuracy_score
import numpy as np
import seaborn as sns
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Constants and configurations
IMAGE_SIZE = 224
batch_size = 64
num_epochs = 100
patience = 101
num_workers = 4
train_dir = "/Akash/ENV/Data_TG/train"
val_dir = "/Akash/ENV/Data_TG/val"



train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=IMAGE_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
   # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
   # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.plant_types, self.diseases = self._find_classes()
        self.plant_to_idx = {plant: idx for idx, plant in enumerate(self.plant_types)}
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(self.diseases)}
        self.samples = self._make_dataset()
        self.transform = transform
        self._analyze_dataset()


    def _analyze_dataset(self):
        plant_counts = {plant: 0 for plant in self.plant_types}
        disease_counts = {disease: 0 for disease in self.diseases}
        for _, plant_idx, disease_idx in self.samples:
            plant_counts[self.plant_types[plant_idx]] += 1
            disease_counts[self.diseases[disease_idx]] += 1
        print("Plant distribution:", plant_counts)
        print("Disease distribution:", disease_counts)

    def _find_classes(self):
        plant_types = set()
        diseases = set()
        for plant_type in sorted(os.listdir(self.data_dir)):
            plant_type_dir = os.path.join(self.data_dir, plant_type)
            if not os.path.isdir(plant_type_dir):
                continue
            plant_types.add(plant_type)
            for disease in sorted(os.listdir(plant_type_dir)):
                diseases.add(disease)
        return sorted(plant_types), sorted(diseases)

    def _make_dataset(self):
        samples = []
        for plant_type in self.plant_types:
            plant_type_dir = os.path.join(self.data_dir, plant_type)
            for disease in self.diseases:
                disease_dir = os.path.join(plant_type_dir, disease)
                if not os.path.isdir(disease_dir):
                    continue
                for root, _, fnames in sorted(os.walk(disease_dir)):
                    for fname in sorted(fnames):
                        path = os.path.join(root, fname)
                        samples.append((path, self.plant_to_idx[plant_type], self.disease_to_idx[disease]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, plant_idx, disease_idx = self.samples[idx]
        image = Image.open(path) #.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, plant_idx, disease_idx






def save_checkpoint(model, optimizer, epoch, plant_loss, disease_loss, plant_acc, disease_acc, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filename = f'checkpoint_epoch_{epoch}.pth'
    filepath = os.path.join(directory, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'plant_loss': plant_loss,
        'disease_loss': disease_loss,
        'plant_acc': plant_acc,
        'disease_acc': disease_acc
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")



from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def calculate_metrics(y_true, y_pred, y_score):
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize the output for multi-class ROC
    y_true_bin = label_binarize(y_true, classes=classes)
    if n_classes == 2:
        y_true_bin = np.hstack((1-y_true_bin, y_true_bin))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Compute precision-recall curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])

    # Compute micro-average precision-recall curve
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_bin.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin, y_score, average="micro")

    # Other metrics
    mse = mean_squared_error(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    hamming = hamming_loss(y_true, y_pred)

    # Gini Coefficient
    gini = 2 * roc_auc["micro"] - 1

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'precision_recall': {'precision': precision, 'recall': recall, 'average_precision': average_precision},
        'mse': mse,
        'mcc': mcc,
        'hamming_loss': hamming,
        'gini': gini
    }




import matplotlib.pyplot as plt

def log_metrics_to_tensorboard(writer, prefix, metrics, epoch):
    for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'mse', 'mcc', 'hamming_loss', 'gini']:
        fig, ax = plt.subplots()
        ax.plot(epoch, metrics[metric_name], 'bo')  # 'bo' for blue dot
        ax.set_xlim(left=0)  # Start x-axis at 0
        ax.set_ylim(bottom=0)  # Start y-axis at 0
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{prefix} {metric_name.capitalize()}')
        writer.add_figure(f'{prefix}/{metric_name}', fig, epoch)
        plt.close(fig)

    # Handle AUC separately as it's nested
    fig, ax = plt.subplots()
    ax.plot(epoch, metrics['roc_auc']['micro'], 'bo')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0, top=1)  # AUC is always between 0 and 1
    ax.set_xlabel('Epoch')
    ax.set_ylabel('AUC')
    ax.set_title(f'{prefix} AUC (micro)')
    writer.add_figure(f'{prefix}/AUC', fig, epoch)
    plt.close(fig)




def log_loss_figures_to_tensorboard(writer, losses, epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training Plant Loss
    train_plant_epochs = range(1, len(losses['train_plant']) + 1)
    ax1.plot(train_plant_epochs, losses['train_plant'], 'bo-', label='Train')

    # Validation Plant Loss
    val_plant_epochs = range(1, len(losses['val_plant']) + 1)
    ax1.plot(val_plant_epochs, losses['val_plant'], 'ro-', label='Validation')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Plant Loss')
    ax1.set_title('Plant Loss')
    ax1.legend()

    # Training Disease Loss
    train_disease_epochs = range(1, len(losses['train_disease']) + 1)
    ax2.plot(train_disease_epochs, losses['train_disease'], 'bo-', label='Train')

    # Validation Disease Loss
    val_disease_epochs = range(1, len(losses['val_disease']) + 1)
    ax2.plot(val_disease_epochs, losses['val_disease'], 'ro-', label='Validation')

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Disease Loss')
    ax2.set_title('Disease Loss')
    ax2.legend()

    plt.tight_layout()
    writer.add_figure('Losses', fig, global_step=epoch)
    plt.close(fig)


losses = {
    'train_plant': [],
    'train_disease': [],
    'val_plant': [],
    'val_disease': []
}



def plot_roc_curves(fpr, tpr, roc_auc, class_names, title, epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'micro-average ROC curve (area = {roc_auc["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'macro-average ROC curve (area = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)

    for i, color in zip(range(len(class_names)), plt.cm.rainbow(np.linspace(0, 1, len(class_names)))):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (area = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title} (Epoch {epoch})')
    plt.legend(loc="lower right", fontsize='small')
    return plt.gcf()

def plot_precision_recall_curves(precision, recall, average_precision, class_names, title, epoch):
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(len(class_names)), plt.cm.rainbow(np.linspace(0, 1, len(class_names)))):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Precision-Recall curve of class {class_names[i]} (AP = {average_precision[i]:.2f})')

    plt.plot(recall["micro"], precision["micro"],
             label=f'micro-average Precision-Recall curve (AP = {average_precision["micro"]:.2f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{title} (Epoch {epoch})')
    plt.legend(loc="lower left", fontsize='small')
    return plt.gcf()



def plot_confusion_matrix(cm, class_names, title, epoch):
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm, dtype=str)
    
    n_classes = len(class_names)
    for i in range(n_classes):
        for j in range(n_classes):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = 'TP'
                annot[i, j] = f'{c}\n{p:.1f}%\n{s}'
            elif j != i:
                s = 'FP' if i == 0 else 'FN'
                annot[i, j] = f'{c}\n{p:.1f}%\n{s}'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'{title} (Epoch {epoch})')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.text(1.05, 1, 'TP: True Positive\nFP: False Positive\nFN: False Negative', 
            transform=ax.transAxes, verticalalignment='top')
    
    plt.tight_layout()
    return fig


# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create datasets and dataloaders
train_dataset = CustomDataset(train_dir, transform=train_transform)
val_dataset = CustomDataset(val_dir, transform=val_transform)

plant_types = train_dataset.plant_types
#diseases = train_dataset.diseases
disease_names = train_dataset.diseases



# Calculate weights for weighted sampling
plant_counts = [sum(1 for _, p, _ in train_dataset.samples if p == i) for i in range(len(train_dataset.plant_types))]
disease_counts = [sum(1 for _, _, d in train_dataset.samples if d == i) for i in range(len(train_dataset.diseases))]

plant_weights = 1. / torch.tensor(plant_counts, dtype=torch.float)
disease_weights = 1. / torch.tensor(disease_counts, dtype=torch.float)

sample_weights = [plant_weights[p] * disease_weights[d] for _, p, d in train_dataset.samples]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
import torch.optim.lr_scheduler as lr_scheduler



# Create an instance of the model and move it to the device
model = MobileNetV2(plant_types=plant_types, disease_names=disease_names)
model = model.to(device)




# Define loss functions, optimizer, and scheduler
#criterion_plant = nn.CrossEntropyLoss(weight=plant_weights.to(device))
#criterion_disease = nn.CrossEntropyLoss(weight=disease_weights.to(device))  # Weighted loss for diseases
from torch.nn.functional import cross_entropy

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, weight=None, alpha=1, gamma=2, ce_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss
    




criterion_plant = CombinedLoss(weight=plant_weights.to(device), ce_weight=0.5, focal_weight=0.5)
criterion_disease = CombinedLoss(weight=disease_weights.to(device), ce_weight=0.5, focal_weight=0.5)
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Assuming 'model' is defined somewhere before the training loop
# Add this part to print the number of parameters in the model
print(f'Total number of trainable parameters: {count_parameters(model):,}')



# Set up TensorBoard
writer = SummaryWriter('runs/plant_disease_experiment')

# Training loop
best_val_acc = 0.0
counter = 0

for epoch in range(num_epochs):
    model.train()
    running_plant_loss = 0.0
    running_disease_loss = 0.0
    total_plant_predictions = []
    total_plant_targets = []
    total_disease_predictions = []
    total_disease_targets = []
    total_plant_outputs = []
    total_disease_outputs = []
    total_plant_scores = []
    total_disease_scores = []

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, Initial LR: {current_lr:.6f}')

    for i, (inputs, plant_labels, disease_labels) in enumerate(train_dataloader):
        inputs = inputs.to(device)
        plant_labels = plant_labels.to(device)
        disease_labels = disease_labels.to(device)

        optimizer.zero_grad()

        plant_outputs, disease_outputs = model(inputs)
        loss_plant = criterion_plant(plant_outputs, plant_labels)
        loss_disease = criterion_disease(disease_outputs, disease_labels)
        loss = loss_plant + 3 * loss_disease  # Give more weight to disease loss
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_plant_loss += loss_plant.item()
        running_disease_loss += loss_disease.item()
        # Calculate average training losses
        avg_train_plant_loss = running_plant_loss / len(train_dataloader)
        avg_train_disease_loss = running_disease_loss / len(train_dataloader)
        # Append losses to the tracking dictionary

        # In the training loop
        losses['train_plant'].append(avg_train_plant_loss)
        losses['train_disease'].append(avg_train_disease_loss)


        plant_outputs, disease_outputs = model(inputs)
        _, plant_preds = torch.max(plant_outputs, 1)
        _, disease_preds = torch.max(disease_outputs, 1)

        total_plant_predictions.extend(plant_preds.cpu().numpy())
        total_plant_targets.extend(plant_labels.cpu().numpy())
        total_plant_scores.extend(plant_outputs.detach().cpu().numpy())

        total_disease_predictions.extend(disease_preds.cpu().numpy())
        total_disease_targets.extend(disease_labels.cpu().numpy())
        total_disease_scores.extend(disease_outputs.detach().cpu().numpy())


        if i % 100 == 99:
            print(f'[{epoch + 1}, {i + 1:5d}] plant loss: {running_plant_loss / 24:.3f}, '
                  f'disease loss: {running_disease_loss / 24:.3f}, LR: {current_lr:.6f}')
            running_plant_loss = 0.0
            running_disease_loss = 0.0

    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, LR after scheduler step: {current_lr:.6f}')

    train_plant_acc = accuracy_score(total_plant_targets, total_plant_predictions)
    train_disease_acc = accuracy_score(total_disease_targets, total_disease_predictions)
    print(f'Training Plant Accuracy: {train_plant_acc:.4f}, Disease Accuracy: {train_disease_acc:.4f}')
    # Inside the training loop, after calculating accuracies
    # After the training loop
    # Calculate and log training metrics
    train_plant_metrics = calculate_metrics(np.array(total_plant_targets), np.array(total_plant_predictions), np.array(total_plant_scores))
    train_disease_metrics = calculate_metrics(np.array(total_disease_targets), np.array(total_disease_predictions), np.array(total_disease_scores))
    log_metrics_to_tensorboard(writer, 'Train/Plant', train_plant_metrics, epoch)
    log_metrics_to_tensorboard(writer, 'Train/Disease', train_disease_metrics, epoch)

    writer.add_figure('Training/Plant_ROC_Curves', 
                      plot_roc_curves(train_plant_metrics['fpr'], train_plant_metrics['tpr'], 
                                      train_plant_metrics['roc_auc'], plant_types, 
                                      'ROC Curves for Plant Types', epoch),
                     global_step=epoch)

    writer.add_figure('Training/Disease_ROC_Curves', 
                      plot_roc_curves(train_disease_metrics['fpr'], train_disease_metrics['tpr'], 
                                      train_disease_metrics['roc_auc'], disease_names, 
                                      'ROC Curves for Diseases', epoch),
                      global_step=epoch)

    writer.add_figure('Training/Plant_PR_Curves', 
                      plot_precision_recall_curves(train_plant_metrics['precision_recall']['precision'], 
                                                   train_plant_metrics['precision_recall']['recall'], 
                                                   train_plant_metrics['precision_recall']['average_precision'], 
                                                   plant_types, 'Precision-Recall Curves for Plant Types', epoch),
                      global_step=epoch)

    writer.add_figure('Training/Disease_PR_Curves', 
                      plot_precision_recall_curves(train_disease_metrics['precision_recall']['precision'], 
                                                   train_disease_metrics['precision_recall']['recall'], 
                                                   train_disease_metrics['precision_recall']['average_precision'], 
                                                   disease_names, 'Precision-Recall Curves for Diseases', epoch),
                      global_step=epoch)

    writer.add_scalar('Training/Plant_Accuracy', train_plant_metrics['accuracy'], epoch)
    writer.add_scalar('Training/Plant_Precision', train_plant_metrics['precision'], epoch)
    writer.add_scalar('Training/Plant_Recall', train_plant_metrics['recall'], epoch)
    writer.add_scalar('Training/Plant_F1', train_plant_metrics['f1'], epoch)
    writer.add_scalar('Training/Plant_ROC_AUC_Micro', train_plant_metrics['roc_auc']['micro'], epoch)
    writer.add_scalar('Training/Plant_ROC_AUC_Macro', train_plant_metrics['roc_auc']['macro'], epoch)
    writer.add_scalar('Losses/Train_Plant_Loss', avg_train_plant_loss, epoch)


    writer.add_scalar('Training/Disease_Accuracy', train_disease_metrics['accuracy'], epoch)
    writer.add_scalar('Training/Disease_Precision', train_disease_metrics['precision'], epoch)
    writer.add_scalar('Training/Disease_Recall', train_disease_metrics['recall'], epoch)
    writer.add_scalar('Training/Disease_F1', train_disease_metrics['f1'], epoch)
    writer.add_scalar('Training/Disease_ROC_AUC_Micro', train_disease_metrics['roc_auc']['micro'], epoch)
    writer.add_scalar('Training/Disease_ROC_AUC_Macro', train_disease_metrics['roc_auc']['macro'], epoch)
    writer.add_scalar('Losses/Train_Disease_Loss', avg_train_disease_loss, epoch)




    # In the training loop, after calculating metrics:
    print(f'Training Plant Metrics - '
          f'Accuracy: {train_plant_metrics["accuracy"]:.4f}, '
          f'Precision: {train_plant_metrics["precision"]:.4f}, '
          f'Recall: {train_plant_metrics["recall"]:.4f}, '
          f'F1: {train_plant_metrics["f1"]:.4f}, '
          f'MCC: {train_plant_metrics["mcc"]:.4f}, '
          f'AUC: {train_plant_metrics["roc_auc"]["micro"]:.4f}, '
          f'Gini: {train_plant_metrics["gini"]:.4f}')
    print(f'Training Disease Metrics - '
          f'Accuracy: {train_disease_metrics["accuracy"]:.4f}, '
          f'Precision: {train_disease_metrics["precision"]:.4f}, '
          f'Recall: {train_disease_metrics["recall"]:.4f}, '
          f'F1: {train_disease_metrics["f1"]:.4f}, '
          f'MCC: {train_disease_metrics["mcc"]:.4f}, '
          f'AUC: {train_disease_metrics["roc_auc"]["micro"]:.4f}, '
          f'Gini: {train_disease_metrics["gini"]:.4f}')




    # Validation
    model.eval()
    val_plant_predictions = []
    val_plant_targets = []
    val_disease_predictions = []
    val_disease_targets = []
    val_plant_outputs = []
    val_disease_outputs = []
    val_plant_scores = []
    val_disease_scores = []
    val_plant_loss = 0.0
    val_disease_loss = 0.0

    with torch.no_grad():
        for inputs, plant_labels, disease_labels in val_dataloader:
            inputs = inputs.to(device)
            plant_labels = plant_labels.to(device)
            disease_labels = disease_labels.to(device)

            plant_outputs, disease_outputs = model(inputs)
            loss_plant = criterion_plant(plant_outputs, plant_labels)
            loss_disease = criterion_disease(disease_outputs, disease_labels)

            val_plant_loss += loss_plant.item()
            val_disease_loss += loss_disease.item()
            # Calculate average validation losses
            avg_val_plant_loss = val_plant_loss / len(val_dataloader)
            avg_val_disease_loss = val_disease_loss / len(val_dataloader)


           # losses['val_plant'].append(avg_val_plant_loss)
           # losses['val_disease'].append(avg_val_disease_lo ss)
            # In the validation loop
            losses['val_plant'].append(avg_val_plant_loss)
            losses['val_disease'].append(avg_val_disease_loss)


            plant_outputs, disease_outputs = model(inputs)
            _, plant_preds = torch.max(plant_outputs, 1)
            _, disease_preds = torch.max(disease_outputs, 1)

            val_plant_predictions.extend(plant_preds.cpu().numpy())
            val_plant_targets.extend(plant_labels.cpu().numpy())
            val_plant_scores.extend(plant_outputs.detach().cpu().numpy())

            val_disease_predictions.extend(disease_preds.cpu().numpy())
            val_disease_targets.extend(disease_labels.cpu().numpy())
            val_disease_scores.extend(disease_outputs.detach().cpu().numpy())


    val_plant_acc = accuracy_score(val_plant_targets, val_plant_predictions)
    val_disease_acc = accuracy_score(val_disease_targets, val_disease_predictions)
    print(f'Validation Plant Accuracy: {val_plant_acc:.4f}, Disease Accuracy: {val_disease_acc:.4f}')


    # Inside the validation loop, after calculating accuracies
   # val_plant_metrics = calculate_metrics(np.array(val_plant_targets), np.array(val_plant_predictions), np.array(val_plant_scores))
   # val_disease_metrics = calculate_metrics(np.array(val_disease_targets), np.array(val_disease_predictions), np.array(val_disease_scores))
    val_plant_metrics = calculate_metrics(np.array(val_plant_targets), np.array(val_plant_predictions), np.array(val_plant_scores))
    val_disease_metrics = calculate_metrics(np.array(val_disease_targets), np.array(val_disease_predictions), np.array(val_disease_scores))
    log_metrics_to_tensorboard(writer, 'Validation/Plant', val_plant_metrics, epoch)
    log_metrics_to_tensorboard(writer, 'Validation/Disease', val_disease_metrics, epoch)
    # Log loss figures
    log_loss_figures_to_tensorboard(writer, losses, epoch)
    # After calculating metrix

    writer.add_figure('Validation/Plant_ROC_Curves', 
                  plot_roc_curves(val_plant_metrics['fpr'], val_plant_metrics['tpr'], 
                                  val_plant_metrics['roc_auc'], plant_types, 
                                  'ROC Curves for Plant Types', epoch),
                  global_step=epoch)

    writer.add_figure('Validation/Disease_ROC_Curves', 
                  plot_roc_curves(val_disease_metrics['fpr'], val_disease_metrics['tpr'], 
                                  val_disease_metrics['roc_auc'], disease_names, 
                                  'ROC Curves for Diseases', epoch),
                  global_step=epoch)


    writer.add_figure('Validation/Plant_PR_Curves', 
                  plot_precision_recall_curves(val_plant_metrics['precision_recall']['precision'], 
                                               val_plant_metrics['precision_recall']['recall'], 
                                               val_plant_metrics['precision_recall']['average_precision'], 
                                               plant_types, 'Precision-Recall Curves for Plant Types', epoch),
                  global_step=epoch)

    writer.add_figure('Validation/Disease_PR_Curves', 
                  plot_precision_recall_curves(val_disease_metrics['precision_recall']['precision'], 
                                               val_disease_metrics['precision_recall']['recall'], 
                                               val_disease_metrics['precision_recall']['average_precision'], 
                                               disease_names, 'Precision-Recall Curves for Diseases', epoch),
                  global_step=epoch)





    # Plot and log confusion matrices
    plant_cm = confusion_matrix(val_plant_targets, val_plant_predictions)
    disease_cm = confusion_matrix(val_disease_targets, val_disease_predictions)

    writer.add_figure('Validation/Plant_Confusion_Matrix', 
                      plot_confusion_matrix(plant_cm, plant_types, 'Plant Types Confusion Matrix', epoch),
                      global_step=epoch)

    writer.add_figure('Validation/Disease_Confusion_Matrix', 
                      plot_confusion_matrix(disease_cm, disease_names, 'Disease Confusion Matrix', epoch),
                      global_step=epoch)

    writer.add_scalar('Validation/Plant_Accuracy', val_plant_metrics['accuracy'], epoch)
    writer.add_scalar('Validation/Plant_Precision', val_plant_metrics['precision'], epoch)
    writer.add_scalar('Validation/Plant_Recall', val_plant_metrics['recall'], epoch)
    writer.add_scalar('Validation/Plant_F1', val_plant_metrics['f1'], epoch)
    writer.add_scalar('Validation/Plant_ROC_AUC_Micro', val_plant_metrics['roc_auc']['micro'], epoch)
    writer.add_scalar('Validation/Plant_ROC_AUC_Macro', val_plant_metrics['roc_auc']['macro'], epoch)
    writer.add_scalar('Metrics/Plant_ROC_AUC_Micro', val_plant_metrics['roc_auc']['micro'], epoch)
    writer.add_scalar('Metrics/Plant_ROC_AUC_Macro', val_plant_metrics['roc_auc']['macro'], epoch)
    writer.add_scalar('Losses/Val_Plant_Loss', avg_val_plant_loss, epoch)
   

    writer.add_scalar('Validation/Disease_Accuracy', val_disease_metrics['accuracy'], epoch)
    writer.add_scalar('Validation/Disease_Precision', val_disease_metrics['precision'], epoch)
    writer.add_scalar('Validation/Disease_Recall', val_disease_metrics['recall'], epoch)
    writer.add_scalar('Validation/Disease_F1', val_disease_metrics['f1'], epoch)
    writer.add_scalar('Validation/Disease_ROC_AUC_Micro', val_disease_metrics['roc_auc']['micro'], epoch)
    writer.add_scalar('Validation/Disease_ROC_AUC_Macro', val_disease_metrics['roc_auc']['macro'], epoch)
    writer.add_scalar('Metrics/Disease_ROC_AUC_Micro', val_disease_metrics['roc_auc']['micro'], epoch)
    writer.add_scalar('Metrics/Disease_ROC_AUC_Macro', val_disease_metrics['roc_auc']['macro'], epoch)
    writer.add_scalar('Losses/Val_Disease_Loss', avg_val_disease_loss, epoch)
    




    # In the validation loop, after calculating metrics:
    print(f'Validation Plant Metrics - '
          f'Accuracy: {val_plant_metrics["accuracy"]:.4f}, '
          f'Precision: {val_plant_metrics["precision"]:.4f}, '
          f'Recall: {val_plant_metrics["recall"]:.4f}, '
          f'F1: {val_plant_metrics["f1"]:.4f}, '
          f'MCC: {val_plant_metrics["mcc"]:.4f}, '
          f'AUC: {val_plant_metrics["roc_auc"]["micro"]:.4f}, '
          f'Gini: {val_plant_metrics["gini"]:.4f}')
    print(f'Validation Disease Metrics - '
          f'Accuracy: {val_disease_metrics["accuracy"]:.4f}, '
          f'Precision: {val_disease_metrics["precision"]:.4f}, '
          f'Recall: {val_disease_metrics["recall"]:.4f}, '
          f'F1: {val_disease_metrics["f1"]:.4f}, '
          f'MCC: {val_disease_metrics["mcc"]:.4f}, '
          f'AUC: {val_disease_metrics["roc_auc"]["micro"]:.4f}, '
          f'Gini: {val_disease_metrics["gini"]:.4f}')




    # Analyze misclassifications
    misclassifications = []
    for true_label, pred_label in zip(val_disease_targets, val_disease_predictions):
        if true_label != pred_label:
            misclassifications.append((disease_names[true_label], disease_names[pred_label]))

    misclassification_counts = Counter(misclassifications)
    print("\nTop 5 Disease Misclassifications:")
    for (true, pred), count in misclassification_counts.most_common(5):
        print(f"True: {true}, Predicted: {pred}, Count: {count}")


    # At the end of each epoch, after validation
    checkpoint_directory = '/Akash/ENV/Checkpoint_Saver/mobilenetv2weights.pth'
    save_checkpoint(model, optimizer, epoch, val_plant_loss, val_disease_loss, 
                val_plant_metrics['accuracy'], val_disease_metrics['accuracy'], 
                checkpoint_directory)

    # Still keep track of the best validation accuracy for early stopping
    val_acc = (val_plant_metrics['accuracy'] + val_disease_metrics['accuracy']) / 2
    if val_acc > best_val_acc:
       best_val_acc = val_acc
       counter = 0
    else:
        counter += 1
        if counter > patience:
            print(f'Early stopping at epoch {epoch}')
            break

print('Training completed.')
writer.close()
