import torch
import torch.nn as nn
from torchvision import models

def build_cnn():
    """
    Builds a simple vanilla CNN architecture
    """
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 56 * 56, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

class CataractModel(nn.Module):
    """
    Custom CNN model for cataract classification
    """
    def __init__(self, model_name='resnet50', pretrained=True):
        super(CataractModel, self).__init__()
        self.model_name = model_name
        
        # Initialize base model
        if model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)
        elif model_name == 'vgg16':
            self.base_model = models.vgg16(pretrained=pretrained)
            self.base_model.classifier[-1] = nn.Linear(4096, 1)
        elif model_name == 'mobilenet_v2':
            self.base_model = models.mobilenet_v2(pretrained=pretrained)
            self.base_model.classifier[-1] = nn.Linear(1280, 1)
        elif model_name == 'vanilla_cnn':
            self.base_model = build_cnn()
        else:
            raise ValueError(f"Model {model_name} not supported")

    def forward(self, x):
        return self.base_model(x)

def get_model(model_name='resnet50', pretrained=True, device='cuda'):
    """
    Factory function to get model instance
    """
    model = CataractModel(model_name=model_name, pretrained=pretrained)
    model = model.to(device)
    return model

class ModelCheckpoint:
    """
    Class to handle model checkpointing
    """
    def __init__(self, filepath, monitor='val_loss', mode='min'):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, current_score, model, epoch):
        if self.mode == 'min':
            improved = current_score < self.best_score
        else:
            improved = current_score > self.best_score
            
        if improved:
            self.best_score = current_score
            self.save_checkpoint(model, epoch, current_score)
            return True
        return False
            
    def save_checkpoint(self, model, epoch, score):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
        }, self.filepath)