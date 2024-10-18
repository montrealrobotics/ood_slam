from models.alexNetClassifier import AlexNetSLAMClassifier, AlexNetSLAMClassifierBase
from models.resnet18 import ResNet18SLAMClassifier
import torch

def get_model(config):
    architecture = config['model']['architecture']
    weights_path = config['model'].get('weights_path', None)
    num_classes = config['model']['num_outputs'][1]
    input_channels = config['model']['input_channels'] * config['dataset']['sequence_length']

    if architecture == 'alexnet':
        return AlexNetSLAMClassifier(weights_path, num_classes, input_channels)
    elif architecture == 'alexnet_base':
        return AlexNetSLAMClassifierBase(weights_path, num_classes)
    elif architecture == 'resnet18':
        return ResNet18SLAMClassifier(weights_path, num_classes, input_channels)
    else:
        raise ValueError(f"Unsupported model architecture: {architecture}")
