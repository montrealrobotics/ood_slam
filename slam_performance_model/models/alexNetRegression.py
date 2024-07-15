import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetSLAM(nn.Module):
    def __init__(self, weights_path, num_classes=2): # Since RPE has 2 components
        super(AlexNetSLAM, self).__init__()
        self.alexnet = models.alexnet()
        self.alexnet.load_state_dict(torch.load(weights_path))

        # Modify the first convolution layer to accept 6 channels instead of 3
        self.alexnet.features[0] = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)

        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.alexnet(x)
    

if __name__ == "__main__":
    model = AlexNetSLAM
    print(model)