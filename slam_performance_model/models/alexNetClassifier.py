import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetSLAMClassifier(nn.Module):
    def __init__(self, weights_path, num_classes):
        super(AlexNetSLAMClassifier, self).__init__()

        # Load the pretrained alexnet model
        self.alexnet = models.alexnet()
        self.alexnet.load_state_dict(torch.load(weights_path))

        # Modify the first convolution layer to accept 6 channels instead of 3
        self.alexnet.features[0] = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)

        # Modify the classifier part to include 2 heads
        self.alexnet.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.fc1 = nn.Linear(4096, num_classes) # first RPE component
        self.fc2 = nn.Linear(4096, num_classes) # Second RPE component

    def forward(self, x):
        x = self.alexnet.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.alexnet.classifier(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2