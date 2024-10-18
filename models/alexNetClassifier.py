import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AlexNetSLAMClassifierBase(nn.Module):
    def __init__(self, weights_path, num_classes):
        super().__init__()
        
        # load the pretrained alexnet model
        alexnet = models.alexnet()
        alexnet.load_state_dict(torch.load(weights_path, weights_only=True))

        self.features = alexnet.features

        # Modify the first convolution layer to accept 6 channels instead of 3
        self.features[0] = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)

        # Initialize the modified first conv layer and new layers
        nn.init.kaiming_normal_(self.features[0].weight)
        nn.init.constant_(self.features[0].bias, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
        )

        self.fc1 = nn.Linear(4096, 3) # first RPE component
        self.fc2 = nn.Linear(4096, num_classes) # first RPE component


        self._initialize_weights()

        self.pretrained_parameters = list(self.features.parameters()) + list(self.classifier.parameters())
        self.non_pretrained_parameters = list(self.fc1.parameters()) + list(self.fc2.parameters())


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)


class AlexNetSLAMClassifier(nn.Module):
    def __init__(self, weights_path, num_classes, input_channels=6):
        super(AlexNetSLAMClassifier, self).__init__()
        
        # Load the pretrained AlexNet model
        alexnet = models.alexnet()
        alexnet.load_state_dict(torch.load(weights_path, weights_only=True))

        self.features = alexnet.features

        # Modify the first convolution layer
        original_weights = self.features[0].weight.data  # Shape: [64, 3, 11, 11]
        if input_channels != 3:
            # Duplicate weights to match new input channels
            new_weights = original_weights.repeat(1, input_channels // 3, 1, 1) / (input_channels // 3)
            self.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
            self.features[0].weight.data = new_weights
            self.features[0].bias.data = alexnet.features[0].bias.data

        self.avgpool = alexnet.avgpool

        # Modify the classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        # Output layers for the two RPE components
        self.fc1 = nn.Linear(4096, 3)           # Rotation component
        self.fc2 = nn.Linear(4096, num_classes) # Translation component

        # Initialize the new layers
        self._initialize_weights()

        self.pretrained_parameters = list(self.features.parameters()) + list(self.classifier.parameters())
        self.non_pretrained_parameters = list(self.fc1.parameters()) + list(self.fc2.parameters())

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
