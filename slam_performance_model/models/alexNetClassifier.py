import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AlexNetSLAMClassifier(nn.Module):
    def __init__(self, weights_path, num_classes):
        super().__init__()
        
        # load the pretrained alexnet model
        alexnet = models.alexnet()
        alexnet.load_state_dict(torch.load(weights_path))

        self.features = alexnet.features

        # Modify the first convolution layer to accept 6 channels instead of 3
        self.features[0] = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)

        # Initialize the modified first conv layer and new layers
        nn.init.kaiming_normal_(self.features[0].weight)
        nn.init.constant_(self.features[0].bias, 0)

        # # Freeze all layers except the first and the classifier
        # for name, param in self.features.named_parameters():
        #     if name != "0.weight" and name != "0.bias":
        #         param.requires_grad = False

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
        )

        self.fc1 = nn.Linear(4096, num_classes) # first RPE component
        self.fc2 = nn.Linear(4096, num_classes) # first RPE component


        self._initialize_weights()

        # for name, param in self.classifier.named_parameters():
        #     if name == "1.weight" or name == "1.bias":
        #         param.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2
    
    def _initialize_weights(self):
        # for m in self.classifier:
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1)  # 6 input channels (concatenated images)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, num_classes)  # Adjust the input size based on the final feature map size
        self.fc2 = nn.Linear(32 * 56 * 56, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # First conv layer + relu + pool
        x = self.pool(F.relu(self.conv2(x)))  # Second conv layer + relu + pool
        x = x.view(-1, 32 * 56 * 56)          # Flatten the feature map
        out1 = self.fc1(x)                    # First fully connected layer
        out2 = self.fc2(x)                    # Second fully connected layer
        return out1, out2
    
# EMD Squared Loss Function
class EMDSquaredLoss(nn.Module):
    def __init__(self):
        super(EMDSquaredLoss, self).__init__()

    def forward(self, input, target):
        # Apply softmax to the input to get probabilities
        input_prob = torch.exp(torch.log_softmax(input, dim=1))
        
        # Calculate the cumulative distribution functions
        input_cdf = torch.cumsum(input_prob, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        
        # Compute the EMD squared
        emd_squared = torch.mean(torch.sum((input_cdf - target_cdf) ** 2, dim=1))
        
        return emd_squared