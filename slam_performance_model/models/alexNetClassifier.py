import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# class AlexNetSLAMClassifier(nn.Module):
#     def __init__(self, weights_path, num_classes):
#         super(AlexNetSLAMClassifier, self).__init__()

#         # Load the pretrained alexnet model
#         self.alexnet = models.alexnet()
#         self.alexnet.load_state_dict(torch.load(weights_path))

#         # Modify the first convolution layer to accept 6 channels instead of 3
#         self.alexnet.features[0] = nn.Conv2d(6, 64, kernel_size=11, stride=4, padding=2)

#         # Modify the classifier part to include 2 heads
#         self.alexnet.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#         )
#         self.fc1 = nn.Linear(4096, num_classes) # first RPE component
#         self.fc2 = nn.Linear(4096, num_classes) # Second RPE component

#     def forward(self, x):
#         x = self.alexnet.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.alexnet.classifier(x)
#         out1 = self.fc1(x)
#         out2 = self.fc2(x)
#         return out1, out2
    
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

        # self.alexnet.features = nn.Sequential(
        #     nn.Conv2d(6, 64, 3, 2, 1), #in_channels, out_channels, kernel_size, stride, padding
        #     nn.MaxPool2d(2), #kernel_size
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(64, 192, 3, padding = 1),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(192, 384, 3, padding = 1),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(384, 256, 3, padding = 1),
        #     nn.ReLU(inplace = True),
        #     nn.Conv2d(256, 256, 3, padding = 1),
        #     nn.MaxPool2d(2),
        #     nn.ReLU(inplace = True)
        # )
        
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

    def forward(self, x):
        x = self.features(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)

        out1 = self.fc1(x)
        out2 = self.fc2(x)

        return out1, out2
    
    def _initialize_weights(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
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
        input_prob = torch.softmax(input, dim=1)
        
        # Calculate the cumulative distribution functions
        input_cdf = torch.cumsum(input_prob, dim=1)
        target_cdf = torch.cumsum(target, dim=1)
        
        # Compute the EMD squared
        emd_squared = torch.mean(torch.sum((input_cdf - target_cdf) ** 2, dim=1))
        
        return emd_squared