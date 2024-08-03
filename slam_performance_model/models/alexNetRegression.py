import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetSLAMRegressor(nn.Module):
    def __init__(self, weights_path):
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

        self.fc = nn.Linear(4096, 2)

        self._initialize_weights()

        # for name, param in self.classifier.named_parameters():
        #     param.requires_grad = False


    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        out = self.fc(x)

        return out
    
    def _initialize_weights(self):
        # for m in self.classifier:
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_normal_(m.weight)
        #         nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    

if __name__ == "__main__":
    model = AlexNetSLAMRegressor()
    print(model)