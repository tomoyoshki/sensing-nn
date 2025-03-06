# src/models/StandardResNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FusionModules import MeanFusionBlock

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_ratio=0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_ratio)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, dropout_ratio=0):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.dropout = nn.Dropout2d(p=dropout_ratio)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        out = self.bn3(self.conv3(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetBackbone(nn.Module):
    def __init__(self, block_type, layers, in_channels, dropout_ratio):
        super().__init__()
        
        self.in_channels = 64
        self.block_type = block_type
        self.dropout_ratio = dropout_ratio
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        
        # Final pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(self.block_type(self.in_channels, out_channels, stride, self.dropout_ratio))
        self.in_channels = out_channels * self.block_type.expansion
        for _ in range(1, blocks):
            layers.append(self.block_type(self.in_channels, out_channels, dropout_ratio=self.dropout_ratio))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class MultiModalResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = args.dataset_config["StandardResNet"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1

        # Define block type
        block_type = BasicBlock if self.config["block_type"] == "basic" else Bottleneck

        # Create ResNet backbone for each modality and location
        self.mod_loc_backbones = nn.ModuleDict()
        for loc in self.locations:
            self.mod_loc_backbones[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.mod_loc_backbones[loc][mod] = ResNetBackbone(
                    block_type=block_type,
                    layers=self.config["layers"],
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    dropout_ratio=self.config["dropout_ratio"]
                )

        # Feature dimension after ResNet backbone
        self.feature_dim = 512 * block_type.expansion

        # Modality fusion for each location
        self.mod_fusion_layers = nn.ModuleDict()
        for loc in self.locations:
            self.mod_fusion_layers[loc] = MeanFusionBlock()

        # Location fusion if multiple locations
        if self.multi_location_flag:
            self.loc_fusion_layer = MeanFusionBlock()

        # Final classification layers
        self.sample_embd_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.config["fc_dim"]),
            nn.ReLU(),
        )
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["fc_dim"], args.dataset_config["num_classes"]),
        )

    def forward(self, freq_x, class_head=True):
        # Extract features for each modality at each location
        loc_mod_features = {}
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                features = self.mod_loc_backbones[loc][mod](freq_x[loc][mod])
                loc_mod_features[loc].append(features)
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=1)

        # Fuse modalities for each location
        fused_loc_features = {}
        for loc in self.locations:
            fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])

        # Fuse locations if multiple locations
        if not self.multi_location_flag:
            final_feature = fused_loc_features[self.locations[0]]
        else:
            loc_features = torch.stack([fused_loc_features[loc] for loc in self.locations], dim=1)
            final_feature = self.loc_fusion_layer(loc_features)


        # print(final_feature.shape)
        # Classification
        sample_features = self.sample_embd_layer(final_feature)

        if class_head:
            logits = self.class_layer(sample_features)
            return logits
        else:
            return sample_features