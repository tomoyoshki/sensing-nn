# src/models/StandardResNet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.FusionModules import MeanFusionBlock
from models.QuantModules import QuanConv, CustomBatchNorm, DoReFaA, PACT

class CustomSequential(nn.Sequential):
    def forward(self, x, teacher_input=None):
        if teacher_input is None:
            for module in self:
                x = module(x)
            return x
        else:
            x_student, x_teacher = x, teacher_input
            for module in self:
                x_student, x_teacher = module(x_student, teacher_input = x_teacher)
            return x_student, x_teacher

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,args, in_channels, out_channels, stride=1, dropout_ratio=0):
        super().__init__()

        self.quantization_config = args.dataset_config["quantization"]
        self.args = args
        
        self.conv1 = QuanConv(in_channels, out_channels, kernel_size=(3,3), 
                              stride=stride, padding=1, bias=False)
        self.bn1 = CustomBatchNorm(out_channels)
        
        self.conv2 = QuanConv(out_channels, out_channels, kernel_size=(3,3),
                              stride=1, padding=1, bias=False)
        self.bn2 = CustomBatchNorm(out_channels)
        self.dropout = nn.Dropout2d(p=dropout_ratio)

        self.shortcut = CustomSequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            # print(f"{'-'*20} BasicBlock - Creating shortcut {'-'*20}")
            # print(f"Shortcut conv initialization - in_channels: {in_channels}, out_channels: {self.expansion * out_channels}")
            self.shortcut = CustomSequential(
                QuanConv(in_channels, self.expansion * out_channels,
                         kernel_size=(1,1), stride=stride, bias=False),
                CustomBatchNorm(self.expansion * out_channels)
            )

        self.bn1.set_corresponding_input_output_convs(input_conv=self.conv1, output_conv=self.conv2)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.bn2.set_corresponding_input_output_convs(input_conv=self.conv2, output_conv=self.shortcut[0])
            self.shortcut[1].set_corresponding_input_conv(input_conv=self.shortcut[0])
            self.last_conv = self.shortcut[0]
            self.last_batch_norm = self.shortcut[1]
        else:
            self.bn2.set_corresponding_input_conv(input_conv=self.conv2)
            self.last_conv = self.conv2
            self.last_batch_norm = self.bn2

        self.first_conv = self.conv1

    def get_last_conv(self):
        return self.last_conv
    
    def get_first_conv(self):
        return self.first_conv
    
    def get_last_batch_norm(self):
        return self.last_batch_norm
    
    def set_next_conv(self, next_conv):

        self.last_batch_norm.set_corresponding_output_conv(output_conv=next_conv)
    
    def forward(self, x, teacher_input = None):
        if not self.quantization_config["enable"]:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.dropout(out)
            out = self.bn2(self.conv2(out))
            out = self.dropout(out)
            out += self.shortcut(x)
            out = F.relu(out)
            return out
        elif self.quantization_config["enable"]:
            if teacher_input is None:
                out = self.bn1(self.conv1(x))
                out = self.dropout(out)
                out = self.bn2(self.conv2(out))
                out = self.dropout(out)
                out += self.shortcut(x)
                # out = F.relu(out) # Point to check whether we need to add activation quantization step here - it would be handled by the next layer ideally
                return out
            else:
                # print(f"{'-'*20} BasicBlock - Student and Teacher Forward Pass {'-'*20}")
                # print(f"Input shape: {x.shape}")
                out_student, out_teacher = self.conv1(x, teacher_input)
                try:
                    out_student, out_teacher = self.bn1(out_student, teacher_input=out_teacher)
                except:
                    print(f"BN1: Error in BasicBlock forward pass")
                    breakpoint()
                    raise
                # print(f"After conv1 - Student features shape: {out_student.shape}")
                # print(f"After conv1 - Teacher features shape: {out_teacher.shape}")
                out_student = self.dropout(out_student)
                out_teacher = self.dropout(out_teacher)

                out_student, out_teacher = self.conv2(out_student, out_teacher)
                # print(f"After conv2 - Student features shape: {out_student.shape}")
                # print(f"After conv2 - Teacher features shape: {out_teacher.shape}")
                try:
                    out_student, out_teacher = self.bn2(out_student, teacher_input=out_teacher)
                except:
                    print(f"BN2: Error in BasicBlock forward pass")
                    breakpoint()
                    raise
                out_student = self.dropout(out_student)
                out_teacher = self.dropout(out_teacher)

                out_student_shortcut, out_teacher_shortcut = self.shortcut(x, teacher_input)
                # print(f"After shortcut - Student features shape: {out_student_shortcut.shape}")
                # print(f"After shortcut - Teacher features shape: {out_teacher_shortcut.shape}")
                out_student += out_student_shortcut
                out_teacher += out_teacher_shortcut
                # print(f"After addition - Student features shape: {out_student.shape}")
                # print(f"After addition - Teacher features shape: {out_teacher.shape}")

                return out_student, out_teacher

# Can ignore this for now -  not using it
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
    def __init__(self, args, block_type, layers, in_channels, dropout_ratio, batch_norm, batch_norm_type):
        super().__init__()
        self.quantization_config = args.dataset_config["quantization"]
        self.args = args
        self.in_channels = 64
        self.block_type = block_type # BasicBlock
        self.dropout_ratio = dropout_ratio
        self.batch_norm = CustomBatchNorm
        self.batch_norm_type = batch_norm_type
        self.alpha1= nn.Parameter(torch.ones(1), requires_grad=True) # Alpha for PACT for activation post first layer.
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7),
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # self.bn1.set_corresponding_input_conv(input_conv=self.conv1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1, last_bn_layer1 = self._make_layer(64, layers[0],prev_batch_norm=self.bn1)
        self.layer2, last_bn_layer2 = self._make_layer(128, layers[1], stride=2,prev_batch_norm=last_bn_layer1)
        self.layer3, last_bn_layer3 = self._make_layer(256, layers[2], stride=2,prev_batch_norm=last_bn_layer2)
        self.layer4, last_bn_layer4 = self._make_layer(512, layers[3], stride=2,prev_batch_norm=last_bn_layer3)
        
        # Final pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #Activation Quantization Function
        if self.quantization_config["enable"]:
            if self.quantization_config["activation_quantization"] == "dorefa":
                self.activation_quant = DoReFaA()
            elif self.quantization_config["activation_quantization"] == "pact":
                self.activation_quant = PACT()
            else:
                raise ValueError(f"Invalid activation quantization function: {self.quantization_config['activation_quantization']}, \
                                correct options are 'dorefa' or 'pact'. \
                                Please add them to the dataset config (YAML) file.")
            
        self.nbit_first_layer = 16


    # check this function again make sure its right
    def _make_layer(self, out_channels, blocks, stride=1,prev_batch_norm=None):
        layers = [] # basicblock1, basickblock2, ... 
        layers.append(self.block_type(self.args, self.in_channels, out_channels, stride, self.dropout_ratio))
        if isinstance(prev_batch_norm, CustomBatchNorm): # if prev_batch_norm is nn.BatchNorm2d then we do not set the corresponding input conv
            # the CustomBatchNorm class will assume that absence of input_conv or output_conv means that use 32 bit floating point
            prev_batch_norm.set_corresponding_output_conv(layers[0].get_first_conv()) # Should be 0 not -1
        self.in_channels = out_channels * self.block_type.expansion
        for i in range(1, blocks):
            layers.append(self.block_type(self.args, self.in_channels, out_channels, dropout_ratio=self.dropout_ratio))
            layers[i-1].set_next_conv(layers[i].get_first_conv()) # Set the next conv layer for the previous block (its 
            # just a helper function for layers[i-1].last_conv.set_corresponding_output_conv(layers[i].first_conv)

        
        # first_conv_layer = layers[0].first_conv
        last_bn_layer = layers[-1].get_last_batch_norm()
        return CustomSequential(*layers), last_bn_layer

    def forward(self, x):
        if not self.quantization_config["enable"]:
            # x = x.float()
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        elif self.quantization_config["enable"]:
            x = self.bn1(self.conv1(x))
            x = self.maxpool(x)
            if not self.training or not self.quantization_config["teacher_student"]:
            # Testing mode - only student forward pass
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x
            else:
                # Training mode - both student and teacher forward passes
                # print(f"Before layer1 - Input shape: {x.shape}")
                x_student, x_teacher = self.layer1(x, x)
                # print(f"After layer1 - Student features shape: {x_student.shape}")
                # print(f"After layer1 - Teacher features shape: {x_teacher.shape}")
                
                x_student, x_teacher = self.layer2(x_student, x_teacher)
                # print(f"After layer2 - Student features shape: {x_student.shape}")
                # print(f"After layer2 - Teacher features shape: {x_teacher.shape}")
                x_student, x_teacher = self.layer3(x_student, x_teacher)
                x_student, x_teacher = self.layer4(x_student, x_teacher)

                x_student = self.avgpool(x_student)
                x_teacher = self.avgpool(x_teacher)

                x_student = torch.flatten(x_student, 1)
                x_teacher = torch.flatten(x_teacher, 1)

                return x_student, x_teacher
         


class ResNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = args.dataset_config["ResNet"]
        self.modalities = args.dataset_config["modality_names"]
        self.locations = args.dataset_config["location_names"]
        self.multi_location_flag = len(self.locations) > 1
        self.bn = CustomBatchNorm
        self.Conv = QuanConv
        self.quantization_config = args.dataset_config["quantization"]
        # breakpoint()
        if self.quantization_config["bn_type"] == "float" or self.quantization_config["bn_type"] == "switch" or self.quantization_config["bn_type"] == "transitional":
            self.bn_type = self.quantization_config["bn_type"]
        else:
            raise ValueError(f"Invalid batch normalization type: {self.quantization_config['bn_type']}, \
                             correct options are 'float', 'switch', or 'transitional'. \
                             Please add them to the dataset config (YAML) file.")

        # Define block type
        block_type = BasicBlock if self.config["block_type"] == "basic" else Bottleneck

        # Create ResNet backbone for each modality and location
        self.mod_loc_backbones = nn.ModuleDict()
        for loc in self.locations:
            self.mod_loc_backbones[loc] = nn.ModuleDict()
            for mod in self.modalities:
                self.mod_loc_backbones[loc][mod] = ResNetBackbone(
                    args=args,
                    block_type=block_type,
                    layers=self.config["layers"],
                    in_channels=args.dataset_config["loc_mod_in_freq_channels"][loc][mod],
                    dropout_ratio=self.config["dropout_ratio"],
                    batch_norm_type=self.bn_type,
                    batch_norm=self.bn
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
        # breakpoint()
        self.class_layer = nn.Sequential(
            nn.Linear(self.config["fc_dim"], args.dataset_config["vehicle_classification"]["num_classes"]),
        )



    def forward_vanilla(self, freq_x, class_head=True):
        """
            Forward Pass for testing or training without distillation, you can run either quantized or non-quantized forward pass
        """
        loc_mod_features = {}
        # testing = not self.training
        # if testing or not self.quantization_config["teacher_student"]:
        for loc in self.locations:
            loc_mod_features[loc] = []
            for mod in self.modalities:
                # breakpoint()
                features = self.mod_loc_backbones[loc][mod](freq_x[loc][mod])
                loc_mod_features[loc].append(features)
            loc_mod_features[loc] = torch.stack(loc_mod_features[loc], dim=1)



        # Fuse modalities for each location
        # breakpoint()
        fused_loc_features = {}
        for loc in self.locations:
            # breakpoint()
            # fused_loc_features[loc] = self.mod_fusion_layers[loc](loc_mod_features[loc])
            fused_loc_features[loc] = loc_mod_features[loc].mean(dim=1)

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
            return logits, None
        else:
            return sample_features, None

    def forward_distillation(self, freq_x, class_head=True):
        """
            Forward Pass for training with distillation
        """
        loc_mod_features_student = {}
        loc_mod_features_teacher = {}
        
        for loc in self.locations:
            loc_mod_features_student[loc] = []
            loc_mod_features_teacher[loc] = []
            for mod in self.modalities:
                # print(f"Before backbone - Input shape for {loc} {mod}: {freq_x[loc][mod].shape}")
                features_student, features_teacher = self.mod_loc_backbones[loc][mod](freq_x[loc][mod])
                # print(f"After backbone - Student features shape: {features_student.shape}")
                # print(f"After backbone - Teacher features shape: {features_teacher.shape}")
                loc_mod_features_student[loc].append(features_student)
                loc_mod_features_teacher[loc].append(features_teacher)
            
            loc_mod_features_student[loc] = torch.stack(loc_mod_features_student[loc], dim=1)
            loc_mod_features_teacher[loc] = torch.stack(loc_mod_features_teacher[loc], dim=1)

        # Fuse modalities for each location
        fused_loc_features_student = {}
        fused_loc_features_teacher = {}
        for loc in self.locations:
            fused_loc_features_student[loc] = loc_mod_features_student[loc].mean(dim=1)
            fused_loc_features_teacher[loc] = loc_mod_features_teacher[loc].mean(dim=1)

        # Fuse locations if multiple locations
        if not self.multi_location_flag:
            final_feature_student = fused_loc_features_student[self.locations[0]]
            final_feature_teacher = fused_loc_features_teacher[self.locations[0]]
        else:
            loc_features_student = torch.stack([fused_loc_features_student[loc] for loc in self.locations], dim=1)
            loc_features_teacher = torch.stack([fused_loc_features_teacher[loc] for loc in self.locations], dim=1)
            final_feature_student = self.loc_fusion_layer(loc_features_student)
            final_feature_teacher = self.loc_fusion_layer(loc_features_teacher)

        # Classification
        sample_features_student = self.sample_embd_layer(final_feature_student)
        sample_features_teacher = self.sample_embd_layer(final_feature_teacher)
        
        if class_head:
            logits_student = self.class_layer(sample_features_student)
            logits_teacher = self.class_layer(sample_features_teacher)
            return logits_student, logits_teacher
        else:
            return sample_features_student, sample_features_teacher

    def forward(self, freq_x, class_head=True):
        '''
            Forward pass for the model, we direct the forward pass to the correct function based on the training mode set in the yaml file. 
        '''
        testing = not self.training
        if testing:
            forward_func = self.forward_vanilla
        else:
             # Training mode - both student and teacher forward passes
            if self.quantization_config["teacher_student"]:
                forward_func = self.forward_distillation
            else:
                # breakpoint()
                forward_func = self.forward_vanilla

        return forward_func(freq_x, class_head=class_head)