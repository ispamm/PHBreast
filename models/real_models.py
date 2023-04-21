'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import load_weights

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channels=4, num_classes=10, gap_output=False, before_gap_output=False, visualize=False):
        super(ResNet, self).__init__()
        self.block = block
        self.num_blocks = num_blocks
        self.in_planes = 64
        self.gap_output = gap_output
        self.before_gap_out = before_gap_output
        self.visualize = visualize

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer5 = None
        self.layer6 = None
        if not gap_output and not before_gap_output:
            self.linear = nn.Linear(512*block.expansion, num_classes)
    
    def add_top_blocks(self, num_classes=1):
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2)
        
        if not self.gap_output and not self.before_gap_out:
            self.linear = nn.Linear(1024, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        features_4 = out
        
        if self.before_gap_out:
            return out
        
        if self.layer5:
            out = self.layer5(out)
            out = self.layer6(out)
            features_6 = out
        
        # global average pooling (GAP)
        n, c, _, _ = out.size()
        out = out.view(n, c, -1).mean(-1)
        
        if self.gap_output:
            return out

        out = self.linear(out)

        if self.visualize:
            # return the final output and activation maps at two different levels
            return out, features_4, features_6
        return out

class Encoder(nn.Module):
    def __init__(self, channels):
        super(Encoder, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        return out

class SharedBottleneck(nn.Module):
    def __init__(self, in_planes):
        super(SharedBottleneck, self).__init__()
        self.in_planes = in_planes

        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer3(x)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        n, c, _, _ = out.size()
        out = out.view(n, c, -1).mean(-1)       
        return out

class Classifier(nn.Module):
    def __init__(self, num_classes, in_planes=512, visualize=False):
        super(Classifier, self).__init__()
        self.in_planes = in_planes
        self.visualize = visualize

        self.layer5 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.layer6 = self._make_layer(Bottleneck, 512, 2, stride=2)
        self.linear = nn.Linear(1024, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer5(x)
        feature_maps = self.layer6(out)

        n, c, _, _ = feature_maps.size()
        out = feature_maps.view(n, c, -1).mean(-1)
        out = self.linear(out)

        if self.visualize:
            return out, feature_maps

        return out

class SBOnet(nn.Module):
    """
    SBOnet.

    Parameters:
    - shared: True to share the Bottleneck between the two sides, False for the 'concat' version. 
    - weights: path to pretrained weights of patch classifier for Encoder branches
    """

    def __init__(self, shared=True, num_classes=1, weights=None):
        super(SBOnet, self).__init__()
        
        self.shared = shared
        
        self.encoder_sx = Encoder(channels=2)
        self.encoder_dx = Encoder(channels=2)
        
        self.shared_resnet = SharedBottleneck(in_planes=128 if shared else 256)
        
        if weights:
            load_weights(self.encoder_sx, weights)
            load_weights(self.encoder_dx, weights)
        
        self.classifier_sx = nn.Linear(1024, num_classes)
        self.classifier_dx = nn.Linear(1024, num_classes)

    def forward(self, x):
        x_sx, x_dx = x
        
        # Apply Encoder
        out_sx = self.encoder_sx(x_sx)
        out_dx = self.encoder_dx(x_dx)
        
        # Shared layers
        if self.shared:
            out_sx = self.shared_resnet(out_sx)
            out_dx = self.shared_resnet(out_dx)
            
            out_sx = self.classifier_sx(out_sx)
            out_dx = self.classifier_dx(out_dx)
            
        else: # Concat version  
            out = torch.cat([out_sx, out_dx], dim=1)
            out = self.shared_resnet(out)
            out_sx = self.classifier_sx(out)
            out_dx = self.classifier_dx(out)
        
        out = torch.cat([out_sx, out_dx], dim=0)
        return out

class SEnet(nn.Module):
    """
    SEnet.

    Parameters:
    - weights: path to pretrained weights of patch classifier for PHCResNet18 encoder or path to whole-image classifier
    - patch_weights: True if the weights correspond to patch classifier, False if they are whole-image. 
                     In the latter case also Classifier branches will be initialized.
    """

    def __init__(self, num_classes=1, weights=None, patch_weights=True, visualize=False):
        super(SEnet, self).__init__()
        self.visualize = visualize
        self.resnet18 = ResNet18(num_classes=num_classes, channels=2, before_gap_output=True)
        
        if weights:
            print("Loading weights for resnet18 from ", weights)
            load_weights(self.resnet18, weights)
        
        self.classifier_sx = Classifier(num_classes, visualize=visualize)
        self.classifier_dx = Classifier(num_classes, visualize=visualize)

        if not patch_weights and weights:
            print("Loading weights for classifiers from ", weights)
            load_weights(self.classifier_sx, weights)
            load_weights(self.classifier_dx, weights)
    
    def forward(self, x):
        x_sx, x_dx = x

        # Apply Encoder
        out_enc_sx = self.resnet18(x_sx)
        out_enc_dx = self.resnet18(x_dx)
        
        if self.visualize:
            out_sx, act_sx = self.classifier_sx(out_enc_sx)
            out_dx, act_dx = self.classifier_dx(out_enc_dx)
        else:
            # Apply refiner blocks + classifier
            out_sx = self.classifier_sx(out_enc_sx)
            out_dx = self.classifier_dx(out_enc_dx)
        
        out = torch.cat([out_sx, out_dx], dim=0)

        if self.visualize:
            return out, out_enc_sx, out_enc_dx, act_sx, act_dx

        return out


def ResNet18(num_classes=10, channels=4, gap_output=False, before_gap_output=False, visualize=False):
    return ResNet(BasicBlock, 
                  [2, 2, 2, 2], 
                  num_classes=num_classes, 
                  channels=channels, 
                  gap_output=gap_output, 
                  before_gap_output=before_gap_output,
                  visualize=visualize)

def ResNet50(num_classes=10, channels=4):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, channels=channels)
