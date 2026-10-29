import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

def build_model(num_classes = 2, pretrained = True):
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)


    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_model(pretrained=False)
    dummy_image = torch.randn(2,3,224,224)

    with torch.no_grad():
        output = model(dummy_image)

    print("input shape:", dummy_image.shape)
    print("output shape:", output.shape)    
    print("Model created successfully")




