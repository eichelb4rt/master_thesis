import torch
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.activation import ReLU
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.nn.modules.pooling import AdaptiveAvgPool2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.container import Sequential
from torch.nn.modules.dropout import Dropout
from nts_net.model import attention_net
from nts_net.model import ProposalNet
from nts_net.resnet import ResNet
from nts_net.resnet import Bottleneck

model = torch.hub.load('nicolalandro/ntsnet-cub200', 'ntsnet', pretrained=True, **{'topN': 6, 'device': 'cpu', 'num_classes': 200})


def transform_conv_2d(model: Conv2d):
    print("Conv2d", {
        "kernel_size": model.kernel_size,
        "in_channels": model.in_channels,
        "out_channels": model.out_channels,
    })


def transform_batch_norm_2d(model: BatchNorm2d):
    print("BatchNorm2d", {
        "num_features": model.num_features,
    })


def transform_relu(_: ReLU):
    print("ReLU")


def transform_linear(model: Linear):
    print("Linear", {
        "weight": model.weight,
        "bias": model.bias,
    })


def transform_max_pool_2d(model: MaxPool2d):
    print("Linear", {
        "kernel_size": model.kernel_size,
    })


def transform_adaptive_avg_pool_2d(model: AdaptiveAvgPool2d):
    print("AdaptiveAveragePool2d", {
        "output_size": model.output_size,
    })


def transform_dropout(model: Dropout):
    print("Dropout", {
        "p": model.p,
    })


def transform(model: Module):
    # if we have to go deeper, go deeper
    if isinstance(model, Sequential | ResNet | ProposalNet | Bottleneck | attention_net):
        for child in model.children():
            transform(child)
        return
    # we don't have to go deeper here
    if isinstance(model, Linear):
        transform_linear(model)
    elif isinstance(model, Conv2d):
        transform_conv_2d(model)
    elif isinstance(model, BatchNorm2d):
        transform_batch_norm_2d(model)
    elif isinstance(model, ReLU):
        transform_relu(model)
    elif isinstance(model, MaxPool2d):
        transform_max_pool_2d(model)
    elif isinstance(model, AdaptiveAvgPool2d):
        transform_adaptive_avg_pool_2d(model)
    elif isinstance(model, Dropout):
        transform_dropout(model)
    else:
        raise ValueError(f"Module not known: {model.__class__}")


def main():
    transform(model)


if __name__ == "__main__":
    main()
