dependencies = ['torch']

import torch
from impl import LeNet5


def lenet5(pretrained=False, **kwargs):
    """ My Implementation of the LeNet5 ConvNet
    LeNet5 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    model = LeNet5()

    if pretrained:
        model.load_state_dict(torch.load('./model/lene5.model'))

    return model