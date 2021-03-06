from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn
from models.UNet import UNet

def get_model(config):
    nc = config['num_classes']
    
    if config['model'] == "deeplab_resnet50":
        # instantiate the pretrained model
        model = deeplabv3_resnet50(pretrained=config['pretrained'], progress=False, num_classes=21)
        
        # replace the final classifier layer with an untrained layer with N classes
        model.classifier[4] = nn.Conv2d(256, nc, kernel_size=(1, 1), stride=(1, 1))
        
    elif config['model'] == "UNet":
        # instantiate the UNet model
        # note: as of now, the image dimensions of UNet only allow
        # dimensions that are exponents of 2, such as 32, 128, 256, etc etc.
        if config['pretrained']:
            raise NotImplementedError("No available pre-trained version of UNet")
        model = UNet(config)
    else:
        raise NotImplementedError

    # Calculate number of parameters. 
    num_parameters = sum([x.nelement() for x in model.parameters()])
    print(f"The number of parameters in {config['model']}: {num_parameters/1000:9.2f}k")
    return model

# from get_config import get_config
# cfg = get_config()
# model = get_model(cfg)
# print(model)