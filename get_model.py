from torchvision.models.segmentation import deeplabv3_resnet50
import torch.nn as nn

def get_model(config):
    nc = config.num_classes
    
    if config.model == "deeplab_resnet50":
        # instantiate the pretrained model
        model = deeplabv3_resnet50(pretrained=True, progress=False, num_classes=21)
        
        # replace the final classifier layer with an untrained layer with N classes
        model.classifier[4] = nn.Conv2d(256, nc, kernel_size=(1, 1), stride=(1, 1))
    else:
        raise NotImplementedError

    # Calculate number of parameters. 
    num_parameters = sum([x.nelement() for x in model.parameters()])
    print(f"The number of parameters in {config.model}: {num_parameters/1000:9.2f}k")
    return model

#model = get_model(10)
#print(model)