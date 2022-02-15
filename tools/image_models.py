import torch
from torchvision import models


def get_image_model(arch, layer, icon_pretrained):
    if arch == 'resnet101' and not icon_pretrained:
        model = models.resnet101(pretrained=True)

        if layer == "pool5":
            # get features after the last AvgPool
            modules = list(model.children())[:-1] # out: [2048,1,1]
        model = torch.nn.Sequential(*modules)

    elif arch == 'resnet101' and icon_pretrained:
        # load image model pretrained on our propossed Icon645 dataset
        model = models.__dict__['resnet101'](pretrained=False, num_classes=377) # icon classes from Icon645

        model_path = "../saved_models/icon_classification_ckpt/icon_resnet101_LDAM_DRW_lr0.01_0/ckpt.epoch66_best.pth.tar"
        print("loading pretrained models on icon data: ", model_path)
        checkpoint = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['state_dict'])

        if layer == "pool5":
            # get features after the last AvgPool
            modules = list(model.children())[:-1] # out: [2048,1,1]
        model = torch.nn.Sequential(*modules)
        
    return model
