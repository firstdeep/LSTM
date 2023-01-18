from models.unet3d_model import UNet3D
from flopth import flopth
import torch
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchsummary import summary as summary
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


from SW_SEG_dataset import *

from models.mia import MIA


def get_instance_mia_model():
    model = MIA()
    return model


model = UNet3D(n_channels=1, n_classes=1)
print(model)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dummy_inputs = torch.rand(1, 1, 8, 512, 512)
dummy_inputs.to(device)
flops, params = flopth(model, inputs=(dummy_inputs,))
print(flops, params)


def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model
#
# import torchvision.models as models
# maskrcnn_resnet50_fpn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
#
# seg_model = get_instance_segmentation_model(2)
# print(seg_model)
# summary(seg_model, (3,512,512))

# seg_model.to(device)
# dummy_inputs = torch.rand(1, 3, 512, 512)
#
# flops, params = flopth(seg_model, inputs=(dummy_inputs,))
# print(flops, params)




from models.unet_model import UNet

def get_instance_unet_model(n_channels, n_classes):
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    return model

import segmentation_models_pytorch as smp
def get_instance_unetPlusPlus_model():
    # model = smp.UnetPlusPlus(in_channels=3, classes=1)
    model = smp.Unet(in_channels=3, classes=1)
    return model

# seg_model = get_instance_unetPlusPlus_model()
# print(seg_model)
#
# seg_model.to(device)
# dummy_inputs = torch.rand(1, 3, 512, 512)
#
# flops, params = flopth(seg_model, inputs=(dummy_inputs,))
# print(flops, params)

#Mia
# seg_model = get_instance_mia_model()
# print(seg_model)
#
# seg_model.to(device)
# dummy_inputs = torch.rand(1, 3, 512, 512)
#
# flops, params = flopth(seg_model, inputs=(dummy_inputs,))
# print(flops, params)

