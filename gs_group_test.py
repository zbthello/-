import torch
from torch import nn
import os
from PIL import Image
N = 4
num_objects = 16
num_classes = 256
classifier = torch.nn.Conv2d(num_objects, num_classes, kernel_size=1).cuda()
objects = torch.randn(1, 16, 2, 2, device="cuda")  # 16维的2*2的图片
logits = classifier(objects)
print(logits.size())  # 变为256维2*2的图片
objects_folder = "C:\\Users\\12178\\Desktop\\大论文代码\\第四章内容\\bear\\object_mask"
image_name ="frame_00021"
object_path = os.path.join(objects_folder, image_name + '.png')
objects = Image.open(object_path) if os.path.exists(object_path) else None
gt_obj = objects
cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
print(gt_obj)

loss_obj = cls_criterion(logits.unsqueeze(0), gt_obj.unsqueeze(0)).squeeze().mean()

