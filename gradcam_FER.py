import os
import PIL
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
from matplotlib import pyplot as plt
from torchvision.utils import make_grid, save_image

from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp
from PIL import Image


def load_imgs(num, img_dir):
    images_tensor = []
    images_numpy = []
    file_names=[]

    files = os.listdir(img_dir)
    files = sorted(files)
    for i in range(num):
        file_name = files[i]
        file_path = os.path.join(img_dir, file_name)
        file_names.append(file_name)

        img = Image.open(file_path)
        img_numpy = np.asarray(img)
        rgb_np = np.expand_dims(img_numpy, axis=-1)
        rgb_np = np.concatenate((rgb_np, rgb_np, rgb_np), axis=-1)
        images_numpy.append(rgb_np)

        normalizer = Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        torch_img = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).float().div(255)#.cuda()
        normed_torch_img = normalizer(torch_img)
        images_tensor.append(normed_torch_img)

    return images_tensor, images_numpy, file_names

def creat_CAM_dict(type, arch, layer_name, input_size=(224, 224)):
    model_dict = dict(type=type, arch=arch, layer_name=layer_name, input_size=input_size)
    gradcam = GradCAM(model_dict, True)
    gradcampp = GradCAMpp(model_dict, True)
    cam_dict['resnet'+layer_name] = [gradcam, gradcampp]

    return cam_dict

cam_dict = dict()
num = 5
images_tensor, images_numpy, paths = load_imgs(num=num, img_dir='test/fear')


torch_imgs = [torch.from_numpy(image_numpy).permute(2, 0, 1).unsqueeze(0).float().div(255).expand(-1, 3, -1, -1) for image_numpy in images_numpy]


# resnet = models.resnet18(pretrained=True)
resnet = torch.load('models/model_res34.pth')
resnet.cpu().eval()
cam_dict = creat_CAM_dict(type='resnet', arch=resnet, layer_name='layer1', input_size=((320, 390)))
cam_dict = creat_CAM_dict(type='resnet', arch=resnet, layer_name='layer2', input_size=((320, 390)))
cam_dict = creat_CAM_dict(type='resnet', arch=resnet, layer_name='layer3', input_size=((320, 390)))
cam_dict = creat_CAM_dict(type='resnet', arch=resnet, layer_name='layer4', input_size=((320, 390)))

images = []
for normed_torch_img, torch_img in zip(images_tensor, torch_imgs):
    images.append(torch_img.squeeze().cpu())
    for gradcam, gradcam_pp in cam_dict.values():
        mask, _ = gradcam(normed_torch_img)
        heatmap, result = visualize_cam(mask.cpu(), torch_img)

        # mask, _ = gradcam_pp(normed_torch_img)
        # heatmap, result = visualize_cam(mask.cpu(), torch_img)

        # images.append(torch.stack([torch_img.squeeze().cpu(), heatmap, heatmap_pp, result, result_pp], 0))
        images.append(result)
images = [np.moveaxis(image.numpy(), 0, -1) for image in images]

fig, axes = plt.subplots(num, 5, figsize=(15, 15))

for i, image in enumerate(images):
    axes[i//5, i%5].imshow(image, interpolation='bicubic')
    if i%5!=0:
        axes[i//5, i%5].set_title(f'g{i%5-1}', fontsize=16)
    else:
        axes[i // 5, i % 5].set_title(paths[i//5], fontsize=16)
    axes[i//5, i%5].axis('off')
plt.tight_layout()
plt.savefig('img/fear_gradcam.png')
plt.show()
