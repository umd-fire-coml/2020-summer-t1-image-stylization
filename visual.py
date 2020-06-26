from __future__ import print_function
from module.photo_wct import PhotoWCT

import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.nn as nn
import torch

class ReMapping:
    def __init__(self):
        self.remapping = []

    def process(self, seg):
        new_seg = seg.copy()
        for k, v in self.remapping.items():
            new_seg[seg == k] = v
        return new_seg


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

def memory_limit_image_resize(cont_img):
    # prevent too small or too big images
    MINSIZE=256
    MAXSIZE=960
    orig_width = cont_img.width
    orig_height = cont_img.height
    if max(cont_img.width,cont_img.height) < MINSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((int(cont_img.width*1.0/cont_img.height*MINSIZE), MINSIZE), Image.BICUBIC)
        else:
            cont_img.thumbnail((MINSIZE, int(cont_img.height*1.0/cont_img.width*MINSIZE)), Image.BICUBIC)
    if min(cont_img.width,cont_img.height) > MAXSIZE:
        if cont_img.width > cont_img.height:
            cont_img.thumbnail((MAXSIZE, int(cont_img.height*1.0/cont_img.width*MAXSIZE)), Image.BICUBIC)
        else:
            cont_img.thumbnail(((int(cont_img.width*1.0/cont_img.height*MAXSIZE), MAXSIZE)), Image.BICUBIC)
    print("Resize image: (%d,%d)->(%d,%d)" % (orig_width, orig_height, cont_img.width, cont_img.height))
    return cont_img.width, cont_img.height


def stylization(stylization_module, smoothing_module, content_image_path, style_image_path, content_seg_path, style_seg_path, output_image_path,
                cont_seg_remapping=None, styl_seg_remapping=None):
    # Load image
    with torch.no_grad():
        cont_img = Image.open(content_image_path).convert('RGB')
        styl_img = Image.open(style_image_path).convert('RGB')

        new_cw, new_ch = memory_limit_image_resize(cont_img)
        new_sw, new_sh = memory_limit_image_resize(styl_img)
        cont_pilimg = cont_img.copy()
        cw = cont_pilimg.width
        ch = cont_pilimg.height
        try:
            cont_seg = Image.open(content_seg_path)
            styl_seg = Image.open(style_seg_path)
            cont_seg.resize((new_cw,new_ch),Image.NEAREST)
            styl_seg.resize((new_sw,new_sh),Image.NEAREST)

        except:
            cont_seg = []
            styl_seg = []

        cont_img = transforms.ToTensor()(cont_img).unsqueeze(0)
        styl_img = transforms.ToTensor()(styl_img).unsqueeze(0)

        # cont_img = Variable(cont_img, volatile=True)
        # styl_img = Variable(styl_img, volatile=True)

        cont_seg = np.asarray(cont_seg)
        styl_seg = np.asarray(styl_seg)
        if cont_seg_remapping is not None:
            cont_seg = cont_seg_remapping.process(cont_seg)
        if styl_seg_remapping is not None:
            styl_seg = styl_seg_remapping.process(styl_seg)

            
        with Timer("Elapsed time in stylization: %f"):
            stylized_img = stylization_module.transform(cont_img, styl_img, cont_seg, styl_seg)
        if ch != new_ch or cw != new_cw:
            print("De-resize image: (%d,%d)->(%d,%d)" %(new_cw,new_ch,cw,ch))
            stylized_img = nn.functional.upsample(stylized_img, size=(ch,cw), mode='bilinear')
        grid = utils.make_grid(stylized_img.data, nrow=1, padding=0)
        ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        out_img = Image.fromarray(ndarr)

        with Timer("Elapsed time in propagation: %f"):
            out_img = smoothing_module.process(out_img, cont_pilimg)

        out_img.save(output_image_path)

        
def show_image(path: str, title=''):
    import cv2
    import matplotlib.pyplot as plt
    import os
    
    img_ext = ['.jpg', '.jpeg', '.png']
    img_arr = []

    if os.path.isdir(path):
        for i in os.listdir(path):
            if str.lower(os.path.splitext(i)[-1]) in img_ext:
                img_path = path + i
                print(img_path)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_arr.append(img)

        x = 1
        for i in img_arr:
            t, ax = plt.subplots()
            ax.plot(len(img_arr), 1, x)
            ax.imshow(i)
            x += 1
    else:
        t, ax = plt.subplots()
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.plot(1, 1, 1)
        ax.set_title(title)
        ax.imshow(img)
    

def process_image(content: str, style: str, output: str):
    import os
    
    class StylizationException(Exception):
        def __init__(self, msg):
            self.msg = msg
    
    while True:
        try:
            if not os.path.isdir(output):
                raise StylizationException("output path is not a folder")
            
            img_ext = ['.jpg', '.jpeg', '.png']
            if not os.path.isdir(style) and not str.lower(os.path.splitext(style)[-1]) in img_ext:
                raise StylizationException("style image doesn't exist")
            
            # Load model
            p_wct = PhotoWCT()
            p_wct.load_state_dict(torch.load('./PhotoWCTModels/photo_wct.pth'))

            from photo_smooth import Propagator
            p_pro = Propagator()
            
            if os.path.isdir(content):
                has_image = False
                for i in os.listdir(content):
                    ext = str.lower(os.path.splitext(i)[-1])
                    if ext in img_ext:
                        has_image = True
                        stylization(
                            stylization_module=p_wct,
                            smoothing_module=p_pro,
                            content_image_path=f"{content}{i}",
                            style_image_path=style,
                            content_seg_path=[],
                            style_seg_path=[],
                            output_image_path=f"{output}processed_{i}",
                            )
                        show_image(f"{output}processed_{i}")
                if not has_image:
                    raise StylizationException("no image in content folder")
            else:
                if not str.lower(os.path.splitext(content)[-1]) in img_ext:
                    raise StylizationException("content image doesn't exist")
                stylization(
                    stylization_module=p_wct,
                    smoothing_module=p_pro,
                    content_image_path=content,
                    style_image_path=style,
                    content_seg_path=[],
                    style_seg_path=[],
                    output_image_path=f"{output}processed_{str.split(content, '/')[-1]}",
                    )
            show_image(output)             
            break
        except StylizationException as se:
            print(se.msg)

if __name__ =='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--content", default="./images/content/", 
                        help="the source of content images, can be a folder or a specific image")
    parser.add_argument("--style", default="./images/style/style.png", 
                        help="the source of style image, must be an image")
    parser.add_argument("--output", default="./results/", 
                        help="the output path of images, must be a folder")
    args = parser.parse_args()
    
    process_image(args.content, args.style, args.output)

    print('Done!')