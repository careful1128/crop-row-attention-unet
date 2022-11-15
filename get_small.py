import glob
import numpy as np
from PIL import Image
import shutil

imgs = glob.glob("test/masks/*")

for img in imgs:
    data = np.array(Image.open(img))
    if np.sum(data/255) < 20 and np.sum(data/255) > 10:
        name = img.split("\\")[-1]
        raw_fovs = f'test/fovs/{name}'
        raw_imgs = f'test/imgs/{name}'
        raw_masks = f'test/masks/{name}'
        new_fovs = f'small_target/fovs/{name}'
        new_imgs = f'small_target/imgs/{name}'
        new_masks = f'small_target/masks/{name}'
        shutil.copy(raw_fovs, new_fovs)
        shutil.copy(raw_imgs, new_imgs)
        shutil.copy(raw_masks, new_masks)