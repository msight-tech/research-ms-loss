import os.path as osp
from PIL import Image


def read_image(img_path, mode='RGB'):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError(f"{img_path} does not exist")
    while not got_img:
        try:
            img = Image.open(img_path).convert("RGB")
            if mode == "BGR":
                r, g, b = img.split()
                img = Image.merge("RGB", (b, g, r))
            got_img = True
        except IOError:
            print(f"IOError incurred when reading '{img_path}'. Will redo.")
            pass
    return img
