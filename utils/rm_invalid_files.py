import os
from PIL import Image

for subdir in ['train', 'test', 'validation']:
    for with_or_not in ['have', 'not']:
        path = os.path.join('manga109_frame_body', subdir, with_or_not)
        imgs = os.listdir(path)
        for img in imgs:
            if img.startswith('.'):
                continue
            try:
                img_arr = Image.open(os.path.join(path, img))
                img_arr.load()
            except Exception as e:
                print('error file:', os.path.join(path, img))
                os.remove(os.path.join(path, img))
print('body finished')

for subdir in ['train', 'test', 'validation']:
    for with_or_not in ['have', 'not']:
        path = os.path.join('manga109_frame_face', subdir, with_or_not)
        imgs = os.listdir(path)
        for img in imgs:
            if img.startswith('.'):
                continue
            try:
                img_arr = Image.open(os.path.join(path, img))
                img_arr.load()
            except Exception as e:
                print('error file:', os.path.join(path, img))
                os.remove(os.path.join(path, img))
print('face finished')

for ele in ['face', 'body']:
    for subdir in ['train', 'test', 'validation']:
        for with_or_not in ['have', 'not']:
            path = os.path.join('manga109_frame_' + ele, subdir, with_or_not)
            print(path, len(os.listdir(path)))

