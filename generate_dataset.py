import os
import cv2
import shutil
from utils.utils import mkdir_if_not_exist
from utils.get_annotation_dict import get_annotation_dict

manga109_dir = '/Users/patrick/Documents/datasets/Manga109_2017_09_28'
target_train_dir = '../manga109/train'
target_test_dir = '../manga109/test'
annotation_list = get_annotation_dict(return_list=True)

for anno in annotation_list:
    for cnt in range(anno[1]['@count']):
        origin_img_path = os.path.join(manga109_dir, 'images', anno[1]['@book'],
                                       '%03d.jpg' % anno[1]['elements'][cnt]['@page'])
        save_img_path = None
        if cnt % 5 == 0:  # test path
            save_img_path = os.path.join(target_test_dir, anno[0] + '_%03d.jpg' % cnt)
        else:
            save_img_path = os.path.join(target_train_dir, anno[0] + '_%03d.jpg' % cnt)
        img = cv2.imread(origin_img_path, 0)
        crop_image = img[anno[1]['elements'][cnt]['@ymin']: anno[1]['elements'][cnt]['@ymax'],
                         anno[1]['elements'][cnt]['@xmin']: anno[1]['elements'][cnt]['@xmax']]
        cv2.imwrite(save_img_path, crop_image)

img_dir = '../manga109'
for subdir in ['train', 'test']:
    dir_name = os.path.join(img_dir, subdir)
    imgs = os.listdir(dir_name)

    for img in imgs:
        if not img.endswith('.jpg'):
            continue
        img_sub_name = img.split('_')[0]
        mkdir_if_not_exist(os.path.join(dir_name, img_sub_name))
        shutil.move(os.path.join(dir_name, img), os.path.join(dir_name, img_sub_name, img))
