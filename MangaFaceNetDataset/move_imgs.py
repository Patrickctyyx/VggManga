import os
import shutil
import random

src_dir = '/home/patrick/VggManga/MangaFaceNetDataset/ground truth'
target_dir = '/home/patrick/VggManga/MangaFaceNetDataset/facenetdata'

for dir_type in ['training', 'testing']:
    src_sub_dir = os.path.join(src_dir, dir_type)
    target_sub_dir = os.path.join(target_dir, dir_type)
    if not os.path.exists(os.path.join(target_sub_dir, 'have')):
        os.makedirs(os.path.join(target_sub_dir, 'have'))
    if not os.path.exists(os.path.join(target_sub_dir, 'not')):
        os.makedirs(os.path.join(target_sub_dir, 'not'))

    all_dirs = os.listdir(src_sub_dir)
    for single_dir in all_dirs:
        if single_dir.startswith('.'):
            continue

        print(dir_type, single_dir)
        single_src_dir_path = os.path.join(src_sub_dir, single_dir)

        have_imgs = os.listdir(os.path.join(single_src_dir_path, 'have'))
        not_imgs = os.listdir(os.path.join(single_src_dir_path, 'not'))
        min_num = min(len(have_imgs), len(not_imgs))

        random.shuffle(have_imgs)
        random.shuffle(not_imgs)

        for i in range(min_num):
            shutil.copy(os.path.join(single_src_dir_path, 'have', have_imgs[i]),
                        os.path.join(target_sub_dir, 'have', have_imgs[i]))
            shutil.copy(os.path.join(single_src_dir_path, 'not', not_imgs[i]),
                        os.path.join(target_sub_dir, 'not', not_imgs[i]))


