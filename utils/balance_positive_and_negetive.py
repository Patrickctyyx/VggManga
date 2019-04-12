import os
from PIL import Image

base_path = None


def img_cmp(img_path):
    global base_path
    img = Image.open(os.path.join(base_path, img_path))
    w, h = img.size
    return w * h


def balance(manga_dir):
    # remove images that are not so square
    for subdir in ['train', 'test', 'validation']:
        for image in os.listdir(os.path.join(manga_dir, subdir, 'have')):
            if image.startswith('.'):
                continue
            img = Image.open(os.path.join(manga_dir, subdir, 'have', image))
            (w, h) = img.size
            if w / h > 2 or h / w > 2:
                os.remove(os.path.join(manga_dir, subdir, 'have', image))

    # make negative and positive images have the same number
    for subdir in ['train', 'test', 'validation']:
        global base_path
        base_path = os.path.join(manga_dir, subdir, 'have')
        list_len = len(os.listdir(os.path.join(manga_dir, subdir, 'not')))
        img_list = os.listdir(os.path.join(manga_dir, subdir, 'have'))
        sorted(img_list, key=img_cmp, reverse=True)
        remove_list = img_list[list_len:]
        for rm_img in remove_list:
            os.remove(os.path.join(manga_dir, subdir, 'have', rm_img))


if __name__ == '__main__':
    manga_base_dir = '/home/patrick/VggManga/manga109_frame_'
    for dir_type in ['body', 'face']:
        manga_dir = manga_base_dir + dir_type
        balance(manga_dir)
        for subdir in ['train', 'test', 'validation']:
            print(len(os.listdir(os.path.join(manga_dir, subdir, 'not'))))
            print(len(os.listdir(os.path.join(manga_dir, subdir, 'have'))))
