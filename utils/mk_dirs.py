import os
from utils import mkdir_if_not_exist


def mk_manga_dirs(manga109_path, target_path):
    manga_dirs = os.listdir(os.path.join(manga109_path, 'images'))
    for single_manga in manga_dirs:
        if single_manga.startswith('.'):
            continue
        mkdir_if_not_exist(os.path.join(target_path, single_manga))


if __name__ == '__main__':
    manga109_path = '/Users/patrick/Documents/datasets/Manga109_2017_09_28'
    target_path = '/Users/patrick/Desktop/VggManga/selective_search_imgs'
    mk_manga_dirs(manga109_path, target_path)

