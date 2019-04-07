import os
import cv2
import selectivesearch
from utils import mkdir_if_not_exist


def apply_selective_search(manga109_path, target_path, threshold_pixels=2000, threshold_ratio=1.8):
    manga_dirs = os.listdir(os.path.join(manga109_path, 'images'))
    for single_manga in manga_dirs:
        if single_manga.startswith('.'):
            continue
        single_manga_path = os.path.join(manga109_path, 'images', single_manga)
        # to make it possible that many programs can be running at the same time
        if os.path.exists(os.path.join(target_path, single_manga)):
            continue
        mkdir_if_not_exist(os.path.join(target_path, single_manga))
        pages = os.listdir(single_manga_path)
        for page in pages:
            if page.startswith('.'):
                continue

            page_num = page.split('.')[0]

            # loading astronaut image
            img = cv2.imread(os.path.join(single_manga_path, page))

            # perform selective search
            img_lbl, regions = selectivesearch.selective_search(
                img, scale=500, sigma=0.9, min_size=10)

            candidates = set()
            for r in regions:
                # excluding same rectangle (with different segments)
                if r['rect'] in candidates:
                    continue
                # excluding regions smaller than 2000 pixels
                if r['size'] < threshold_pixels:
                    continue
                # distorted rects
                x, y, w, h = r['rect']
                if w / h > threshold_ratio or h / w > threshold_ratio:
                    continue
                candidates.add(r['rect'])

            # save sub region files
            i = 0
            for x, y, w, h in candidates:
                cv2.imwrite(os.path.join(target_path, single_manga,
                                         page_num + '_%04d.jpg' % i), img[y: y + h, x: x + w])
                i += 1

            return


if __name__ == '__main__':
    manga109_path = '/Users/patrick/Documents/datasets/Manga109_2017_09_28'
    target_path = '/Users/patrick/Desktop/VggManga/selective_search_imgs'
    apply_selective_search(manga109_path, target_path)
