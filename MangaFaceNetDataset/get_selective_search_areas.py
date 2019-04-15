import os
import cv2
import shutil
import selectivesearch
from manga_facenet_api import Parser


# /Users/patrick/Desktop/VggManga/MangaFaceNetDataset/training/Arisa/have
def get_selective_search_areas(manga_path, threshold_pixels=2000, threshold_ratio=1.2):
    for dir_type in ['training', 'testing']:
        dir_path = os.path.join(manga_path, 'ground truth', dir_type)
        manga_dirs = os.listdir(dir_path)
        for single_manga in manga_dirs:
            if single_manga.startswith('.'):
                continue

            single_manga_path = os.path.join(dir_path, single_manga)
            if os.path.exists(os.path.join(single_manga_path, 'have')):
                continue

            parser = Parser(manga_path, book_titles=[single_manga])
            annotations = parser.annotations[dir_type][single_manga]['Manga']['Face']
            if annotations is None:
                annotations = []
            else:
                annotations = annotations['item']
            for manga_image in os.listdir(single_manga_path):
                if not manga_image.endswith('.jpg'):
                    continue

                manga_image_path = os.path.join(single_manga_path, manga_image)
                # loading astronaut image
                img = cv2.imread(manga_image_path)

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
                    if w / (h + 0.001) > threshold_ratio or h / (w + 0.001) > threshold_ratio:
                        continue

                    candidates.add(r['rect'])

                # save sub region files
                i = 0
                for x, y, w, h in candidates:
                    if is_region_a_face(x, y, w, h, annotations, manga_image):
                        # 加到 have 文件夹
                        mkdir_if_not_exist(os.path.join(single_manga_path, 'have'))
                        cv2.imwrite(os.path.join(single_manga_path, 'have', manga_image.split('.')[0] +
                                                 '_%d_%d_%d_%d.jpg' % (x, y, w, h)), img[y: y + h, x: x + w])
                    else:
                        # 加到 have 文件夹
                        mkdir_if_not_exist(os.path.join(single_manga_path, 'not'))
                        cv2.imwrite(os.path.join(single_manga_path, 'not', manga_image.split('.')[0] +
                                                 '_%d_%d_%d_%d.jpg' % (x, y, w, h)), img[y: y + h, x: x + w])

                    i += 1

                print(manga_image, 'finished!')


def is_region_a_face(x, y, w, h, item_list, page):
    for item in item_list:
        if item['Page'] != page:
            continue

        X = item['X']
        Y = item['Y']
        W = item['Width']
        H = item['Height']
        x_inter = min(x + w, X + W) - max(x, X)
        y_inter = min(y + h, Y + H) - max(y, Y)

        if x_inter <= 0 or y_inter <= 0:
            continue

        if x_inter * y_inter / (W * H) >= 0.7:
            return True

    return False


def mkdir_if_not_exist(dir_name, is_delete=False):
    """
    创建文件夹
    :param dir_name: 文件夹
    :param is_delete: 是否删除
    :return: 是否成功
    """
    try:
        if is_delete:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print('[Info] 文件夹 "%s" 存在, 删除文件夹.' % dir_name)

        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print('[Info] 文件夹 "%s" 不存在, 创建文件夹.' % dir_name)
        return True
    except Exception as e:
        print('[Exception] %s' % e)
        return False


if __name__ == '__main__':
    # get_selective_search_areas('/home/patrick/VggManga/MangaFaceNetDataset')
    get_selective_search_areas('/Users/patrick/Desktop/VggManga/MangaFaceNetDataset')
