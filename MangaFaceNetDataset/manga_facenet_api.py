import os
import json
import xmltodict


class Parser(object):
    def __init__(self, root_dir, book_titles="all"):
        """
        Manga FaceNet annotation parser
        Args:
            root_dir (str): The path of the root directory of Manga109 data, e.g., 'YOUR_PATH/Manga109_2017_09_28'
            book_titles (str or list): The book titles to be parsed.
                For example, if book_titles = ["ARMS", "AisazuNihaIrarenai"], these two books are read.
                The default value is "all", where all books are read
        """
        self.books = []  # book titles
        self.annotations = {
            'training': {},
            'testing': {}
        }  # annotation in the form of dict

        for book_name in os.listdir(os.path.join(root_dir, 'ground truth', 'training')):
            if book_name.startswith('.'):
                continue

            cur_path = os.path.join(root_dir, 'ground truth', 'training', book_name)

            if not os.path.isdir(cur_path):
                continue

            self.books.append(book_name)

        for book in self.books:
            if book_titles != "all" and book not in book_titles:
                continue

            cur_path = os.path.join(root_dir, 'ground truth', 'training', book)
            with open(os.path.join(cur_path, 'Ground_Truth_40.xml'), "rt", encoding='utf-8') as f:
                annotation = xmltodict.parse(f.read())
            annotation = json.loads(json.dumps(annotation))  # OrderedDict -> dict
            with open(os.path.join(cur_path, 'Ground_Truth_40_2.xml'), "rt", encoding='utf-8') as f:
                annotation2 = xmltodict.parse(f.read())
            annotation2 = json.loads(json.dumps(annotation2))  # OrderedDict -> dict
            if annotation2['Manga']['Face'] is not None:
                annotation['Manga']['Face']['item'].extend(annotation2['Manga']['Face']['item'])
            _convert_str_to_int_recursively(annotation)  # str -> int, for some attributes
            self.annotations['training'][book] = annotation

            cur_path = os.path.join(root_dir, 'ground truth', 'testing', book)
            with open(os.path.join(cur_path, 'Ground_Truth.xml'), "rt", encoding='utf-8') as f:
                annotation = xmltodict.parse(f.read())
            annotation = json.loads(json.dumps(annotation))  # OrderedDict -> dict
            _convert_str_to_int_recursively(annotation)  # str -> int, for some attributes
            self.annotations['testing'][book] = annotation


def _convert_str_to_int_recursively(annotation):
    """
    Given annotation data (nested list or dict), convert some attributes from string to integer.
    For example, {'Page': 'Raphael_018.jpg', 'X': 210, 'Y': 91, 'Width': 141, 'Height': 204} ->
    {'Page': 'Raphael_018.jpg', 'X': '210', 'Y': '91', 'Width': '141', 'Height': '204'}
    Args:
        annotation  (list or dict): Annotation date that consists of list or dict. Can be deeply nested.
    """
    item_list = annotation['Manga']['Face']['item']
    for i in range(len(item_list)):
        for item_type in ['Height', 'Width', 'X', 'Y']:
            item_list[i][item_type] = int(item_list[i][item_type])


if __name__ == '__main__':
    p = Parser('/Users/patrick/Desktop/VggManga/MangaFaceNetDataset')
    print(p.books)
    print(p.annotations['training']['Raphael'])
