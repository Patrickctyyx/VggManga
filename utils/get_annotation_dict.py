import manga109api
from collections import OrderedDict


def get_annotation_dict(annotation_type='face', threshold_num=70, dict_len=200, 
                        manga109_dir='/Users/patrick/Documents/datasets/Manga109_2017_09_28',
                        return_list=False):
    """
        :param annotation_type: face/body
        :param threshold_num: minimum number of this annotation type appears in a book
        :param dict_len: the length of return dict
        :param manga109_dir: the dir of manga109 dataset
        :param return_list: whether the type of return value is a list

        :return:  the annotation ordered dict in this format
        {
            'id_of_the_element': {
                '@count': int,
                '@book': book_name,
                'elements': [
                    {
                        '@page': int,
                        '@xmax': int,
                        '@xmin': int,
                        '@ymax': int,
                        '@ymin': int
                    },
                    ...
                ]
            }, 
            ...
        }
    """

    p = manga109api.Parser(root_dir=manga109_dir)

    books = p.books
    all_characters = {}

    for book_name in books:
        book = p.annotations[book_name]['book']

        characters = book['characters']['character']
        #     print(characters)
        characters_count = {}
        for char in characters:
            characters_count[char['@id']] = {
                '@count': 0,
                '@book': book_name,
                'elements': []
            }

        pages = book['pages']['page']

        for i in range(len(pages)):
            page = pages[i]
            if annotation_type not in page:
                continue
            # 不止有一个 face
            if type(page[annotation_type]) is type([]):
                for face in page[annotation_type]:
                    temp_dict = dict()
                    temp_dict['@page'] = i
                    temp_dict['@xmax'] = face['@xmax']
                    temp_dict['@xmin'] = face['@xmin']
                    temp_dict['@ymax'] = face['@ymax']
                    temp_dict['@ymin'] = face['@ymin']
                    characters_count[face['@character']]['elements'].append(temp_dict)
                    characters_count[face['@character']]['@count'] += 1
            else:
                temp_dict = dict()
                temp_dict['@page'] = i
                temp_dict['@xmax'] = page['face']['@xmax']
                temp_dict['@xmin'] = page['face']['@xmin']
                temp_dict['@ymax'] = page['face']['@ymax']
                temp_dict['@ymin'] = page['face']['@ymin']
                characters_count[page['face']['@character']]['elements'].append(temp_dict)
                characters_count[page['face']['@character']]['@count'] += 1

        for key, val in characters_count.items():
            if val['@count'] >= threshold_num:
                all_characters[key] = val

    ordered_characters = OrderedDict()

    for key, val in all_characters.items():
        ordered_characters[key] = val

    ordered_characters = sorted(list(ordered_characters.items()), 
                                key=lambda x: x[1]['@count'], reverse=True)

    if return_list:
        return ordered_characters[: dict_len]
    
    return OrderedDict(ordered_characters[: dict_len])


if __name__ == '__main__':
    print(get_annotation_dict())
