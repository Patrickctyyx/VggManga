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
                temp_dict['@xmax'] = page[annotation_type]['@xmax']
                temp_dict['@xmin'] = page[annotation_type]['@xmin']
                temp_dict['@ymax'] = page[annotation_type]['@ymax']
                temp_dict['@ymin'] = page[annotation_type]['@ymin']
                characters_count[page[annotation_type]['@character']]['elements'].append(temp_dict)
                characters_count[page[annotation_type]['@character']]['@count'] += 1

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


def is_element_in_frame(frame, element):
    # 先严格按照在内部来，如果效果不好再加一个误差度
    return frame['@xmin'] <= element['@xmin'] and \
           frame['@xmax'] >= element['@xmax'] and \
           frame['@ymin'] <= element['@ymin'] and \
           frame['@ymax'] >= element['@ymax']


def get_frame_dict(manga109_dir='/Users/patrick/Documents/datasets/Manga109_2017_09_28'):
    p = manga109api.Parser(root_dir=manga109_dir)

    books = p.books
    all_frames = {}
    frame_without_face = 0
    frame_without_body = 0

    for book_name in books:
        book = p.annotations[book_name]['book']

        pages = book['pages']['page']

        for i in range(len(pages)):
            page = pages[i]
            frame_list = []
            face_list = []
            body_list = []

            if 'frame' not in page:
                continue
            # 统一为 list，方便后面的处理
            if type(page['frame']) is type([]):
                frame_list.extend(page['frame'])
            else:
                frame_list.append(page['frame'])

            if 'face' in page:
                if type(page['face']) is type([]):
                    face_list.extend(page['face'])
                else:
                    face_list.append(page['face'])
            if 'body' in page:
                if type(page['body']) is type([]):
                    body_list.extend(page['body'])
                else:
                    body_list.append(page['body'])

            for frame in frame_list:
                # 初始化每个 frame
                frame_dict = dict()
                frame_dict['@book'] = book_name
                frame_dict['@page'] = i
                frame_dict['@xmax'] = frame['@xmax']
                frame_dict['@xmin'] = frame['@xmin']
                frame_dict['@ymax'] = frame['@ymax']
                frame_dict['@ymin'] = frame['@ymin']
                frame_dict['@body_count'] = 0
                frame_dict['@face_count'] = 0
                frame_dict['body'] = []
                frame_dict['face'] = []

                # 添加到返回的结果中
                all_frames[frame['@id']] = frame_dict

                # 把 face 和 body 添加到对应的 frame 之中
            for frame in frame_list:
                # 将 face 添加到对应的 frame 中
                for face in face_list:
                    if is_element_in_frame(frame, face):
                        all_frames[frame['@id']]['@face_count'] += 1
                        all_frames[frame['@id']]['face'].append(face)

                # 将 body 添加到对应的 frame 中
                for body in body_list:
                    if is_element_in_frame(frame, body):
                        all_frames[frame['@id']]['@body_count'] += 1
                        all_frames[frame['@id']]['body'].append(face)

            for frame in frame_list:
                if all_frames[frame['@id']]['@face_count'] == 0:
                    frame_without_face += 1
                if all_frames[frame['@id']]['@body_count'] == 0:
                    frame_without_body += 1

    print('all frames count:', len(all_frames))
    print('frames without face count:', frame_without_face)
    print('frames without body count:', frame_without_body)
    return all_frames


if __name__ == '__main__':
    result = get_annotation_dict(annotation_type='body')
    print(len(result))
