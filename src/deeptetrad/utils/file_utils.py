'''
Created on Mar 4, 2019

@author: vincent
'''

import os
import re
import glob
import pickle

def get_prefix_tokens(a_path):
    filename = os.path.split(a_path)[1]
    result = re.findall('-(\d+).jpg', filename)[0]
    filename = filename.replace('-1.jpg', '').replace('-2.jpg', '').replace('-3.jpg', '').replace('-4.jpg', '')
    tokens = filename.split(' ')
    exposure = tokens[-1]
    del tokens[-1]
    file_id = ' '.join(tokens)
    return (file_id, exposure, result)

def collect_a_path(root_path, verbose=None):
    if None is verbose:
        verbose = False
    prefix_pattern = '{}/*.jpg'.format(root_path)
    result_dict = {}
    for file_path in glob.iglob(prefix_pattern, recursive=True):
        tokens = get_prefix_tokens(file_path)
        (_, _, channel_id) = tokens
        if '1' == channel_id:
            result_dict[channel_id] = file_path
        elif '2' == channel_id:
            result_dict[channel_id] = file_path
        elif '3' == channel_id:
            result_dict[channel_id] = file_path
        elif '4' == channel_id:
            result_dict[channel_id] = file_path
    if verbose:
        print('[collect_a_prefix] {}'.format(result_dict))
    return result_dict

def get_all_prefixes(root_path):
    root_path = os.path.abspath(root_path)
    prefix_pattern = '{}/**/*.jpg'.format(root_path)
    prefixes = set()
    for a_path in glob.iglob(prefix_pattern, recursive=True):
        a_path = a_path.replace('1.jpg', '').replace('2.jpg', '').replace('3.jpg', '').replace('4.jpg', '')
        if '.jpg' in a_path:
            continue
        prefixes.add(a_path)
    return sorted(list(prefixes))

def is_valid_prefix(a_prefix):
    bright_path = '{}1.jpg'.format(a_prefix)
    red_path = '{}2.jpg'.format(a_prefix)
    green_path = '{}3.jpg'.format(a_prefix)
    blue_path = '{}4.jpg'.format(a_prefix)
    n_valid_count = -1
    if os.path.exists(bright_path):
        n_valid_count = n_valid_count + 1

    if os.path.exists(red_path):
        n_valid_count = n_valid_count + 1

    if os.path.exists(green_path):
        n_valid_count = n_valid_count + 1

    if os.path.exists(blue_path):
        n_valid_count = n_valid_count + 1
        
#     has_bright = True
#     has_red = True
#     has_green = True
#     has_blue = True
    
#     n_valid_count = 4
#     if not os.path.exists(bright_path):
#         has_bright = False
#         n_valid_count = n_valid_count - 1

#     if not os.path.exists(red_path):
#         has_red = False
#         n_valid_count = n_valid_count - 1
#     green_path = '{}3.jpg'.format(a_prefix)
#     if not os.path.exists(green_path):
#         has_green = False
#         n_valid_count = n_valid_count - 1
#     blue_path = '{}4.jpg'.format(a_prefix)
#     if not os.path.exists(blue_path):
#         has_blue = False
#         n_valid_count = n_valid_count - 1
    
    return n_valid_count

def get_path_list(a_prefix):
    bright_path = '{}1.jpg'.format(a_prefix)
    red_path = '{}2.jpg'.format(a_prefix)
    green_path = '{}3.jpg'.format(a_prefix)
    blue_path = '{}4.jpg'.format(a_prefix)
    return [bright_path, red_path, green_path, blue_path]

def get_path_dict(a_prefix):
    bright_path = '{}1.jpg'.format(a_prefix)
    red_path = '{}2.jpg'.format(a_prefix)
    green_path = '{}3.jpg'.format(a_prefix)
    blue_path = '{}4.jpg'.format(a_prefix)
    return {'1': bright_path, '2': red_path, '3': green_path, '4': blue_path}

def get_path_dict_two(a_prefix):
    bright_path = '{}1.jpg'.format(a_prefix)
    red_path = '{}2.jpg'.format(a_prefix)
    green_path = '{}3.jpg'.format(a_prefix)
    return {'1': bright_path, '2': red_path, '3': green_path}

def get_valid_prefixes(prefixes):
    valid_two_channel_prefixes = []
    valid_three_channel_prefixes = []
    for a_prefix in prefixes:
        if 2 == is_valid_prefix(a_prefix):
            valid_two_channel_prefixes.append(a_prefix)
        elif 3 == is_valid_prefix(a_prefix):
            valid_three_channel_prefixes.append(a_prefix)
    return valid_two_channel_prefixes, valid_three_channel_prefixes
        
def collect_a_prefix(root_path, a_pattern, exposures, verbose=None):
    if None is verbose:
        verbose = False
    prefix_pattern = '{}/*{}*.jpg'.format(root_path, a_pattern)
    result_dict = {}
    for file_path in glob.iglob(prefix_pattern, recursive=True):
        tokens = get_prefix_tokens(file_path)
        (_, exposure, channel_id) = tokens
        if '1' == channel_id and exposures[0] == int(exposure):
            result_dict[channel_id] = file_path
        elif '2' == channel_id and exposures[1] == int(exposure):
            result_dict[channel_id] = file_path
        elif '3' == channel_id and exposures[2] == int(exposure):
            result_dict[channel_id] = file_path
        elif '4' == channel_id and exposures[3] == int(exposure):
            result_dict[channel_id] = file_path
    if verbose:
        print('[collect_a_prefix] {}'.format(result_dict))
    return result_dict

def get_pickle_path(a_path, prefix):
    parent_path, filename = os.path.split(a_path)
    output_path = os.path.join(parent_path, 'test')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_filename = filename.replace('.jpg', '.{}.pickle'.format(prefix))
    output_path = '{}/{}'.format(output_path, output_filename)
    return output_path

def dump_data_to_pickle(a_path, a_data):
    with open(a_path, 'wb') as out_data:
        pickle.dump(a_data, out_data)
        
def load_data_from_pickle(a_path):
    with open(a_path, 'rb') as in_data:
        a_data = pickle.load(in_data)
    return a_data