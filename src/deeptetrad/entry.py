'''
Created on Mar 4, 2019

@author: vincent
'''

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
POLLEN_IMAGE_SINGLE_SIDE_SIZE = 256
TETRAD_IMAGE_SINGLE_SIDE_SIZE = 256

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) 
import warnings
warnings.filterwarnings("ignore")

from keras.backend.tensorflow_backend import get_session, clear_session,\
    set_session
from scipy.spatial.ckdtree import cKDTree
from scipy.spatial.kdtree import KDTree

from utils import *
from utils.file_utils import *
from utils.image_utils import *
from joblib import Parallel, delayed
import matplotlib
matplotlib.use('Agg')
from matplotlib import gridspec
import re

# from sklearn.neighbors import KDTree
import colorsys
from mrcnn import utils
from mrcnn import visualize

from mrcnn.config import Config
import mrcnn.model as modellib
import copy

from keras import backend as keras_backend
from matplotlib.colors import LinearSegmentedColormap
from functools import reduce
from collections import Counter
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
import gc

import numpy as np
import cv2

import urllib, shutil

maximum_object_count = 500
size_range_min, size_range_max = 10, 25
threshold_smoothing_scale = 1.3488
smoothing_filter_size = 10
suppress_diameter = 4
border_exclude = True
size_exclude = True

class PollenConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "pollen"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
#     BACKBONE = "resnet50"
    BACKBONE = "resnet101"
    
    # Input image resizing
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_MIN_SCALE = 1.0
    
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (2, 4, 8, 16, 32)
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([39.694053141276044, 39.332364298502604, 28.645670166015623])
#     MEAN_PIXEL = np.array([35.694053141276044, 35.332364298502604, 24.645670166015623])
#     MEAN_PIXEL = np.array([31.448479715983073, 31.64795110066732, 20.64955123901367])

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 8000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1000

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 1000
    
    LEARNING_RATE = 0.01

class TetradConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tetrad"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
#     NUM_CLASSES = 1 + 4  # Background + monad + dyad + triad + tetrad
    NUM_CLASSES = 1 + 2  # Background + non_tetrad + tetrad

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
#     BACKBONE = "resnet50"
    BACKBONE = "resnet101"

    # Input image resizing
    # Random crops of size 512x512
    IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    IMAGE_MIN_SCALE = 1.0
    
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 512
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([35.694053141276044, 35.332364298502604, 24.645670166015623])
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
#     USE_MINI_MASK = True
    USE_MINI_MASK = False
#     MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1000

class PollenInferenceConfig(PollenConfig):
    def __init__(self):
        if 64 == POLLEN_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 5
        elif 128 == POLLEN_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 21
        elif 256 == POLLEN_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 5
        elif 512 == POLLEN_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 5
        super(self.__class__, self).__init__()
    IMAGE_MIN_DIM = POLLEN_IMAGE_SINGLE_SIDE_SIZE
    IMAGE_MAX_DIM = POLLEN_IMAGE_SINGLE_SIDE_SIZE
    IMAGE_RESIZE_MODE = "pad64"
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0
#     DETECTION_MAX_INSTANCES = 150
    
class TetradInferenceConfig(TetradConfig):
    def __init__(self):
        if 64 == TETRAD_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 32
        elif 128 == TETRAD_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 32
        elif 256 == TETRAD_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 21
        elif 512 == TETRAD_IMAGE_SINGLE_SIDE_SIZE:
            self.IMAGES_PER_GPU = 21
        super(self.__class__, self).__init__()
    IMAGE_MIN_DIM = TETRAD_IMAGE_SINGLE_SIDE_SIZE
    IMAGE_MAX_DIM = TETRAD_IMAGE_SINGLE_SIDE_SIZE
    IMAGE_RESIZE_MODE = "pad64"
    GPU_COUNT = 1
    DETECTION_MIN_CONFIDENCE = 0
#     DETECTION_MAX_INSTANCES = 200

def get_simple_centroid(contour):
    temp_centroid = np.median(contour, axis=0)
    y = temp_centroid[0]
    x = temp_centroid[1]
    if np.isnan(x) or np.isnan(y):
        return (-1, -1)
    return (int(y), int(x))

def find_min_max_valid_pixels(contour, I, J):
    if 0 == contour.shape[0]:
        return []
    i_min = np.min(contour[:,0])
    i_max = np.max(contour[:,0])

    j_min = np.min(contour[:,1])
    j_max = np.max(contour[:,1])
    
    all_points = []
    for i, j in zip(contour[:,0], contour[:,1]):
        if i == i_min:
            all_points.append((i, j))
#             break
    for i, j in zip(contour[:,0], contour[:,1]):
        if i == i_max:
            all_points.append((i, j))
    
    for i, j in zip(contour[:,0], contour[:,1]):
        if j == j_min:
            all_points.append((i, j))
        
    for i, j in zip(contour[:,0], contour[:,1]):
        if j == j_max:
            all_points.append((i, j))
    return all_points

def find_min_max_valid_contour_pixels(contour, I, J):
    if 0 == contour.shape[0]:
        return []
    all_points = []
    
    i_min = np.min(I)
    i_max = np.max(I)
    j_min = np.min(J)
    j_max = np.max(J)
    for i, j in zip(I, J):
        if i == i_min:
            all_points.append((i, j))
        if i == i_max:
            all_points.append((i, j))
        if j == j_min:
            all_points.append((i, j))
        if j == j_max:
            all_points.append((i, j))
        
    return all_points

def align_all_images(a_prefix, is_parallel = None, is_purge = None, verbose=None):
    if None is is_parallel:
        is_parallel = False
    if None is verbose:
        verbose = False
    if verbose:
        print('[align_all_images] started')
    if None is is_purge:
        is_purge = False
    
    path_dict = get_path_dict(a_prefix)
    
    a_bright_path = path_dict['1']
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    
    bright_aligned_path = get_pickle_path(a_bright_path, 'aligned')
    red_aligned_path = get_pickle_path(a_red_path, 'aligned')
    green_aligned_path = get_pickle_path(a_green_path, 'aligned')
    blue_aligned_path = get_pickle_path(a_blue_path, 'aligned')
    if not os.path.exists(bright_aligned_path) or not os.path.exists(red_aligned_path) or \
    not os.path.exists(green_aligned_path) or not os.path.exists(blue_aligned_path) or is_purge:
        bright_image = cv2.imread(a_bright_path)
        red_image = cv2.imread(a_red_path)
        green_image = cv2.imread(a_green_path)
        blue_image = cv2.imread(a_blue_path)
        parameters = [(bright_image, (1.0, 1.0, 1.0)),
                      (blue_image, (1.0, 0, 0)),
                      (green_image, (0, 1.0, 0)),
                      (red_image, (0, 0, 1.0))]
        if is_parallel:
            gray_images = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(color_to_gray_from_image)(an_image, contributions) for an_image, contributions in parameters)
        else:
            gray_images = []
            for an_image, contributions in parameters:
                a_gray_image = color_to_gray_from_image(an_image, contributions)
                gray_images.append(a_gray_image)

    #     gray_images = []
        aligned_bright_tuple, aligned_blue_tuple, aligned_green_tuple, aligned_red_tuple = align_images_by_motion_translation(bright_image, [blue_image, green_image, red_image], gray_images, C_CROP)
        aligned_bright = aligned_bright_tuple[0]
        aligned_red = aligned_red_tuple[0]
        aligned_green = aligned_green_tuple[0]
        aligned_blue = aligned_blue_tuple[0]
    
        parameters = [(bright_aligned_path, aligned_bright), (red_aligned_path, aligned_red), (green_aligned_path, aligned_green), (blue_aligned_path, aligned_blue)]
        
        if is_parallel:
            Parallel(n_jobs=-1, backend="multiprocessing")(delayed(dump_data_to_pickle)(an_output_path, a_data) for an_output_path, a_data in parameters)
        else:
            for an_output_path, a_data in parameters:
                dump_data_to_pickle(an_output_path, a_data) 
        
    if verbose:
        print('[align_all_images] done')

def align_all_images_two(a_prefix, is_parallel = None, is_purge = None, verbose=None):
    if None is verbose:
        verbose = False
        
    if None is is_parallel:
        is_parallel = False
        
    path_dict = get_path_dict_two(a_prefix)
    if None is is_purge:
        is_purge = False
    a_bright_path = path_dict['1']
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    
    bright_aligned_path = get_pickle_path(a_bright_path, 'aligned')
    red_aligned_path = get_pickle_path(a_red_path, 'aligned')
    green_aligned_path = get_pickle_path(a_green_path, 'aligned')
    if not os.path.exists(bright_aligned_path) or not os.path.exists(red_aligned_path) or \
    not os.path.exists(green_aligned_path) or is_purge:
        bright_image = cv2.imread(a_bright_path)
        red_image = cv2.imread(a_red_path)
        green_image = cv2.imread(a_green_path)
        parameters = [(bright_image, (1.0, 1.0, 1.0)),
                      (green_image, (0, 1.0, 0)),
                      (red_image, (0, 0, 1.0))]
        gray_images = []
        if is_parallel:
            gray_images = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(color_to_gray_from_image)(an_image, contributions) for an_image, contributions in parameters)
        else:
            gray_images = []
            for an_image, contributions in parameters:
                gray_image = color_to_gray_from_image(an_image, contributions)
                gray_images.append(gray_image)
                
        aligned_bright_tuple, aligned_green_tuple, aligned_red_tuple = align_images_by_motion_translation(bright_image, [green_image, red_image], gray_images, C_CROP)
        aligned_bright = aligned_bright_tuple[0]
        aligned_red = aligned_red_tuple[0]
        aligned_green = aligned_green_tuple[0]
    
        parameters = [(bright_aligned_path, aligned_bright), (red_aligned_path, aligned_red), (green_aligned_path, aligned_green)]

        if is_parallel:    
            Parallel(n_jobs=-1, backend="multiprocessing")(delayed(dump_data_to_pickle)(an_output_path, a_data) for an_output_path, a_data in parameters)
        else:
            for an_output_path, a_data in parameters:
                dump_data_to_pickle(an_output_path, a_data) 
        
    if verbose:
        print('[align_all_images_two] done')

def create_suppressed_image(aligned_path, output_path, contributions):
    aligned_img = load_data_from_pickle(aligned_path)
    gray_img = color_to_gray_from_image(aligned_img, contributions)
    suppressed_image = suppress_by_radius(gray_img, suppress_diameter)
    dump_data_to_pickle(output_path, suppressed_image)
    
def create_suppressed_image_from_image(aligned_img, contributions):
    gray_img = color_to_gray_from_image(aligned_img, contributions)
    return suppress_by_radius(gray_img, suppress_diameter)

def get_all_centroids(object_path):
    object_image, object_count = load_data_from_pickle(object_path)
    centroids = []
    centroid_ids = {}
    for idx in range(1, object_count + 1):
        indices = np.argwhere(idx == object_image)
        centroid = get_simple_centroid(indices)
#         if centroid[0] < 0 or centroid[1] < 0:
#             continue
        centroid_ids[centroid] = idx
        centroids.append(centroid)
    return (centroid_ids, centroids)

def find_objects_two(path_dict, is_purge=None, verbose=None):
    if None is verbose:
        verbose = False
    if verbose:
        print('[find_objects] started')
    if None is is_purge:
        is_purge = False
    a_bright_path = path_dict['1']
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    bright_output_path = get_pickle_path(a_bright_path, 'objects')
    red_output_path = get_pickle_path(a_red_path, 'objects')
    green_output_path = get_pickle_path(a_green_path, 'objects')
    
    if not os.path.exists(bright_output_path) or not os.path.exists(red_output_path) or \
        not os.path.exists(green_output_path) or is_purge:
        
        bright_input_path = get_pickle_path(a_bright_path, 'aligned')
        red_input_path = get_pickle_path(a_red_path, 'aligned')
        green_input_path = get_pickle_path(a_green_path, 'aligned')
        parameters = [bright_input_path, red_input_path, green_input_path]
        [bright_image, green_image, red_image] = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(load_data_from_pickle)(an_input_path) for an_input_path in parameters)
        
        parameters = [(bright_image, (1.0, 1.0, 1.0)),
                      (green_image, (0, 1.0, 0)),
                      (red_image, (0, 0, 1.0))]
        gray_images = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(color_to_gray_from_image)(an_image, contributions) for an_image, contributions in parameters)
        
        object_results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(identify_primary_objects)(an_aligned, threshold_smoothing_scale, size_range_min, size_range_max, smoothing_filter_size, maximum_object_count, border_exclude, size_exclude) for an_aligned in gray_images)
        
        output_paths = [bright_output_path, red_output_path, green_output_path]
        parameters = zip(output_paths, object_results)
        
        Parallel(n_jobs=-1, backend="multiprocessing")(delayed(dump_data_to_pickle)(an_output_path, a_data) for an_output_path, a_data in parameters)
        
def find_objects(path_dict, is_purge=None, verbose=None):
    if None is verbose:
        verbose = False
    if verbose:
        print('[find_objects] started')
    if None is is_purge:
        is_purge = False
    a_bright_path = path_dict['1']
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    bright_output_path = get_pickle_path(a_bright_path, 'objects')
    red_output_path = get_pickle_path(a_red_path, 'objects')
    green_output_path = get_pickle_path(a_green_path, 'objects')
    blue_output_path = get_pickle_path(a_blue_path, 'objects')
    
    if not os.path.exists(bright_output_path) or not os.path.exists(red_output_path) or \
        not os.path.exists(green_output_path) or not os.path.exists(blue_output_path) or is_purge:
        bright_input_path = get_pickle_path(a_bright_path, 'gray')
        red_input_path = get_pickle_path(a_red_path, 'gray')
        green_input_path = get_pickle_path(a_green_path, 'gray')
        blue_input_path = get_pickle_path(a_blue_path, 'gray')
        parameters = [bright_input_path, red_input_path, green_input_path, blue_input_path]
        aligned_results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(load_data_from_pickle)(an_input_path) for an_input_path in parameters)
        
        object_results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(identify_primary_objects)(an_aligned, threshold_smoothing_scale, size_range_min, size_range_max, smoothing_filter_size, maximum_object_count, border_exclude, size_exclude) for an_aligned in aligned_results)
        output_paths = [bright_output_path, red_output_path, green_output_path, blue_output_path]
        parameters = zip(output_paths, object_results)
        
        Parallel(n_jobs=-1, backend="multiprocessing")(delayed(dump_data_to_pickle)(an_output_path, a_data) for an_output_path, a_data in parameters)
    if verbose:
        print('[find_objects] done')

def build_centroids(path_dict, is_purge=None, verbose=None):
    if None is verbose:
        verbose = False
    if verbose:
        print('[build_centroids] started')
    if None is is_purge:
        is_purge = False
    a_bright_path = path_dict['1']
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    bright_centroid_path = get_pickle_path(a_bright_path, 'centroids')
    red_centroid_path = get_pickle_path(a_red_path, 'centroids')
    green_centroid_path = get_pickle_path(a_green_path, 'centroids')
    blue_centroid_path = get_pickle_path(a_blue_path, 'centroids')
    
    if not os.path.exists(bright_centroid_path) or not os.path.exists(red_centroid_path) or \
    not os.path.exists(green_centroid_path) or not os.path.exists(blue_centroid_path) or is_purge:
        bright_object_path = get_pickle_path(a_bright_path, 'objects')
        red_object_path = get_pickle_path(a_red_path, 'objects')
        green_object_path = get_pickle_path(a_green_path, 'objects')
        blue_object_path = get_pickle_path(a_blue_path, 'objects')
        
        parameters = [bright_object_path, red_object_path, green_object_path, blue_object_path]
        # centroid_ids = {centroid coord, object_id}
        results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(get_all_centroids)(object_path) for object_path in parameters)
        # results = [(centroid_ids_bright, centroids_bright), (centroid_ids_red, centroids_red), (centroid_ids_green, centroids_green), (centroid_ids_blue, centroids_blue)]
        all_output_paths = [bright_centroid_path, red_centroid_path, green_centroid_path, blue_centroid_path]
        parameters = zip(all_output_paths, results)
        Parallel(n_jobs=-1, backend="multiprocessing")(delayed(dump_data_to_pickle)(an_output_path, a_data) for an_output_path, a_data in parameters)
    if verbose:
        print('[build_centroids] done')

def build_kd_trees(path_dict, is_purge = None, verbose=None):
    if None is verbose:
        verbose = False
    if verbose:
        print('[build_kd_trees] started')
    if None is is_purge:
        is_purge = True
 
    a_bright_path = path_dict['1']
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    bright_tree_path = get_pickle_path(a_bright_path, 'tree')
    red_tree_path = get_pickle_path(a_red_path, 'tree')
    green_tree_path = get_pickle_path(a_green_path, 'tree')
    blue_tree_path = get_pickle_path(a_blue_path, 'tree')
    if not os.path.exists(bright_tree_path) or not os.path.exists(red_tree_path) or \
        not os.path.exists(green_tree_path) or not os.path.exists(blue_tree_path) or is_purge:
        bright_centroid_path = get_pickle_path(a_bright_path, 'centroids')
        red_centroid_path = get_pickle_path(a_red_path, 'centroids')
        green_centroid_path = get_pickle_path(a_green_path, 'centroids')
        blue_centroid_path = get_pickle_path(a_blue_path, 'centroids')
        parameters = [bright_centroid_path, red_centroid_path, green_centroid_path, blue_centroid_path]
        centroid_results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(load_data_from_pickle)(a_path) for a_path in parameters)
#       centroid_results = [(centroid_ids_bright, centroids_bright), (centroid_ids_red, centroids_red), (centroid_ids_green, centroids_green), (centroid_ids_blue, centroids_blue)]
        tree_results = Parallel(n_jobs=4)(delayed(KDTree)(the_centroids, leaf_size = 2) for _, the_centroids in centroid_results)
#       tree_results = [bright_tree, red_tree, green_tree, blue_tree]
        all_output_paths = [bright_tree_path, red_tree_path, green_tree_path, blue_tree_path]
        parameters = zip(all_output_paths, tree_results)
        Parallel(n_jobs=4, backend="multiprocessing")(delayed(dump_data_to_pickle)(an_output_path, a_data) for an_output_path, a_data in parameters)
    if verbose:
        print('[build_kd_trees] done')

def remove_small_dots(r, thr):
    n_objects = r['masks'].shape[-1]
    for idx in range(n_objects):
        mask = r['masks'][:, :, idx]
        indices = np.argwhere(True == mask)
        if indices.shape[0] < thr:
            r['masks'][:, :, idx] = False
#         else:
#             print('[remove_small_dots] # area: {}'.format(indices.shape[0]))
    return r

def remove_large_dots(r, thr):
    n_objects = r['masks'].shape[-1]
    for idx in range(n_objects):
        mask = r['masks'][:, :, idx]
        indices = np.argwhere(True == mask)
        if len(indices) > thr:
            r['masks'][:, :, idx] = False
    return r

def remove_small_large_dots(r, thr_small, thr_large):
    n_objects = r['masks'].shape[-1]
    for idx in range(n_objects):
        mask = r['masks'][:, :, idx]
        indices = np.argwhere(True == mask)
        mask_area = len(indices) 
        if mask_area < thr_small or mask_area > thr_large:
            r['masks'][:, :, idx] = False
    return r

def remove_distanced_overlapped_pollens(r, thr):
    n_objects = r['masks'].shape[-1]
    centroids = []
    for i in range(n_objects):
        mask = r['masks'][:, :, i]
#                 I_mask, J_mask = np.non_zero(1 == mask)
        indices = np.argwhere(1 == mask)
        centroid = get_simple_centroid(indices)
        centroids.append(centroid)
    if len(centroids) < 3:
        return r
    k_tree = KDTree(centroids, leaf_size = 2)
    for idx in range(n_objects):
        mask = r['masks'][:, :, idx]
        indices = np.argwhere(1 == mask)
        a_centroid = get_simple_centroid(indices)
        dists, inds = k_tree.query([a_centroid], k=3)
        
        a_distance = dists[0][1]
        if a_distance > thr:
            r['masks'][:, :, idx] = False
        else:
            major_area_one = np.count_nonzero(mask)
            
            pollen_id_two = inds[0][1]
            close_mask_two = r['masks'][:, :, pollen_id_two]
            close_area_two = np.count_nonzero(close_mask_two)
            
            overlapped_mask_two = mask & close_mask_two
            overlapped_area_two = np.count_nonzero(overlapped_mask_two)
            
            pollen_id_three = inds[0][2]
            close_mask_three = r['masks'][:, :, pollen_id_three]
            close_area_three = np.count_nonzero(close_mask_two)
            
            overlapped_mask_three = mask & close_mask_three
            overlapped_area_three = np.count_nonzero(overlapped_mask_three)
            if overlapped_area_two > 0 and 0 == overlapped_area_three:
                if 0 == major_area_one:
                    overlapped_ratio_major = np.iinfo(np.int32).max
                else:
                    overlapped_ratio_major = overlapped_area_two / float(major_area_one)
                    
                if 0 == close_area_two:
                    overlapped_ratio_two = np.iinfo(np.int32).max
                else:
                    overlapped_ratio_two = overlapped_area_two / float(close_area_two)
#                 overlapped_ratio_major = overlapped_area_two / float(major_area_one)
#                 overlapped_ratio_two = overlapped_area_two / float(close_area_two)
                if overlapped_ratio_major > 0.3 or overlapped_ratio_two > 0.3:
                    if overlapped_ratio_major > overlapped_ratio_two:
                        r['masks'][:, :, idx] = False
                    else:
                        r['masks'][:, :, pollen_id_two] = False
            elif 0 < overlapped_area_two or 0 < overlapped_area_three:
                if 0 == major_area_one:
                    overlapped_ratio_major = np.iinfo(np.int32).max
                else:
                    overlapped_ratio_major = overlapped_area_two / float(major_area_one)
#                 overlapped_ratio_major = overlapped_area_two / float(major_area_one)
                if 0 == close_area_two:
                    overlapped_ratio_two = np.iinfo(np.int32).max
                else:
                    overlapped_ratio_two = overlapped_area_two / float(close_area_two)
                if 0 == close_area_three:
                    overlapped_ratio_three = np.iinfo(np.int32).max
                else:
                    overlapped_ratio_three = overlapped_area_three / float(close_area_three)
                if overlapped_ratio_major > 0.3 or overlapped_ratio_two > 0.3:
                    if overlapped_ratio_major > overlapped_ratio_two:
                        r['masks'][:, :, idx] = False
                    else:
                        r['masks'][:, :, pollen_id_two] = False
                if overlapped_ratio_major > 0.3 or overlapped_ratio_three > 0.3:
                    if overlapped_ratio_major > overlapped_ratio_three:
                        r['masks'][:, :, idx] = False
                    else:
                        r['masks'][:, :, pollen_id_three] = False
    return r

def remove_empty_masks(r):
    tmp_mask = r['masks']
    idx = np.flatnonzero((tmp_mask == 0).all((0, 1)))
    r['masks'] = np.delete(tmp_mask, idx, axis=2)
    r['class_ids'] = np.delete(r['class_ids'], idx)
    return r

def detect_tetrad_like_objects(r):
    n_objects = r['masks'].shape[-1]
    centroids = []
    for i in range(n_objects):
        mask = r['masks'][:, :, i]
#                 I_mask, J_mask = np.non_zero(1 == mask)
        indices = np.argwhere(1 == mask)
        centroid = get_simple_centroid(indices)
        centroids.append(centroid)
    if len(centroids) < 4:
        return r
    k_tree = KDTree(centroids, leaf_size = 2)
    selected_id = []
    for _, a_centroid in enumerate(centroids):
        dists, inds = k_tree.query([a_centroid], k=4)
        dists = dists[0][1:]
        inds = inds[0][1:]
        dist_sum = np.sum(dists)
        normalized_dist = dists / np.float(dist_sum)
        dist_min = np.min(normalized_dist)
        dist_max = np.max(normalized_dist)
        dist_delta = dist_max - dist_min
        if dist_delta < 0.23:
            selected_id.append(a_centroid)
            selected_id.extend(inds)
    selected_id_set = set(selected_id)
    for i in range(n_objects):
        if i not in selected_id_set:
            r['masks'][:, :, i] = False
    return r

def detect_left_right_most_centroids(centroids):
    left_most_centroid = (0, np.iinfo(np.int32).max)
    right_most_centroid = (0, 0)
    for a_centroid in centroids:
        if a_centroid[1] < left_most_centroid[1]:
            left_most_centroid = a_centroid
        if a_centroid[1] > right_most_centroid[1]:
            right_most_centroid = a_centroid
    return left_most_centroid, right_most_centroid

def detect_left_most_centroids(centroids):
    left_most_centroid = (np.iinfo(np.int32).max, 0)
    left_most_centroid_id = -1
    for a_centroid_id, a_centroid in enumerate(centroids):
        if a_centroid[0] < left_most_centroid[0]:
            left_most_centroid = a_centroid
            left_most_centroid_id = a_centroid_id
    return left_most_centroid, left_most_centroid_id

def detect_right_most_centroids(centroids):
    right_most_centroid = (0, 0)
    right_most_centroid_id = -1
    for a_centroid_id, a_centroid in enumerate(centroids):
        if a_centroid[0] > right_most_centroid[0]:
            right_most_centroid = a_centroid
            right_most_centroid_id = a_centroid_id
    return right_most_centroid, right_most_centroid_id

def collect_centroids_from_masks(r):
    centroids = []
    n_objects = r['masks'].shape[-1]
    for i in range(n_objects):
        mask = r['masks'][:, :, i]
        indices = np.argwhere(1 == mask)
        centroid = get_simple_centroid(indices)
        centroids.append(centroid)
    return centroids

def calculate_dist_delta(dists):
    dist_sum = np.sum(dists)
    normalized_dist = dists / np.float(dist_sum)
    dist_min = np.min(normalized_dist)
    dist_max = np.max(normalized_dist)
    dist_delta = dist_max - dist_min
    return normalized_dist, dist_delta

def detect_closest_centroids(initial_centroid, initial_centroid_id, centroids, k_tree, thr):
    print('[detect_closest_centroids] initial centroid: {}'.format(initial_centroid))
    current_centroids = [initial_centroid]
    current_centroid_ids = set([initial_centroid_id])
    
    while True:
        current_centroids_np = np.array(current_centroids)
        print('[detect_closest_centroids] current: {}, {}'.format(current_centroids_np, current_centroids_np.shape))
        a_centroid = get_simple_centroid(current_centroids_np)
        print('[detect_closest_centroids] current centroid: {}'.format(a_centroid))
        dists, inds = k_tree.query([a_centroid], k=len(centroids))
        for i in range(1, len(centroids)):
            dist = dists[0][i]
            ind = inds[0][i]
            if ind in current_centroid_ids:
                continue
            if dist < thr:
                break
            
        print('[detect_closest_centroids] dists: {}, inds: {}, dist: {}, ind: {}'.format(dists, inds, dist, ind))
        if dist > thr:
            break
        current_centroids.append(centroids[ind])
        current_centroid_ids.add(ind)
        if 4 == len(current_centroids):
            break
    return current_centroids, list(current_centroid_ids)

def select_pollen_and_tetrads(pollen_r, tetrad_r):
    copied_pollen_r = copy.deepcopy(pollen_r)
    copied_tetrad_r = copy.deepcopy(tetrad_r)
    
    n_tetrad_like_objects = copied_tetrad_r['masks'].shape[-1]
    n_pollens = copied_pollen_r['masks'].shape[-1]
#     print('[select_pollen_and_tetrads] (before) # tetrad-like objects: {}, # pollens: {}'.format(n_tetrad_like_objects, n_pollens))
    
    tetrad_pixels = []
    tetrad_pixel_ids = []
    for tetrad_id in range(n_tetrad_like_objects):
        tetrad_mask = copied_tetrad_r['masks'][:, :, tetrad_id]
#         tetrad_indices = np.argwhere(1 == tetrad_mask)
        I, J = np.nonzero(1 == tetrad_mask)
        for i, j in zip(I, J):
            tetrad_pixels.append((i, j))
            tetrad_pixel_ids.append(tetrad_id)
    
    k_tree = KDTree(tetrad_pixels, leaf_size = 2)
    
    number_of_pollens_in_tetrad_dict = {}
    for pollen_id in range(n_pollens):
        pollen_mask = copied_pollen_r['masks'][:, :, pollen_id]
        pollen_indices = np.argwhere(1 == pollen_mask)
        pollen_centroid = get_simple_centroid(pollen_indices)
        tetrad_dists, tetrad_inds = k_tree.query([pollen_centroid], k=1)
        tetrad_dist = tetrad_dists[0][0]
        # no overlapping tetrads available
        if 0 != tetrad_dist:
            continue
        tetrad_ind = tetrad_inds[0][0]
        actual_tetrad_id = tetrad_pixel_ids[tetrad_ind]
        if actual_tetrad_id not in number_of_pollens_in_tetrad_dict:
            number_of_pollens_in_tetrad_dict[actual_tetrad_id] = np.int32(0)
        number_of_pollens_in_tetrad_dict[actual_tetrad_id] += 1
    
    wrong_tetrad_ids = []
    for tetrad_id in range(n_tetrad_like_objects):
        if tetrad_id in number_of_pollens_in_tetrad_dict:
            n_pollens_in_this_tetrad = number_of_pollens_in_tetrad_dict[tetrad_id]
            if 4 == n_pollens_in_this_tetrad:
                continue
        wrong_tetrad_ids.append(tetrad_id)
    wrong_tetrad_ids = set(wrong_tetrad_ids)
    
    # remove irrelevant pollens
    for pollen_id in range(n_pollens):
        pollen_mask = copied_pollen_r['masks'][:, :, pollen_id]
        pollen_indices = np.argwhere(1 == pollen_mask)
        pollen_centroid = get_simple_centroid(pollen_indices)
        tetrad_dists, tetrad_inds = k_tree.query([pollen_centroid], k=1)
        tetrad_dist = tetrad_dists[0][0]
        # no overlapping tetrads available
        if 0 == tetrad_dist:
            tetrad_ind = tetrad_inds[0][0]
            actual_tetrad_id = tetrad_pixel_ids[tetrad_ind]
            if actual_tetrad_id in wrong_tetrad_ids:
                copied_pollen_r['masks'][:, :, pollen_id] = False
        else:
            copied_pollen_r['masks'][:, :, pollen_id] = False
    
    # remove irrelevant tetrads
    for tetrad_id in wrong_tetrad_ids:
#         print('[select_pollen_and_tetrads] wrong tetrad: {}'.format(tetrad_id))
        copied_tetrad_r['masks'][:, :, tetrad_id] = False
         
    copied_tetrad_r = remove_empty_masks(copied_tetrad_r)
    copied_pollen_r = remove_empty_masks(copied_pollen_r)
#     n_tetrad_like_objects = copied_tetrad_r['masks'].shape[-1]
#     n_pollens = copied_pollen_r['masks'].shape[-1]
    print('[select_pollen_and_tetrads] # pollens in tetrad : {}'.format(number_of_pollens_in_tetrad_dict))
#         I, J = np.nonzero(1 == tetrad_mask)
    return copied_pollen_r, copied_tetrad_r

def get_zoomed_mask(a_mask, mask_id, centroid, selected_color):
    image_h, image_w = a_mask.shape[0:2]
    I, J = np.nonzero(mask_id == a_mask)
    zoomed_mask = np.zeros((a_mask.shape[0], a_mask.shape[1], 3), dtype=np.uint8)
    zoomed_mask[I, J] = selected_color
    min_y = max(centroid[0] - 64, 0)
    max_y = min(centroid[0] + 64, image_h)
    min_x = max(centroid[1] - 64, 0)
    max_x = min(centroid[1] + 64, image_w)
    zoomed_mask = zoomed_mask[min_y:max_y,min_x:max_x,:]
    return zoomed_mask


def get_gray_zoomed_mask(a_mask, mask_id, centroid):
    image_h, image_w = a_mask.shape[0:2]
    I, J = np.nonzero(mask_id == a_mask)
    zoomed_mask = np.zeros((a_mask.shape[0], a_mask.shape[1]), dtype=np.uint8)
    zoomed_mask[I, J] = 255
    min_y = max(centroid[0] - 64, 0)
    max_y = min(centroid[0] + 64, image_h)
    min_x = max(centroid[1] - 64, 0)
    max_x = min(centroid[1] + 64, image_w)
    zoomed_mask = zoomed_mask[min_y:max_y,min_x:max_x]
    return zoomed_mask

def get_gray_zoom_and_enlarged_mask(a_mask, mask_id, centroid):
    image_h, image_w = a_mask.shape[0:2]
    I, J = np.nonzero(mask_id == a_mask)
    zoomed_mask = np.zeros((a_mask.shape[0], a_mask.shape[1]), dtype=np.uint8)
    I, J = get_shrinken_contour(I, J, 1, 1.5, (image_h, image_w))
    zoomed_mask[I, J] = 255
    min_y = max(centroid[0] - 64, 0)
    max_y = min(centroid[0] + 64, image_h)
    min_x = max(centroid[1] - 64, 0)
    max_x = min(centroid[1] + 64, image_w)
    zoomed_mask = zoomed_mask[min_y:max_y,min_x:max_x]
    kernel = np.ones((3, 3), np.uint8)
    zoomed_mask = cv2.morphologyEx(zoomed_mask, cv2.MORPH_CLOSE, kernel)
#     zoomed_mask = cv2.morphologyEx(zoomed_mask, cv2.MORPH_OPEN, kernel)
#     multiplier = 1.3
#     zoomed_mask = cv2.resize(zoomed_mask, (0,0), fx=multiplier, fy=multiplier)
#     min_y = max(int(centroid[0] * multiplier) - 64, 0)
#     max_y = min(int(centroid[0] * multiplier) + 64, image_h)
#     min_x = max(int(centroid[1] * multiplier) - 64, 0)
#     max_x = min(int(centroid[1] * multiplier) + 64, image_w)
#     print('[get_gray_zoom_and_enlarged_mask] min_y: {}, max_y: {}, min_x: {}, max_x: {}'.format(min_y, max_y, min_x, max_x))
        
    zoomed_mask[zoomed_mask > 0] = 255
    return zoomed_mask

def get_positioned_image(o_img, centroid, n_shape):
    copied_img = o_img.copy()
    image_h, image_w = copied_img.shape[0:2]
    min_y = max(centroid[0] - 128, 0)
    max_y = min(centroid[0], image_h)
    min_x = max(centroid[1] - 64, 0)
    max_x = min(centroid[1] + 64, image_w)

    cv2.rectangle(copied_img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 5)
    
    copied_img = cv2.resize(copied_img, n_shape)
    return copied_img

def try_to_get_additional_pollen(tetrad_id, tetrad_mask, pollen_ids, pollen_mask):
    tetrad_indices = np.argwhere(tetrad_id == tetrad_mask)
    tetrad_centroid = get_simple_centroid(tetrad_indices)
    zoomed_tetrad_mask = get_gray_zoomed_mask(tetrad_mask, tetrad_id, tetrad_centroid)
    for pollen_id in pollen_ids:
        zoomed_pollen_mask = get_gray_zoom_and_enlarged_mask(pollen_mask, pollen_id, tetrad_centroid)
#         print('[try_to_get_additional_pollen] tetrad_mask: {}, pollen_mask: {}'.format(zoomed_tetrad_mask.shape, zoomed_pollen_mask.shape))
        if zoomed_tetrad_mask.shape != zoomed_pollen_mask.shape:
            continue
        zoomed_tetrad_mask = cv2.subtract(zoomed_tetrad_mask, zoomed_pollen_mask)
    I, J = np.nonzero(0 < zoomed_tetrad_mask)
    if 0 != I.shape[0]:
#         I, J = get_shrinken_contour(I, J, 1, 0.90)
        zoomed_tetrad_mask[...] = 0
        zoomed_tetrad_mask[I, J] = 255
        kernel = np.ones((3, 3), np.uint8)
        zoomed_tetrad_mask = cv2.morphologyEx(zoomed_tetrad_mask, cv2.MORPH_OPEN, kernel)
    pollen_indices = np.argwhere(0 < zoomed_tetrad_mask)
    pollen_area = pollen_indices.shape[0]
    image_h, image_w = tetrad_mask.shape[0:2]
    min_y = max(tetrad_centroid[0] - 64, 0)
    max_y = min(tetrad_centroid[0] + 64, image_h)
    min_x = max(tetrad_centroid[1] - 64, 0)
    max_x = min(tetrad_centroid[1] + 64, image_w)
    return zoomed_tetrad_mask, pollen_area, (min_y, max_y, min_x, max_x)

def try_to_remove_extra_pollen(tetrad_id, tetrad_mask, pollen_ids, pollen_mask, allowed_min_area):
    tetrad_indices = np.argwhere(tetrad_id == tetrad_mask)
    tetrad_centroid = get_simple_centroid(tetrad_indices)
    zoomed_tetrad_mask = get_gray_zoomed_mask(tetrad_mask, tetrad_id, tetrad_centroid)
    pollen_areas = []
    pollen_centroids = []
    for pollen_id in pollen_ids:
        zoomed_pollen_mask = get_gray_zoom_and_enlarged_mask(pollen_mask, pollen_id, tetrad_centroid)
#         print('[try_to_get_additional_pollen] tetrad_mask: {}, pollen_mask: {}'.format(zoomed_tetrad_mask.shape, zoomed_pollen_mask.shape))
        if zoomed_tetrad_mask.shape != zoomed_pollen_mask.shape:
            continue
        zoomed_pollen_mask = cv2.bitwise_and(zoomed_tetrad_mask, zoomed_pollen_mask)
        pollen_indices = np.argwhere(0 < zoomed_pollen_mask)
        pollen_area = pollen_indices.shape[0]
        pollen_areas.append(pollen_area)
        pollen_centroid = get_simple_centroid(pollen_indices)
        pollen_centroids.append([pollen_centroid[0], pollen_centroid[1]])
    pollen_dists = euclidean_distances(pollen_centroids, pollen_centroids)
    pollen_dist_sum = np.sum(pollen_dists, axis=0)
    the_distanced_pollen_ids = np.nonzero(pollen_dist_sum == np.max(pollen_dist_sum))
    min_pollen_area = np.min(pollen_areas)
    smallest_pollen_ids = np.nonzero(pollen_areas == min_pollen_area)
    the_distanced_pollen_id = the_distanced_pollen_ids[0][-1]
    smallest_pollen_id = smallest_pollen_ids[0][-1]
    # remove recently added pollen to avoid removing the pollen predicted by deep tetrad
    if the_distanced_pollen_id == smallest_pollen_id:
        return smallest_pollen_id, min_pollen_area
    if allowed_min_area > min_pollen_area:
        return smallest_pollen_id, min_pollen_area
    if 4 == smallest_pollen_id:
        return the_distanced_pollen_id, pollen_areas[the_distanced_pollen_id]
    return -1, -1

# @timeit
def select_pollen_and_tetrads_from_combined_masks_alt(o_bright_image, pollen_mask, tetrad_mask):
    n_pollens = np.max(pollen_mask) + 1
    new_pollen_id = n_pollens
    n_tetrad_like_objects = np.max(tetrad_mask) + 1
    
    tetrad_pixels = []
    tetrad_pixel_ids = []
#
    pollen_areas = []
    for pollen_id in range(1, n_pollens):
        pollen_indices = np.argwhere(pollen_id == pollen_mask)
        pollen_area = pollen_indices.shape[0]
        pollen_areas.append(pollen_area)
    
#     min_per_pollen_areas = min(70, np.quantile(pollen_areas, 0.04))
    min_per_pollen_areas = np.quantile(pollen_areas, 0.04)
#     med_pollen_areas = np.quantile(pollen_areas, 0.5)
#     max_per_pollen_areas = min(300, np.quantile(pollen_areas, 0.90))
    max_per_pollen_areas = np.quantile(pollen_areas, 0.90)
#     if med_pollen_areas < 50:
#         min_per_pollen_areas = 70
#         med_pollen_areas = 90
#     print('[select_pollen_and_tetrads_from_combined_masks_alt] {}, area 4%: {:.2f}, median: {:.2f}, 90%: {:.2f}'.format(a_bright_prefix, min_per_pollen_areas, med_pollen_areas, max_per_pollen_areas))
    
    for tetrad_id in range(1, n_tetrad_like_objects):
        I, J = np.nonzero(tetrad_id == tetrad_mask)
        for i, j in zip(I, J):
            tetrad_pixels.append((i, j))
            tetrad_pixel_ids.append(tetrad_id)
     
    k_tree = cKDTree(tetrad_pixels, leafsize = 40)
    tetrad_to_pollen_dict = {}
    isolated_pollen_ids = []
    for pollen_id in range(1, n_pollens):
        pollen_indices = np.argwhere(pollen_id == pollen_mask)
        pollen_centroid = get_simple_centroid(pollen_indices)
        tetrad_dists, tetrad_inds = k_tree.query([pollen_centroid], k=1)
        tetrad_dist = tetrad_dists[0]
        # no overlapping tetrads available or the pollen is too small
        if 5 < tetrad_dist or pollen_indices.shape[0] < min_per_pollen_areas:
            isolated_pollen_ids.append(pollen_id)
            continue
        tetrad_ind = tetrad_inds[0]
        actual_tetrad_id = tetrad_pixel_ids[tetrad_ind]
        if actual_tetrad_id not in tetrad_to_pollen_dict:
            tetrad_to_pollen_dict[actual_tetrad_id] = []
        tetrad_to_pollen_dict[actual_tetrad_id].append(pollen_id)
    is_visualize_recover_three_pollen_tetrads = False
    pollen_max_distances = []
    an_empty_image = np.zeros((64, 64, 3), dtype=np.uint8)
    if is_visualize_recover_three_pollen_tetrads:
        target_positions = [(660, 30), (990, 30), (1290, 30), (660, 330), (990, 330), (1290, 330), (660, 660), (990, 660), (1290, 660)]
        cv2.namedWindow('pollen_mask_added')
        cv2.moveWindow('pollen_mask_added', 0, 510)
        cv2.namedWindow('zoomed_tetrad')
        cv2.moveWindow('zoomed_tetrad', target_positions[0][0], target_positions[0][1])
        for fig_id in range(6):
            cv2.namedWindow('zoomed_pollen_{}'.format(fig_id))
            cv2.moveWindow('zoomed_pollen_{}'.format(fig_id), target_positions[fig_id + 1][0], target_positions[fig_id + 1][1])
    
    n_recovered_tetrads = 0
    for tetrad_id in range(1, n_tetrad_like_objects):
        if tetrad_id in tetrad_to_pollen_dict:
            n_pollens_in_this_tetrad = len(tetrad_to_pollen_dict[tetrad_id])
            if 2 <= n_pollens_in_this_tetrad <= 4:
                zoomed_additional_pollen_mask, zoomed_additional_pollen_area, crop_pos = try_to_get_additional_pollen(tetrad_id, tetrad_mask, tetrad_to_pollen_dict[tetrad_id], pollen_mask)
                n_added = 0
                if zoomed_additional_pollen_area > 0:
                    _, contours, _ = cv2.findContours(zoomed_additional_pollen_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                    pollen_mask_disp = np.zeros(pollen_mask.shape[0:2])
                    for a_contour in contours:
                        n_mask = np.zeros_like(zoomed_additional_pollen_mask)
                        cv2.drawContours(n_mask, [a_contour], 0, 255, -1)
                        I, J = np.nonzero(n_mask > 0)
                        I += crop_pos[0]
                        J += crop_pos[2]
                        if min_per_pollen_areas < I.shape[0] < max_per_pollen_areas:
                            pollen_mask[I, J] = new_pollen_id
                            tetrad_to_pollen_dict[tetrad_id].append(new_pollen_id)
                            n_added = n_added + 1
                            pollen_mask_disp[I, J] = 255
                            new_pollen_id = new_pollen_id + 1
                    
                    if is_visualize_recover_three_pollen_tetrads:
                        pollen_mask_disp = cv2.resize(pollen_mask_disp, (640, 480))
                        cv2.imshow('pollen_mask_added', pollen_mask_disp)
                        print('[select_pollen_and_tetrads_from_combined_masks_alt] tetrad: {}, area: {}, crop_pos: {}, # new pollens: {}'.format(tetrad_id, zoomed_additional_pollen_area, crop_pos, n_added))
                        cv2.imshow('zoomed_pollen_5', zoomed_additional_pollen_mask)
                        tetrad_indices = np.argwhere(tetrad_id == tetrad_mask)
                        tetrad_centroid = get_simple_centroid(tetrad_indices)
                        zoomed_obright_image = get_positioned_image(o_bright_image, tetrad_centroid, (640, 480))
                        cv2.imshow('bright', zoomed_obright_image)
                        the_color = (0, 255, 255)
                        zoomed_tetrad_mask = get_zoomed_mask(tetrad_mask, tetrad_id, tetrad_centroid, the_color)
                        
                        cv2.imshow('zoomed_tetrad', zoomed_tetrad_mask)
                        for fig_id in range(5):
                            cv2.imshow('zoomed_pollen_{}'.format(fig_id), an_empty_image)
                        
                        for fig_id, pollen_id in enumerate(tetrad_to_pollen_dict[tetrad_id]):
                            zoomed_pollen_mask = get_zoomed_mask(pollen_mask, pollen_id, tetrad_centroid, the_color)
                            cv2.imshow('zoomed_pollen_{}'.format(fig_id), zoomed_pollen_mask)
            n_pollens_in_this_tetrad = len(tetrad_to_pollen_dict[tetrad_id])
            if 5 == n_pollens_in_this_tetrad:
                strange_pollen_local_id, removed_area = try_to_remove_extra_pollen(tetrad_id, tetrad_mask, tetrad_to_pollen_dict[tetrad_id], pollen_mask, min_per_pollen_areas)
                if -1 == strange_pollen_local_id:
                    continue
                del tetrad_to_pollen_dict[tetrad_id][strange_pollen_local_id]
                
                if is_visualize_recover_three_pollen_tetrads:
                    print('[select_pollen_and_tetrads_from_combined_masks_alt] tetrad: {}, area: {}, # removed pollens: 1'.format(tetrad_id, removed_area))
                    tetrad_indices = np.argwhere(tetrad_id == tetrad_mask)
                    tetrad_centroid = get_simple_centroid(tetrad_indices)
                    zoomed_obright_image = get_positioned_image(o_bright_image, tetrad_centroid, (640, 480))
                    cv2.imshow('bright', zoomed_obright_image)
                    the_color = (0, 255, 255)
                    zoomed_tetrad_mask = get_zoomed_mask(tetrad_mask, tetrad_id, tetrad_centroid, the_color)
                    cv2.imshow('zoomed_tetrad', zoomed_tetrad_mask)
                    for fig_id in range(5):
                        cv2.imshow('zoomed_pollen_{}'.format(fig_id), an_empty_image)
                    
                    for fig_id, pollen_id in enumerate(tetrad_to_pollen_dict[tetrad_id]):
                        zoomed_pollen_mask = get_zoomed_mask(pollen_mask, pollen_id, tetrad_centroid, the_color)
                        cv2.imshow('zoomed_pollen_{}'.format(fig_id), zoomed_pollen_mask)
            if 4 == len(tetrad_to_pollen_dict[tetrad_id]):
                if n_pollens_in_this_tetrad != 4:
                    n_recovered_tetrads = n_recovered_tetrads + 1
                local_pollen_centroids = []
                for pollen_id in tetrad_to_pollen_dict[tetrad_id]:
                    pollen_indices = np.argwhere(pollen_id == pollen_mask)
                    pollen_centroid = get_simple_centroid(pollen_indices)
                    local_pollen_centroids.append([pollen_centroid[0], pollen_centroid[1]])
                pollen_dists = euclidean_distances(local_pollen_centroids, local_pollen_centroids)
                local_max_per_pollen_dist = np.max(pollen_dists)
                pollen_max_distances.append(local_max_per_pollen_dist)
                if is_visualize_recover_three_pollen_tetrads:
                    print('[select_pollen_and_tetrads_from_combined_masks_alt] tetrad: {} seems to be a tetrad'.format(tetrad_id))
            if is_visualize_recover_three_pollen_tetrads:
                print('[select_pollen_and_tetrads_from_combined_masks_alt] tetrad: {}'.format(tetrad_id)) 
                tetrad_indices = np.argwhere(tetrad_id == tetrad_mask)
                tetrad_centroid = get_simple_centroid(tetrad_indices)
                zoomed_obright_image = get_positioned_image(o_bright_image, tetrad_centroid, (640, 480))
                cv2.imshow('bright', zoomed_obright_image)
                the_color = (0, 255, 255)
                zoomed_tetrad_mask = get_zoomed_mask(tetrad_mask, tetrad_id, tetrad_centroid, the_color)
                cv2.imshow('zoomed_tetrad', zoomed_tetrad_mask)
                for fig_id in range(5):
                    cv2.imshow('zoomed_pollen_{}'.format(fig_id), an_empty_image)
                
                for fig_id, pollen_id in enumerate(tetrad_to_pollen_dict[tetrad_id]):
                    zoomed_pollen_mask = get_zoomed_mask(pollen_mask, pollen_id, tetrad_centroid, the_color)
                    cv2.imshow('zoomed_pollen_{}'.format(fig_id), zoomed_pollen_mask)
                cv2.waitKey(0)
    
    pollen_median_distance = np.median(pollen_max_distances)
    valid_tetrad_dict = {}
    for tetrad_id in range(1, n_tetrad_like_objects):
        if tetrad_id in tetrad_to_pollen_dict:
            n_pollens_in_this_tetrad = len(tetrad_to_pollen_dict[tetrad_id])
            if 4 == n_pollens_in_this_tetrad:
                local_pollen_centroids = []
                for pollen_id in tetrad_to_pollen_dict[tetrad_id]:
                    pollen_indices = np.argwhere(pollen_id == pollen_mask)
                    pollen_centroid = get_simple_centroid(pollen_indices)
                    local_pollen_centroids.append([pollen_centroid[0], pollen_centroid[1]])
                pollen_dists = euclidean_distances(local_pollen_centroids, local_pollen_centroids)
                local_max_per_pollen_dist = np.max(pollen_dists)
                allowed_pollen_dist = 2*pollen_median_distance
                if local_max_per_pollen_dist < allowed_pollen_dist:
                    valid_tetrad_dict[tetrad_id] = tetrad_to_pollen_dict[tetrad_id]
            elif 3 == n_pollens_in_this_tetrad:
                valid_tetrad_dict[tetrad_id] = tetrad_to_pollen_dict[tetrad_id]
            elif 4 < n_pollens_in_this_tetrad:
                tetrad_indices = np.argwhere(tetrad_id == tetrad_mask)
                tetrad_centroid = get_simple_centroid(tetrad_indices)
                local_pollen_centroids = []
                local_distances = []
                local_areas = []
                for pollen_id in tetrad_to_pollen_dict[tetrad_id]:
                    pollen_indices = np.argwhere(pollen_id == pollen_mask)
                    pollen_centroid = get_simple_centroid(pollen_indices)
                    pollen_area = pollen_indices.shape[0]
                    local_pollen_centroids.append([pollen_centroid[0], pollen_centroid[1]])
                    pollen_dist = distance.euclidean(pollen_centroid, tetrad_centroid)
                    local_distances.append(pollen_dist)
                    local_areas.append(pollen_area)
                pollen_dists = euclidean_distances(local_pollen_centroids, local_pollen_centroids)
                pollen_dist_sum = np.sum(pollen_dists, axis=0)
                the_distanced_pollen_id = np.argwhere(pollen_dist_sum == np.max(pollen_dist_sum))[0][0]
                del tetrad_to_pollen_dict[tetrad_id][the_distanced_pollen_id]
                if 4 == len(tetrad_to_pollen_dict[tetrad_id]):
                    local_pollen_centroids = []
                    for pollen_id in tetrad_to_pollen_dict[tetrad_id]:
                        pollen_indices = np.argwhere(pollen_id == pollen_mask)
                        pollen_centroid = get_simple_centroid(pollen_indices)
                        local_pollen_centroids.append([pollen_centroid[0], pollen_centroid[1]])
                    pollen_dists = euclidean_distances(local_pollen_centroids, local_pollen_centroids)
                    local_max_per_pollen_dist = np.max(pollen_dists)
                    allowed_pollen_dist = 2*pollen_median_distance
                    if local_max_per_pollen_dist < allowed_pollen_dist:
                        valid_tetrad_dict[tetrad_id] = tetrad_to_pollen_dict[tetrad_id]
                        n_recovered_tetrads = n_recovered_tetrads + 1
                if is_visualize_recover_three_pollen_tetrads:
                    zoomed_obright_image = get_positioned_image(o_bright_image, tetrad_centroid, (640, 480))
                    cv2.imshow('bright', zoomed_obright_image)
                    the_color = (0, 255, 255)
                    zoomed_tetrad_mask = get_zoomed_mask(tetrad_mask, tetrad_id, tetrad_centroid, the_color)
                    cv2.imshow('zoomed_tetrad', zoomed_tetrad_mask)
                    for fig_id in range(5):
                        cv2.imshow('zoomed_pollen_{}'.format(fig_id), an_empty_image)
                        
                    for fig_id, pollen_id in enumerate(tetrad_to_pollen_dict[tetrad_id]):
                        zoomed_pollen_mask = get_zoomed_mask(pollen_mask, pollen_id, tetrad_centroid, the_color)
                        pollen_indices = np.argwhere(pollen_id == pollen_mask)
                        pollen_centroid = get_simple_centroid(pollen_indices)
                        pollen_dist = distance.euclidean(pollen_centroid, tetrad_centroid)
                        print('[select_pollen_and_tetrads_from_combined_masks_alt] fig_id: {}, dist: {}, area: {}'.format(fig_id, pollen_dist, pollen_indices.shape[0]))
                        cv2.imshow('zoomed_pollen_{}'.format(fig_id), zoomed_pollen_mask)
                    n_pollens_in_this_tetrad = len(tetrad_to_pollen_dict[tetrad_id])
                    print('[select_pollen_and_tetrads_from_combined_masks_alt] tetrad: {} has {} pollens.'.format(tetrad_id, n_pollens_in_this_tetrad))
                    cv2.waitKey(0)
    return valid_tetrad_dict, pollen_mask, new_pollen_id + 1

def base_round_up(x, base=512):
    return base * round(x/base)

def collect_pollen_mask_areas(x_min, y_min, mask):
    I_mask, J_mask = np.nonzero(True == mask)
    object_indices = np.argwhere(True == mask)
    object_indices[...,0] += x_min
    object_indices[...,1] += y_min
    I_mask += x_min
    J_mask += y_min
    area = I_mask.shape[0]
    return (object_indices, I_mask, J_mask, area)

def get_merged_mask(image, results, img_a_half_size):
    combined_mask = np.zeros(image.shape[0:2], dtype=np.uint32)
    h, w = image.shape[0:2]
    image_id = 0
    parameters = []
    for x_min in range(0, h - img_a_half_size, img_a_half_size):
        for y_min in range(0, w - img_a_half_size, img_a_half_size):
            cur_mask_dict = results[image_id]
            n_masks = cur_mask_dict['masks'].shape[-1]
            for local_mask_id in range(n_masks):
                mask = cur_mask_dict['masks'][:, :, local_mask_id].copy()
                parameters.append((x_min, y_min, mask))
            image_id = image_id + 1
    object_mask_areas = Parallel(n_jobs=-1, backend="threading")(delayed(collect_pollen_mask_areas)(x_min, y_min, mask) for x_min, y_min, mask in parameters)
    object_pixels = []
    object_ids = []
    for object_id, (object_indices, I, J, area) in enumerate(object_mask_areas):
        for i, j in zip(I, J):
            object_pixels.append((i, j))
            object_ids.append(object_id)
            
    k_tree = cKDTree(object_pixels, leafsize = 40)
    selected_object_ids = set()
    selected_object_id_additional_dict = {}
    
            
    # the fourth element in object_mask_areas[an_actual_id] is the area of an object
    visited = set()
    for object_id, (object_indices, I, J, area) in enumerate(object_mask_areas):
        if 0 == area:
            continue 
        if object_id in visited:
            continue
        valid_points = find_min_max_valid_contour_pixels(object_indices, I, J)
        all_adjacent_inds = set()
        for a_point in valid_points:
            object_inds = k_tree.query_ball_point([a_point], r=0)
            if len(object_inds[0]) > 0:
                actual_inds = np.unique([object_ids[a_pixel_id] for a_pixel_id in object_inds[0]])
                all_adjacent_inds.update(actual_inds)
        
        all_adjacent_inds = list(all_adjacent_inds)
        if len(all_adjacent_inds) > 0:
            actual_areas = np.array([object_mask_areas[an_actual_id][3] for an_actual_id in all_adjacent_inds])
            max_area = np.max(actual_areas)
            selected_id = np.nonzero(actual_areas == max_area)[0][0]
            actual_selected_id = all_adjacent_inds[selected_id]
            if actual_selected_id not in selected_object_id_additional_dict:
                selected_object_id_additional_dict[actual_selected_id] = set()
            selected_object_id_additional_dict[actual_selected_id].add(actual_selected_id)
            selected_object_ids.add(actual_selected_id)
            visited.update(actual_inds)
        else:
            selected_object_ids.add(object_id)
            if object_id not in selected_object_id_additional_dict:
                selected_object_id_additional_dict[object_id] = set()
            selected_object_id_additional_dict[object_id].add(object_id)
        
    for object_idx, selected_object_id in enumerate(selected_object_ids):
        the_n_id = object_idx + 1
        for local_object_id in selected_object_id_additional_dict[selected_object_id]:
            object_indices, I, J, area = object_mask_areas[local_object_id]
            combined_mask[I, J] = the_n_id
    return combined_mask, len(selected_object_ids), len(object_mask_areas)

def collect_all_single_pollens(a_prefix, pollen_model = None, is_purge=None):
    if None is pollen_model:
        pollen_model = load_pollen_prediction_model(21)
    path_dict = get_path_dict_two(a_prefix)
    if None is is_purge:
        is_purge = False
    a_bright_path = path_dict['1']

    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens')
    if os.path.exists(bright_single_pollen_output_path) and not is_purge:
        return
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    single_pollens = []
    o_bright_image = load_data_from_pickle(bright_output_path)

    h, w = o_bright_image.shape[:2]
    h = base_round_up(h, 512)
    w = base_round_up(w, 512)
    # h, w will be (2048, 2560) for a pollen picture of size (1920, 2560)
    o_bright_image = pad_image(o_bright_image, (h, w))
    h, w = o_bright_image.shape[:2]
    all_images = generate_batch_images(o_bright_image, pollen_model.config.IMAGES_PER_GPU, POLLEN_IMAGE_SINGLE_SIDE_SIZE)
    
    results = []
    n_batches = pollen_model.config.IMAGES_PER_GPU
    for batch_id in range(int(all_images.shape[0]/n_batches)):
        start_id = batch_id * n_batches
        end_id = min(all_images.shape[0], start_id + n_batches)
        selected_images = all_images[start_id:end_id,...]
        local_results = pollen_model.detect(selected_images, verbose=0)
        results.extend(local_results)

    for pollen_r in results:
        pollen_r = remove_small_dots(pollen_r, 1)
        single_pollens.append(pollen_r)
        
    img_a_side_size = POLLEN_IMAGE_SINGLE_SIDE_SIZE
    img_a_half_size = img_a_side_size >> 1    
    combined_pollen_mask, n_pollens, n_original_pollens = get_merged_mask(o_bright_image, single_pollens, img_a_half_size)
    print('[collect_all_single_pollens] {} combined mask: {}, # pollens: {}->{}'.format(
        a_prefix.split(os.sep)[-1], combined_pollen_mask.shape, n_original_pollens, n_pollens))
    dump_data_to_pickle(bright_single_pollen_output_path, (combined_pollen_mask, n_pollens))

def collect_all_tetrads(a_prefix, tetrad_model = None, is_purge=None):
    if None is tetrad_model:
        tetrad_model = load_tetrad_prediction_model()
    if None is is_purge:
        is_purge = False
    path_dict = get_path_dict_two(a_prefix)
    a_bright_path = path_dict['1']

    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    if os.path.exists(bright_tetrad_output_path) and not is_purge:
        return

    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    o_bright_image = load_data_from_pickle(bright_output_path)
    
    h, w = o_bright_image.shape[:2]
    h = base_round_up(h, 512)
    w = base_round_up(w, 512)
    # h, w will be (2048, 2560) for a pollen picture of size (1920, 2560)
    o_bright_image = pad_image(o_bright_image, (h, w))
    h, w = o_bright_image.shape[:2]
    all_images = generate_batch_images(o_bright_image, tetrad_model.config.IMAGES_PER_GPU, TETRAD_IMAGE_SINGLE_SIDE_SIZE)
    
    results = []
    n_batches = tetrad_model.config.IMAGES_PER_GPU
    for batch_id in range(int(all_images.shape[0]/n_batches)):
        start_id = batch_id * n_batches
        end_id = min(all_images.shape[0], start_id + n_batches)
        selected_images = all_images[start_id:end_id,...]
        local_results = tetrad_model.detect(selected_images, verbose=0)
        results.extend(local_results)
    tetrad_areas = []
    for tetrad_r in results:
        n_objects = tetrad_r['masks'].shape[-1]
        for tetrad_id in range(n_objects):
            tetrad_mask = tetrad_r['masks'][:, :, tetrad_id]
            indices = np.argwhere(True == tetrad_mask)
            tetrad_mask_area = indices.shape[0]
            tetrad_areas.append(tetrad_mask_area)
#     n_tetrad_like_objects = len(tetrad_areas)
#     if n_tetrad_like_objects > 1000:
#         min_thr = 0.05
#     else:
#         min_thr = 0.10
#     tetrad_areas = np.array(tetrad_areas)
#     min_per_tetrad_areas = np.quantile(tetrad_areas, min_thr)
#     med_tetrad_areas = np.quantile(tetrad_areas, 0.5)
#     max_per_tetrad_areas = np.quantile(tetrad_areas, 0.95)
#     if n_tetrad_like_objects > 1000:
#         print('[collect_all_tetrads] area 10%: {}, median: {}, 95%: {}'.format(min_per_tetrad_areas, med_tetrad_areas, max_per_tetrad_areas))
#     else:
#         print('[collect_all_tetrads] area 5%: {}, median: {}, 95%: {}'.format(min_per_tetrad_areas, med_tetrad_areas, max_per_tetrad_areas))
             
    tetrads = []
    for tetrad_r in results:
        tetrad_r = remove_small_dots(tetrad_r, 1)
        tetrad_r = remove_empty_masks(tetrad_r)
        tetrads.append(tetrad_r)
    img_a_side_size = TETRAD_IMAGE_SINGLE_SIDE_SIZE
    img_a_half_size = img_a_side_size >> 1
    combined_tetrad_mask, n_tetrads, n_original_tetrads = get_merged_mask(o_bright_image, tetrads, img_a_half_size)
    print('[collect_all_tetrads] {} combined mask: {}, # tetrads: {}->{}'.format(
        a_prefix.split(os.sep)[-1], combined_tetrad_mask.shape, n_original_tetrads, n_tetrads))
    
    dump_data_to_pickle(bright_tetrad_output_path, (combined_tetrad_mask, n_tetrads))        

def relabel_mask(mask):
    max_mask_id = np.max(mask) + 1
    n_prev_id = 1
    mask_id_change_lst = []

    for cur_mask_id in range(1, max_mask_id):
        I_prev_mask, J_prev_mask = np.nonzero(mask == cur_mask_id)
        if 0 != I_prev_mask.shape[0]:
            if cur_mask_id != n_prev_id:
                mask_id_change_lst.append((cur_mask_id, n_prev_id))
            n_prev_id = n_prev_id + 1
    for cur_mask_id, n_mask_id in mask_id_change_lst:
        I_prev_mask, J_prev_mask = np.nonzero(mask == cur_mask_id)
        mask[I_prev_mask, J_prev_mask] = n_mask_id
    return mask, n_mask_id + 1

def collect_all_valid_tetrads(a_prefix, root_path, debug_set, is_purge=None, is_capture=None, is_visualize=None):
    if None is is_purge:
        is_purge = False
    if None is is_capture:
        is_capture = False
    if None is is_visualize:
        is_visualize = True
    path_dict = get_path_dict(a_prefix)
    a_bright_path = path_dict['1']
    a_bright_prefix = a_bright_path.split(os.sep)[-1]
    if len(debug_set) > 0:
        is_in_debug_set = False
        for a_key in debug_set:
            if a_key in a_bright_path:
                is_in_debug_set = True
                break
        if not is_in_debug_set:
            return
    
    bright_tetrad_pollen_output_path = get_pickle_path(a_bright_path, 'tetrad_pollen')
    bright_single_pollen_revised_output_path = get_pickle_path(a_bright_path, 'single_pollens_revised')
    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    
    shorten_prefix = a_prefix.split(os.sep)[-1]
    if is_capture and not is_visualize:
        if not os.path.exists(bright_single_pollen_revised_output_path) or not os.path.exists(bright_tetrad_output_path) or \
            not os.path.exists(bright_tetrad_pollen_output_path):
            return
        pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_revised_output_path)
        tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
        valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
        fig_pollen, ax_pollen = plt.subplots()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1200, 30, 640, 480)
        fig_tetrad, ax_tetrad  = plt.subplots()
        plt.axis('off')
         
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1200, 540, 640, 480)
        valid_pollen_ids = set()
        valid_tetrad_ids = set()
        for tetrad_id, pollen_ids in valid_tetrad_dict.items():
            valid_tetrad_ids.add(tetrad_id)
            for pollen_id in pollen_ids:
                valid_pollen_ids.add(pollen_id)
         
        n_tetrads = np.max(tetrad_mask) + 1
        n_pollens = np.max(pollen_mask) + 1
        for tetrad_id in range(1, n_tetrads):
            if tetrad_id in valid_tetrad_ids:
                continue
            I, J = np.nonzero(tetrad_mask == tetrad_id)
            tetrad_mask[I, J] = 0
             
        for pollen_id in range(1, n_pollens):
            if pollen_id in valid_pollen_ids:
                continue
            I, J = np.nonzero(pollen_mask == pollen_id)
            pollen_mask[I, J] = 0
         
        relabeld_tetrad_mask, n_tetrads = relabel_mask(tetrad_mask)
        relabeld_pollen_mask, n_single_pollens = relabel_mask(pollen_mask)
        ax_pollen.cla()
        ax_tetrad.cla()
        pollen_colors = random_colors(n_single_pollens)
        tetrad_colors = random_colors(n_tetrads)
        pollen_colors[0] = (0.952941, 0.878431, 0.745098)
        tetrad_colors[0] = (0.952941, 0.878431, 0.745098)
        cm_pollen = LinearSegmentedColormap.from_list('pollen', pollen_colors, N=len(pollen_colors))
        cm_tetrad = LinearSegmentedColormap.from_list('tetrad', tetrad_colors, N=len(tetrad_colors))
        ax_pollen.imshow(relabeld_pollen_mask, cmap=cm_pollen)
        ax_tetrad.imshow(relabeld_tetrad_mask, cmap=cm_tetrad)
        print('[collect_all_valid_tetrads] {} # pollens: {}, # tetrads: {}, # valid tetrads: {}'.format(shorten_prefix, n_single_pollens, n_tetrads - 1, len(valid_tetrad_dict)))
        fig_pollen.tight_layout()
        fig_tetrad.tight_layout()
        
        last_prefix = a_prefix.split(os.sep)[-1]
        capture_dir = '{}/captures/{}'.format(root_path, last_prefix[0:-1])                
        if not os.path.exists(capture_dir):
            os.makedirs(capture_dir)
            
        all_tetrads_out_path = '{}/{}0_0_all_tetrads.pdf'.format(capture_dir, last_prefix)
        all_pollens_out_path = '{}/{}0_0_all_pollens.pdf'.format(capture_dir, last_prefix)
        fig_tetrad.savefig(all_tetrads_out_path, dpi=300, bbox_inches='tight', pad_inches=0)
        fig_pollen.savefig(all_pollens_out_path, dpi=300, bbox_inches='tight', pad_inches=0)
#         plt.show(block=True)
        return   
        
    if os.path.exists(bright_tetrad_pollen_output_path) and os.path.exists(bright_single_pollen_revised_output_path) and not is_purge and not is_visualize:
        return
    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens')
    
    if not os.path.exists(bright_single_pollen_output_path) or not os.path.exists(bright_tetrad_output_path):
        return
    
    single_pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_output_path)
    tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
    
    n_single_pollens = n_single_pollens + 1
    n_tetrads = n_tetrads + 1
    print('[collect_all_valid_tetrads] {} initial single pollen mask: {}, # pollens: {}({}), tetrad mask: {}, # tetrads: {}({})'.format(
        shorten_prefix, single_pollen_mask.shape, n_single_pollens, np.max(single_pollen_mask) + 1, tetrad_mask.shape, n_tetrads, np.max(tetrad_mask) + 1))
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    
    o_bright_image = load_data_from_pickle(bright_output_path)
     
    if is_visualize:
        fig_pollen, ax_pollen = plt.subplots()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1200, 30, 640, 480)
        fig_tetrad, ax_tetrad  = plt.subplots()
         
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1200, 540, 640, 480)
        n_tetrads = np.max(tetrad_mask) + 1
#         n_pollens = np.max(single_pollen_mask) + 1
        
#         n_single_pollens = np.unique(single_pollen_mask).shape[0]
#         n_tetrads = np.unique(tetrad_mask).shape[0]        
        ax_pollen.cla()
        ax_tetrad.cla()
        pollen_colors = random_colors(n_single_pollens)
        tetrad_colors = random_colors(n_tetrads)
        pollen_colors[0] = (1.0, 1.0, 1.0)
        tetrad_colors[0] = (1.0, 1.0, 1.0)
        cm_pollen = LinearSegmentedColormap.from_list('pollen', pollen_colors, N=len(pollen_colors))
        cm_tetrad = LinearSegmentedColormap.from_list('tetrad', tetrad_colors, N=len(tetrad_colors))
        ax_pollen.imshow(single_pollen_mask, cmap=cm_pollen)
        ax_tetrad.imshow(tetrad_mask, cmap=cm_tetrad)
        print('[collect_all_valid_tetrads] {} # pollens: {}, # tetrads: {}'.format(shorten_prefix, n_single_pollens, n_tetrads))
        fig_pollen.tight_layout()
        fig_tetrad.tight_layout()
#         plt.show(block=True)       
    if is_visualize:
        pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_revised_output_path)
        tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
        valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
    else:
        valid_tetrad_dict, pollen_mask, n_pollens = select_pollen_and_tetrads_from_combined_masks_alt(o_bright_image, single_pollen_mask, tetrad_mask)
        dump_data_to_pickle(bright_tetrad_pollen_output_path, valid_tetrad_dict)
        dump_data_to_pickle(bright_single_pollen_revised_output_path, (pollen_mask, n_pollens))
        
    if is_visualize:
        fig_pollen, ax_pollen = plt.subplots()
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1200, 30, 640, 480)
        fig_tetrad, ax_tetrad  = plt.subplots()
         
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1200, 540, 640, 480)
        valid_pollen_ids = set()
        valid_tetrad_ids = set()
        for tetrad_id, pollen_ids in valid_tetrad_dict.items():
            valid_tetrad_ids.add(tetrad_id)
            for pollen_id in pollen_ids:
                valid_pollen_ids.add(pollen_id)
         
        n_tetrads = np.max(tetrad_mask) + 1
        n_pollens = np.max(pollen_mask) + 1
        for tetrad_id in range(1, n_tetrads):
            if tetrad_id in valid_tetrad_ids:
                continue
            I, J = np.nonzero(tetrad_mask == tetrad_id)
            tetrad_mask[I, J] = 0
             
        for pollen_id in range(1, n_pollens):
            if pollen_id in valid_pollen_ids:
                continue
            I, J = np.nonzero(pollen_mask == pollen_id)
            pollen_mask[I, J] = 0
         
        relabeld_tetrad_mask, n_tetrads = relabel_mask(tetrad_mask)
        relabeld_pollen_mask, n_single_pollens = relabel_mask(pollen_mask)
#         n_single_pollens = np.unique(single_pollen_mask).shape[0]
#         n_tetrads = np.unique(tetrad_mask).shape[0]        
        ax_pollen.cla()
        ax_tetrad.cla()
        pollen_colors = random_colors(n_single_pollens)
        tetrad_colors = random_colors(n_tetrads)
        pollen_colors[0] = (1.0, 1.0, 1.0)
        tetrad_colors[0] = (1.0, 1.0, 1.0)
        cm_pollen = LinearSegmentedColormap.from_list('pollen', pollen_colors, N=len(pollen_colors))
        cm_tetrad = LinearSegmentedColormap.from_list('tetrad', tetrad_colors, N=len(tetrad_colors))
        ax_pollen.imshow(relabeld_pollen_mask, cmap=cm_pollen)
        ax_tetrad.imshow(relabeld_tetrad_mask, cmap=cm_tetrad)
        print('[collect_all_valid_tetrads] {} # pollens: {}, # tetrads: {}, # valid tetrads: {}'.format(shorten_prefix, n_single_pollens, n_tetrads, len(valid_tetrad_dict)))
        fig_pollen.tight_layout()
        fig_tetrad.tight_layout()
        plt.show(block=True)        

def get_padded_image_for_deep(an_img):
    h, w = an_img.shape[:2]
    h = base_round_up(h, 512)
    w = base_round_up(w, 512)
    # h, w will be (2048, 2560) for a pollen picture of size (1920, 2560)
    return pad_image(an_img, (h, w))

def is_valid_intensity(an_arr, intensity_thr):
    if len(an_arr) != 4:
        return False
#     if an_arr[2] > 1.2:
#         return False
    if an_arr[1] > 0 and abs(an_arr[1] - an_arr[2]) > intensity_thr:
        return True
    elif an_arr[1] > 0:
        return False
    elif an_arr[2] > 0:
        return True
    elif an_arr[2] < 0 and abs(an_arr[1] - an_arr[2]) > intensity_thr:
        return True
    return False

def validate_all_tetrad_types_two(a_prefix, path_dict, root_path, intensity_thr, debug_set, desired_min_area_ratio, is_purge=None, is_capture=None, is_visualize=None):
    if None is is_purge:
        is_purge = False
    if None is is_capture:
        is_capture = False
    if None is is_visualize:
        is_visualize = False
    a_bright_path = path_dict['1']
    
    bright_tetrad_pollen_output_path = get_pickle_path(a_bright_path, 'tetrad_pollen')
    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens_revised')
    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    bright_tetrad_typing_path = get_pickle_path(a_bright_path, 'tetrad_types_in_monad')
    if not os.path.exists(bright_tetrad_pollen_output_path) or not os.path.exists(bright_single_pollen_output_path) or not os.path.exists(bright_tetrad_output_path):
        return
    
    if os.path.exists(bright_tetrad_typing_path) and not is_purge and not is_capture:
        return load_data_from_pickle(bright_tetrad_typing_path)

    if len(debug_set) > 0:
        is_in_debug_set = False
        for a_key in debug_set:
            if a_key in a_bright_path:
                is_in_debug_set = True
                break
        if not is_in_debug_set:
            return load_data_from_pickle(bright_tetrad_typing_path)
    
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    red_output_path = get_pickle_path(a_red_path, 'aligned')
    green_output_path = get_pickle_path(a_green_path, 'aligned')
    
    o_bright_image = load_data_from_pickle(bright_output_path)
    o_red_image = load_data_from_pickle(red_output_path)
    o_green_image = load_data_from_pickle(green_output_path)
    
    o_bright_image = get_padded_image_for_deep(o_bright_image)
    o_red_image = get_padded_image_for_deep(o_red_image)
    o_green_image = get_padded_image_for_deep(o_green_image)
    
    
    single_pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_output_path)
    tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
    valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
    
    I_bg, J_bg = np.nonzero(single_pollen_mask == 0)
    bg_mean_bright = np.mean(o_bright_image[I_bg, J_bg, 0])
    bg_mean_green = np.mean(o_green_image[I_bg, J_bg, 0])
    bg_mean_red = np.mean(o_red_image[I_bg, J_bg, 0])
    
    tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}

    verbose = False
    if is_capture:
        is_visualize = True
    image_h, image_w = tetrad_mask.shape[0:2]
#     for debugging tetrad
    if is_visualize:
        all_tetrad_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
        fig_hist, ax_hist = plt.subplots(num=1)
        ax_hist.cla()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1000, 330, 640, 480)
    
    if verbose:
        bg_mean_intensities = np.array([bg_mean_bright, bg_mean_green, bg_mean_red], dtype=np.float64)
        print('[collect_all_tetrad_types_two] bg mean: {}'.format(bg_mean_intensities))

    is_include_dyad = True
    all_intensities_tuple_green, all_intensities_tuple_red = get_all_valid_intensity_statistics_two(valid_tetrad_dict, single_pollen_mask, o_green_image, o_red_image, desired_min_area_ratio, is_include_dyad)
    (all_intensities_dict_green, all_mean_green, all_std_green) = all_intensities_tuple_green 
    (all_intensities_dict_red, all_mean_red, all_std_red) = all_intensities_tuple_red 
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        if tetrad_id not in all_intensities_dict_green:
            continue

        local_intensities_green = all_intensities_dict_green[tetrad_id]
        local_intensities_red = all_intensities_dict_red[tetrad_id]
        
        standardized_green = (local_intensities_green - all_mean_green) / all_std_green
        standardized_red = (local_intensities_red - all_mean_red) / all_std_red
        sorted_intensities_green = np.sort(standardized_green)
        sorted_intensities_red = np.sort(standardized_red)
        
        if is_visualize:
                # normalize
            local_intensities_green = local_intensities_green/np.max(local_intensities_green)
            local_intensities_red = local_intensities_red/np.max(local_intensities_red)

        repr_auto_tetrad_type = 'Z'
        if (is_valid_intensity(sorted_intensities_green, 0) and is_valid_intensity(sorted_intensities_red, intensity_thr)):
            repr_auto_tetrad_type = 'N'
        
        if (not is_visualize or is_capture) and repr_auto_tetrad_type != 'N':
            continue
        
        if is_visualize:
            if repr_auto_tetrad_type != 'N' and not is_capture:
                print('========================= STRANGE ==============================')
                print('[collect_all_tetrad_types_two] tetrad: {}\ngreen: {}\nred: {}'.format(
                        tetrad_id, standardized_green, standardized_red))
                
            x = np.array(list(range(len(local_intensities_green))))
            ax_hist.cla()
            ax_hist.bar(x - 0.1, standardized_green, width=0.2, align='center', color='green')
            ax_hist.bar(x + 0.1, standardized_red, width=0.2, align='center', color='red')
            ax_hist.axhline(linewidth=0.5, y=0, color='black')
            ax_hist.set_ylim(-2, 2)
        
        median_green = np.median(standardized_green)
        median_red = np.median(standardized_red)
        green_on_ids = np.nonzero(standardized_green > median_green)[0]
        red_on_ids = np.nonzero(standardized_red > median_red)[0]
        gr_ids = np.intersect1d(green_on_ids, red_on_ids)
        current_type = 'Z'
        if repr_auto_tetrad_type == 'N':
            if 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 2 == gr_ids.shape[0]:
                current_type = 'P'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 1 == gr_ids.shape[0]:
                current_type = 'T'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 0 == gr_ids.shape[0]:
                current_type = 'NPD'
            else:
                current_type = 'Z'
                
        tetrad_type_dict[current_type] += 1
        
    manual_count_dict = {'G': np.int64(0), 'R': np.int64(0), 'GR': np.int64(0), 'None': np.int64(0)}
    manual_count_dict['GR'] += tetrad_type_dict['P'] * 2 + tetrad_type_dict['T']
    manual_count_dict['G'] += tetrad_type_dict['NPD'] * 2 + tetrad_type_dict['T']
    manual_count_dict['R'] += tetrad_type_dict['NPD'] * 2 + tetrad_type_dict['T']
    manual_count_dict['None'] += tetrad_type_dict['P'] * 2 + tetrad_type_dict['T']
    dump_data_to_pickle(bright_tetrad_typing_path, manual_count_dict)
    return manual_count_dict

def collect_all_tetrad_types_two(a_prefix, path_dict, root_path, intensity_thr, debug_set, desired_min_area_ratio, is_purge=None, is_capture=None, is_visualize=None):
    if None is is_purge:
        is_purge = False
    if None is is_capture:
        is_capture = False
    if None is is_visualize:
        is_visualize = False
    a_bright_path = path_dict['1']
    
    bright_tetrad_pollen_output_path = get_pickle_path(a_bright_path, 'tetrad_pollen')
    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens_revised')
    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    bright_tetrad_typing_path = get_pickle_path(a_bright_path, 'tetrad_types')
    if not os.path.exists(bright_tetrad_pollen_output_path) or not os.path.exists(bright_single_pollen_output_path) or not os.path.exists(bright_tetrad_output_path):
        return
    
    if os.path.exists(bright_tetrad_typing_path) and not is_purge and not is_capture:
        return load_data_from_pickle(bright_tetrad_typing_path)

    if len(debug_set) > 0:
        is_in_debug_set = False
        for a_key in debug_set:
            if a_key in a_bright_path:
                is_in_debug_set = True
                break
        if not is_in_debug_set:
            return load_data_from_pickle(bright_tetrad_typing_path)
    
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    red_output_path = get_pickle_path(a_red_path, 'aligned')
    green_output_path = get_pickle_path(a_green_path, 'aligned')
    
    o_bright_image = load_data_from_pickle(bright_output_path)
    o_red_image = load_data_from_pickle(red_output_path)
    o_green_image = load_data_from_pickle(green_output_path)
    
    o_bright_image = get_padded_image_for_deep(o_bright_image)
    o_red_image = get_padded_image_for_deep(o_red_image)
    o_green_image = get_padded_image_for_deep(o_green_image)
    
    single_pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_output_path)
    tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
    valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
    
    I_bg, J_bg = np.nonzero(single_pollen_mask == 0)
    bg_mean_bright = np.mean(o_bright_image[I_bg, J_bg, 0])
    bg_mean_green = np.mean(o_green_image[I_bg, J_bg, 0])
    bg_mean_red = np.mean(o_red_image[I_bg, J_bg, 0])
    
    tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}

    verbose = False
    if is_capture:
        is_visualize = True
    image_h, image_w = tetrad_mask.shape[0:2]
    if is_visualize:
        all_tetrad_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
        fig_hist, ax_hist = plt.subplots(num=1)
        ax_hist.cla()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1000, 330, 640, 480)
    
    if verbose:
        bg_mean_intensities = np.array([bg_mean_bright, bg_mean_green, bg_mean_red], dtype=np.float64)
        print('[collect_all_tetrad_types_two] bg mean: {}'.format(bg_mean_intensities))

    is_include_dyad = True
    all_intensities_tuple_green, all_intensities_tuple_red = get_all_valid_intensity_statistics_two(valid_tetrad_dict, single_pollen_mask, o_green_image, o_red_image, desired_min_area_ratio, is_include_dyad)
    (all_intensities_dict_green, all_mean_green, all_std_green) = all_intensities_tuple_green 
    (all_intensities_dict_red, all_mean_red, all_std_red) = all_intensities_tuple_red 

    pollen_colors = np.array(random_colors(4)) * 255
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        if tetrad_id not in all_intensities_dict_green:
            continue

        local_intensities_green = all_intensities_dict_green[tetrad_id]
        local_intensities_red = all_intensities_dict_red[tetrad_id]
        
        standardized_green = (local_intensities_green - all_mean_green) / all_std_green
        standardized_red = (local_intensities_red - all_mean_red) / all_std_red
        sorted_intensities_green = np.sort(standardized_green)
        sorted_intensities_red = np.sort(standardized_red)
        
        if is_visualize:
                # normalize
            local_intensities_green = local_intensities_green/np.max(local_intensities_green)
            local_intensities_red = local_intensities_red/np.max(local_intensities_red)

        repr_auto_tetrad_type = 'Z'
        if (is_valid_intensity(sorted_intensities_green, 0) and is_valid_intensity(sorted_intensities_red, intensity_thr)):
            repr_auto_tetrad_type = 'N'
        
            
        if (not is_visualize or is_capture) and repr_auto_tetrad_type != 'N':
            continue
        
        if is_visualize:
            if repr_auto_tetrad_type != 'N' and not is_capture:
                print('========================= STRANGE ==============================')
                print('[collect_all_tetrad_types_two] tetrad: {}\ngreen: {}\nred: {}'.format(
                        tetrad_id, standardized_green, standardized_red))
                
            x = np.array(list(range(len(local_intensities_green))))
            ax_hist.cla()
            ax_hist.bar(x - 0.1, standardized_green, width=0.2, align='center', color='green')
            ax_hist.bar(x + 0.1, standardized_red, width=0.2, align='center', color='red')
            ax_hist.axhline(linewidth=0.5, y=0, color='black')
            ax_hist.set_ylim(-2, 2)
        
        median_green = np.median(standardized_green)
        median_red = np.median(standardized_red)
        green_on_ids = np.nonzero(standardized_green > median_green)[0]
        red_on_ids = np.nonzero(standardized_red > median_red)[0]
        gr_ids = np.intersect1d(green_on_ids, red_on_ids)
        current_type = 'Z'
        if repr_auto_tetrad_type == 'N':
            if 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 2 == gr_ids.shape[0]:
                current_type = 'P'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 1 == gr_ids.shape[0]:
                current_type = 'T'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 0 == gr_ids.shape[0]:
                current_type = 'NPD'
            else:
                current_type = 'Z'
        
        tetrad_type_dict[current_type] += 1

        if is_visualize:
            if not is_capture:
                print('Tetrad ID: {}/{}'.format(tetrad_id, len(valid_tetrad_dict)))
                print('green: {}, red: {}, gr: {}'.format(green_on_ids, red_on_ids, gr_ids))
                print('[collect_all_tetrad_types_two] Auto Predicted: {}, {}'.format(repr_auto_tetrad_type, current_type))
            single_pollen_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
            tetrad_mask_subsampled = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1]), dtype=np.uint8)
     
            I, J = np.nonzero(tetrad_id == tetrad_mask)
            copied_all_tetrad_disp = all_tetrad_disp.copy()
            copied_all_tetrad_disp[I, J] = (0, 255, 255)
            copied_bright = o_bright_image.copy()
            copied_red = o_red_image.copy()
            copied_green = o_green_image.copy()
             
            for idx, pollen_id in enumerate(pollen_ids):
                I_pollen, J_pollen = np.nonzero(single_pollen_mask == pollen_id)
                single_pollen_disp[I_pollen, J_pollen] = pollen_colors[idx]
            tetrad_mask_subsampled[I, J] = 255
            min_y = max(np.min(I) - 64, 0)
            max_y = min(np.max(I) + 64, image_h)
            min_x = max(np.min(J) - 64, 0)
            max_x = min(np.max(J) + 64, image_w)
             
            copied_all_tetrad_disp = copied_all_tetrad_disp[min_y:max_y,min_x:max_x,:]
            copied_bright_resized = copied_bright.copy()
            copied_bright_resized = cv2.rectangle(copied_bright_resized, (min_x, min_y), (max_x, max_y), (0, 0, 255), 5)
            copied_bright_resized = cv2.resize(copied_bright_resized, (640, 480))
            copied_bright = copied_bright[min_y:max_y,min_x:max_x,...]
            copied_red = copied_red[min_y:max_y,min_x:max_x,...]
            copied_green = copied_green[min_y:max_y,min_x:max_x,...]
             
            combined_image = np.zeros_like(copied_bright)
            combined_image[...,1] = copied_green[...,1]
            combined_image[...,2] = copied_red[...,2]
             
            single_pollen_disp = single_pollen_disp[min_y:max_y,min_x:max_x]
            tetrad_mask_subsampled = tetrad_mask_subsampled[min_y:max_y,min_x:max_x]
             
            combined_image = cv2.bitwise_and(combined_image, combined_image, mask=tetrad_mask_subsampled)
            copied_red = cv2.bitwise_and(copied_red, copied_red, mask=tetrad_mask_subsampled)
            copied_green = cv2.bitwise_and(copied_green, copied_green, mask=tetrad_mask_subsampled)
            if not is_capture and visualize:
                cv2.imshow('all_pollens', copied_all_tetrad_disp)
                cv2.imshow('a_pollen', single_pollen_disp)
                cv2.imshow('bright', copied_bright)
                cv2.imshow('copied_bright_resized', copied_bright_resized)
                cv2.imshow('red', copied_red)
                cv2.imshow('green', copied_green)
                cv2.imshow('combined', combined_image)
            if verbose:
                print('[collect_all_tetrad_types_two] select the type of the tetrad (A-L)')
        if is_visualize:
            if not is_capture:
                examiner_key = chr(cv2.waitKey(0)).upper()
            else:
                last_prefix = a_prefix.split(os.sep)[-1]
                capture_dir = '{}/captures/{}/{}'.format(root_path, last_prefix[0:-1], current_type)                
                if not os.path.exists(capture_dir):
                    os.makedirs(capture_dir)
                    
                copied_bright_resized_out_path = '{}/{}{}_0_copied_bright_resized.png'.format(capture_dir, last_prefix, tetrad_id)
                tetrad_mask_out_path = '{}/{}{}_1_tetrad_mask.png'.format(capture_dir, last_prefix, tetrad_id)
                single_pollen_masks_out_path = '{}/{}{}_2_single_pollen_masks.png'.format(capture_dir, last_prefix, tetrad_id)
                copied_bright_out_path = '{}/{}{}_2_bright.png'.format(capture_dir, last_prefix, tetrad_id)
                copied_red_out_path = '{}/{}{}_3_red.png'.format(capture_dir, last_prefix, tetrad_id)
                copied_green_out_path = '{}/{}{}_4_green.png'.format(capture_dir, last_prefix, tetrad_id)
                copied_combined_out_path = '{}/{}{}_5_combined.png'.format(capture_dir, last_prefix, tetrad_id)
                predict_hist_out_path = '{}/{}{}_6_prediction_hist.pdf'.format(capture_dir, last_prefix, tetrad_id)
                
                cv2.imwrite(copied_bright_resized_out_path, copied_bright_resized)
                cv2.imwrite(tetrad_mask_out_path, copied_all_tetrad_disp)
                cv2.imwrite(single_pollen_masks_out_path, single_pollen_disp)
                cv2.imwrite(copied_bright_out_path, copied_bright)
                cv2.imwrite(copied_red_out_path, copied_red)
                cv2.imwrite(copied_green_out_path, copied_green)
                cv2.imwrite(copied_combined_out_path, combined_image)
                
                fig_hist.savefig(predict_hist_out_path, edgecolor='none', bbox_inches='tight', pad_inches=0, dpi=300)
                
    
    
    ground_truth_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}
    if not is_capture:
        dump_data_to_pickle(bright_tetrad_typing_path, (tetrad_type_dict, ground_truth_tetrad_type_dict))
    return (tetrad_type_dict, ground_truth_tetrad_type_dict)

def get_channel_order_dict(order_str):
    order_rev_dict = {'R': 0, 'G': 1, 'B': 2,
                         'r': 0, 'g': 1, 'b': 2,
                         'Y': 1, 'C': 2,
                         'y': 1, 'c': 2}
    order_for_dict = {0: ['R', 'r'], 1: ['G', 'g', 'Y', 'y'], 2: ['C', 'c', 'B', 'b']}
    
    order_dict = {}
    for idx, cur_channel in enumerate(order_str):
        cur_channel = cur_channel.upper()
        if 'C' == cur_channel:
            rep_channel = 'C'
        elif 'B' == cur_channel:
            rep_channel = 'C'
        elif 'G' == cur_channel:
            rep_channel = 'G'
        elif 'Y' == cur_channel:
            rep_channel = 'G'
        elif 'R' == cur_channel:
            rep_channel = 'R'
         
        rep_channel_upper = rep_channel.upper()

        for a_channel_chr in order_for_dict[idx]:
            order_dict[a_channel_chr] = rep_channel_upper
        
    return order_dict, order_rev_dict

def get_converted_interval_str(order_dict, a_str):
    result_str = ''
    for a_chr in a_str:
        result_str = result_str + order_dict[a_chr]
    return get_representative_interval_str(result_str)

def get_representative_interval_str(a_str):
    # RG, RC, GC
    if 'CR' == a_str:
        return 'RC'
    elif 'GR' == a_str:
        return 'RG' 
    elif 'CG' == a_str:
        return 'GC'
    return a_str

def get_all_valid_intensity_statistics_two(valid_tetrad_dict, single_pollen_mask, o_green_image, o_red_image, desired_min_area_ratio, is_include_dyad = None):
    if None is is_include_dyad:
        is_include_dyad = False
    all_intensities_green = []
    all_intensities_red = []
    all_intensities_dict_green = {}
    all_intensities_dict_red = {}
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        local_pollen_areas = []
        if is_include_dyad:
            if len(pollen_ids) > 4 or len(pollen_ids) < 3:
                continue
        else:
            if len(pollen_ids) != 4:
                continue
        for pollen_id in pollen_ids:
            I, J = np.nonzero(single_pollen_mask == pollen_id)
            local_pollen_areas.append(I.shape[0])
            
        min_area = np.min(local_pollen_areas)
        max_area = np.max(local_pollen_areas)
        min_max_area_ratio = max_area / float(min_area)
        if min_max_area_ratio > desired_min_area_ratio:
            continue
        local_intensities_green = []
        local_intensities_red = []
        for pollen_id in pollen_ids:
            I, J = np.nonzero(single_pollen_mask == pollen_id)
            intensity_green = np.mean(o_green_image[I, J, 1])
            intensity_red = np.mean(o_red_image[I, J, 2])
            all_intensities_green.append(intensity_green)
            all_intensities_red.append(intensity_red)
            
            local_intensities_green.append(intensity_green)
            local_intensities_red.append(intensity_red)
        if 3 == len(pollen_ids):
            local_intensities_green.append(0)
            local_intensities_red.append(0)
            
            all_intensities_green.append(0)
            all_intensities_red.append(0)
        all_intensities_dict_green[tetrad_id] = local_intensities_green
        all_intensities_dict_red[tetrad_id] = local_intensities_red

    all_intensities_green = np.array(all_intensities_green, dtype=np.float64)
    all_mean_green = np.mean(all_intensities_green)
    all_std_green = np.std(all_intensities_green)
    all_intensities_tuple_green = (all_intensities_dict_green, all_mean_green, all_std_green)
    
    all_intensities_red = np.array(all_intensities_red, dtype=np.float64)
    all_mean_red = np.mean(all_intensities_red)
    all_std_red = np.std(all_intensities_red)
    
    all_intensities_tuple_red = (all_intensities_dict_red, all_mean_red, all_std_red)
    return all_intensities_tuple_green, all_intensities_tuple_red


def get_all_valid_intensity_statistics(valid_tetrad_dict, single_pollen_mask, o_blue_image, o_green_image, o_red_image, desired_min_area_ratio, is_include_dyad=None):
    if None is is_include_dyad:
        is_include_dyad = False
    all_intensities_blue = []
    all_intensities_green = []
    all_intensities_red = []
    all_intensities_dict_blue = {}
    all_intensities_dict_green = {}
    all_intensities_dict_red = {}
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        local_pollen_areas = []
        if is_include_dyad:
            if len(pollen_ids) > 4 or len(pollen_ids) < 3:
                continue
        else:
            if len(pollen_ids) != 4:
                continue
        for pollen_id in pollen_ids:
            I, J = np.nonzero(single_pollen_mask == pollen_id)
            local_pollen_areas.append(I.shape[0])
            
        min_area = np.min(local_pollen_areas)
        max_area = np.max(local_pollen_areas)
        min_max_area_ratio = max_area / float(min_area)
        if min_max_area_ratio > desired_min_area_ratio:
            continue
        local_intensities_blue = []
        local_intensities_green = []
        local_intensities_red = []
        for pollen_id in pollen_ids:
#             I, J = np.nonzero((single_pollen_mask == pollen_id) & (tetrad_mask == tetrad_id))
            I, J = np.nonzero(single_pollen_mask == pollen_id)
            intensity_blue = np.mean(o_blue_image[I, J, 0])
            intensity_green = np.mean(o_green_image[I, J, 1])
            intensity_red = np.mean(o_red_image[I, J, 2])
            all_intensities_blue.append(intensity_blue)
            all_intensities_green.append(intensity_green)
            all_intensities_red.append(intensity_red)
            
            local_intensities_blue.append(intensity_blue)
            local_intensities_green.append(intensity_green)
            local_intensities_red.append(intensity_red)
        if 3 == len(pollen_ids):
            local_intensities_blue.append(0)
            local_intensities_green.append(0)
            local_intensities_red.append(0)
            
            all_intensities_blue.append(0)
            all_intensities_green.append(0)
            all_intensities_red.append(0)
            
        all_intensities_dict_blue[tetrad_id] = local_intensities_blue
        all_intensities_dict_green[tetrad_id] = local_intensities_green
        all_intensities_dict_red[tetrad_id] = local_intensities_red

    all_intensities_blue = np.array(all_intensities_blue, dtype=np.float64)
    all_mean_blue = np.mean(all_intensities_blue)
    all_std_blue = np.std(all_intensities_blue)
    all_intensities_tuple_blue = (all_intensities_dict_blue, all_mean_blue, all_std_blue)
    
    all_intensities_green = np.array(all_intensities_green, dtype=np.float64)
    all_mean_green = np.mean(all_intensities_green)
    all_std_green = np.std(all_intensities_green)
    all_intensities_tuple_green = (all_intensities_dict_green, all_mean_green, all_std_green)
    
    all_intensities_red = np.array(all_intensities_red, dtype=np.float64)
    all_mean_red = np.mean(all_intensities_red)
    all_std_red = np.std(all_intensities_red)
    
    all_intensities_tuple_red = (all_intensities_dict_red, all_mean_red, all_std_red)
    return all_intensities_tuple_blue, all_intensities_tuple_green, all_intensities_tuple_red

def get_tetrad_type_three_channels(bgr_ids, gr_ids, bg_ids, br_ids):
    current_type = 'Z'
    if 2 == bgr_ids.shape[0]:
        current_type = 'A'
    elif 1 == bgr_ids.shape[0]:
        if 1 == gr_ids.shape[0] and 1 == br_ids.shape[0] and 2 == bg_ids.shape[0]:
            current_type = 'B'
        elif 2 == gr_ids.shape[0] and 1 == br_ids.shape[0] and 1 == bg_ids.shape[0]:
            current_type = 'C'
        elif 1 == gr_ids.shape[0] and 2 == br_ids.shape[0] and 1 == bg_ids.shape[0]:
            current_type = 'D'
        elif 1 == gr_ids.shape[0] and 1 == br_ids.shape[0] and 1 == bg_ids.shape[0]:
            current_type = 'E'
    elif 0 == bgr_ids.shape[0]:
        if 1 == gr_ids.shape[0] and 1 == br_ids.shape[0] and 1 == bg_ids.shape[0]:
            current_type = 'F'
        elif 1 == gr_ids.shape[0] and 0 == br_ids.shape[0] and 1 == bg_ids.shape[0]:
            current_type = 'G'
        elif 0 == gr_ids.shape[0] and 0 == br_ids.shape[0] and 2 == bg_ids.shape[0]:
            current_type = 'H'
        elif 2 == gr_ids.shape[0] and 0 == br_ids.shape[0] and 0 == bg_ids.shape[0]:
            current_type = 'I'
        elif 0 == gr_ids.shape[0] and 1 == br_ids.shape[0] and 1 == bg_ids.shape[0]:
            current_type = 'J'
        elif 1 == gr_ids.shape[0] and 1 == br_ids.shape[0] and 0 == bg_ids.shape[0]:
            current_type = 'K'
        elif 0 == gr_ids.shape[0] and 2 == br_ids.shape[0] and 0 == bg_ids.shape[0]:
            current_type = 'L'
        else:
            current_type = 'Z'
    return current_type

def validate_all_tetrad_types(a_prefix, path_dict, root_path, physical_channels, intensity_thr, debug_set, desired_min_area_ratio, is_purge = None, is_capture = None, is_visualize = None):
    if None is is_purge:
        is_purge = False
    if None is is_capture:
        is_capture = False
    if None is is_visualize:
        is_visualize = False
    a_bright_path = path_dict['1']
    rep_physical_channel_str = get_rep_physical_channel_str(physical_channels, a_prefix.split(os.sep)[-2])

    bright_tetrad_pollen_output_path = get_pickle_path(a_bright_path, 'tetrad_pollen')
    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens_revised')
    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    bright_tetrad_typing_path = get_pickle_path(a_bright_path, 'tetrad_types_in_monad')
    if not os.path.exists(bright_tetrad_pollen_output_path) or not os.path.exists(bright_single_pollen_output_path) or not os.path.exists(bright_tetrad_output_path):
        return {}
    
    if len(debug_set) > 0:
        is_in_debug_set = False
        for a_key in debug_set:
            if a_key in a_bright_path:
                is_in_debug_set = True
                break
        if not is_in_debug_set:
            if os.path.exists(bright_tetrad_typing_path):
                return load_data_from_pickle(bright_tetrad_typing_path)
    
    if os.path.exists(bright_tetrad_typing_path) and not is_purge and not is_capture and not is_visualize:
        return load_data_from_pickle(bright_tetrad_typing_path)

    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    red_output_path = get_pickle_path(a_red_path, 'aligned')
    green_output_path = get_pickle_path(a_green_path, 'aligned')
    blue_output_path = get_pickle_path(a_blue_path, 'aligned')
    
    o_bright_image = load_data_from_pickle(bright_output_path)
    o_red_image = load_data_from_pickle(red_output_path)
    o_green_image = load_data_from_pickle(green_output_path)
    o_blue_image = load_data_from_pickle(blue_output_path)
    
    o_bright_image = get_padded_image_for_deep(o_bright_image)
    o_red_image = get_padded_image_for_deep(o_red_image)
    o_green_image = get_padded_image_for_deep(o_green_image)
    o_blue_image = get_padded_image_for_deep(o_blue_image)
    
    single_pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_output_path)
    tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
    valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
    
    I_bg, J_bg = np.nonzero(single_pollen_mask == 0)
    bg_mean_bright = np.mean(o_bright_image[I_bg, J_bg, 0])
    bg_mean_blue = np.mean(o_blue_image[I_bg, J_bg, 0])
    bg_mean_green = np.mean(o_green_image[I_bg, J_bg, 1])
    bg_mean_red = np.mean(o_red_image[I_bg, J_bg, 2])
    
    verbose = False
    if is_capture:
        is_visualize = True
    image_h, image_w = tetrad_mask.shape[0:2]
    if is_visualize:        
        all_tetrad_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
        fig_hist, ax_hist = plt.subplots(num=1)
        
    if is_visualize and not is_capture:
        cv2.namedWindow('bright')
        cv2.namedWindow('red')
        cv2.namedWindow('green')
        cv2.namedWindow('blue')
        cv2.namedWindow('combined')
        cv2.namedWindow('all_pollens')
        cv2.namedWindow('a_pollen')
        cv2.namedWindow('copied_bright_resized')
        
        cv2.moveWindow('bright', 0, 30)
        cv2.moveWindow('red', 330, 30)
        cv2.moveWindow('green', 660, 30)
        cv2.moveWindow('blue', 990, 30)
        cv2.moveWindow('all_pollens', 0, 250)
        cv2.moveWindow('combined', 330, 250)
        cv2.moveWindow('a_pollen', 660, 250)
        cv2.moveWindow('copied_bright_resized', 0, 500)
        plt.ion()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1000, 330, 640, 480)
    
    if verbose:
        bg_mean_intensities = np.array([bg_mean_bright, bg_mean_blue, bg_mean_green, bg_mean_red], dtype=np.float64)
        print('[validate_all_tetrad_types] bg mean: {}'.format(bg_mean_intensities))
    
    all_intensities_tuple_blue, all_intensities_tuple_green, all_intensities_tuple_red = get_all_valid_intensity_statistics(valid_tetrad_dict, single_pollen_mask, o_blue_image, o_green_image, o_red_image, desired_min_area_ratio)
    all_intensities_dict_blue, all_mean_blue, all_std_blue = all_intensities_tuple_blue
    all_intensities_dict_green, all_mean_green, all_std_green = all_intensities_tuple_green
    all_intensities_dict_red, all_mean_red, all_std_red = all_intensities_tuple_red
    
    _, channel_order_rev_dict = get_channel_order_dict(rep_physical_channel_str)
        
#     print('[collect_all_tetrad_types] rep: {}, rev_dict: {}'.format(rep_physical_channel_str, channel_order_rev_dict))
#     the_rev_dict_order = (channel_order_rev_dict['R'], channel_order_rev_dict['G'], channel_order_rev_dict['B'])
    the_for_dict_order = (channel_order_rev_dict[rep_physical_channel_str[0]], channel_order_rev_dict[rep_physical_channel_str[1]], channel_order_rev_dict[rep_physical_channel_str[2]])
#     print('[collect_all_tetrad_types] {}->RGC, for dict order: {}'.format(rep_physical_channel_str, the_for_dict_order))
    manual_count_dict = {'RGB': np.int64(0), 'R': np.int64(0), 'G': np.int64(0), 'B': np.int64(0), 'BG': np.int64(0), 'BR': np.int64(0), 'GR': np.int64(0), 'Z': np.int64(0)}
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        if tetrad_id not in all_intensities_dict_blue:
            continue
        local_intensities_blue = all_intensities_dict_blue[tetrad_id]
        local_intensities_green = all_intensities_dict_green[tetrad_id]
        local_intensities_red = all_intensities_dict_red[tetrad_id]
        
        standardized_blue = (local_intensities_blue - all_mean_blue) / all_std_blue
        standardized_green = (local_intensities_green - all_mean_green) / all_std_green
        standardized_red = (local_intensities_red - all_mean_red) / all_std_red
        
        sorted_intensities_blue = np.sort(standardized_blue)
        sorted_intensities_green = np.sort(standardized_green)
        sorted_intensities_red = np.sort(standardized_red)
        if is_visualize:
            # normalize
            local_intensities_blue = local_intensities_blue/np.max(local_intensities_blue)
            local_intensities_green = local_intensities_green/np.max(local_intensities_green)
            local_intensities_red = local_intensities_red/np.max(local_intensities_red)
            
        repr_auto_tetrad_type = 'Z'
        if (is_valid_intensity(sorted_intensities_blue, intensity_thr) and is_valid_intensity(sorted_intensities_green, intensity_thr)
        and is_valid_intensity(sorted_intensities_red, intensity_thr)):
            repr_auto_tetrad_type = 'N'
#         
        if is_visualize:
            if 'N' != repr_auto_tetrad_type and not is_capture:
                print('========================= STRANGE ==============================')
                print('[validate_all_tetrad_types] tetrad: {}\nblue: {}\ngreen: {}\nred: {}'.format(
                        tetrad_id, standardized_blue, standardized_green, standardized_red))

            x = np.array(list(range(len(local_intensities_blue))))
            
            ax_hist.cla()
            ax_hist.bar(x - 0.2, standardized_blue, width=0.2, align='center', color='blue')
            ax_hist.bar(x, standardized_green, width=0.2, align='center', color='green')
            ax_hist.bar(x + 0.2, standardized_red, width=0.2, align='center', color='red')
            ax_hist.axhline(linewidth=0.5, y=0, color='black')
            ax_hist.set_ylim(-2, 2)
        
        if 'N' != repr_auto_tetrad_type:
            continue
        original_order_channel_arr = [standardized_red, standardized_green, standardized_blue]
        tetrad_typing_order_channel_arr = [original_order_channel_arr[the_for_dict_order[0]],
        original_order_channel_arr[the_for_dict_order[1]],
        original_order_channel_arr[the_for_dict_order[2]]]
        
        median_red = np.median(tetrad_typing_order_channel_arr[0])
        median_green = np.median(tetrad_typing_order_channel_arr[1])
        median_blue = np.median(tetrad_typing_order_channel_arr[2])
        
        red_on_ids = np.nonzero(tetrad_typing_order_channel_arr[0] > median_red)[0]
        green_on_ids = np.nonzero(tetrad_typing_order_channel_arr[1] > median_green)[0]
        blue_on_ids = np.nonzero(tetrad_typing_order_channel_arr[2] > median_blue)[0]
        
        gr_ids = np.intersect1d(green_on_ids, red_on_ids)
        bg_ids = np.intersect1d(blue_on_ids, green_on_ids)
        br_ids = np.intersect1d(blue_on_ids, red_on_ids)
        
        bgr_ids = reduce(np.intersect1d, (blue_on_ids, green_on_ids, red_on_ids))
        current_type = get_tetrad_type_three_channels(bgr_ids, gr_ids, bg_ids, br_ids)
        if 'Z' == current_type:
            continue
            
        median_channel_red = np.median(standardized_red)
        median_channel_green = np.median(standardized_green)
        median_channel_blue = np.median(standardized_blue)
        
        for local_pollen_id in range(len(standardized_red)):
            intensity_blue = standardized_blue[local_pollen_id]
            intensity_green = standardized_green[local_pollen_id]
            intensity_red = standardized_red[local_pollen_id]
            
            if intensity_red > median_channel_red and intensity_blue > median_channel_blue and intensity_green > median_channel_green:
                cur_type = 'RGB'
            elif intensity_red < median_channel_red and intensity_blue < median_channel_blue and intensity_green < median_channel_green:
                cur_type = 'None'
            elif intensity_red > median_channel_red and intensity_blue < median_channel_blue and intensity_green < median_channel_green:
                cur_type = 'R'
            elif intensity_red < median_channel_red and intensity_blue < median_channel_blue and intensity_green > median_channel_green:
                cur_type = 'G'
            elif intensity_red < median_channel_red and intensity_blue > median_channel_blue and intensity_green < median_channel_green:
                cur_type = 'B'
            elif intensity_red < median_channel_red and intensity_blue > median_channel_blue and intensity_green > median_channel_green:
                cur_type = 'BG'
            elif intensity_red > median_channel_red and intensity_blue > median_channel_blue and intensity_green < median_channel_green:
                cur_type = 'BR'
            elif intensity_red > median_channel_red and intensity_blue < median_channel_blue and intensity_green > median_channel_green:
                cur_type = 'GR'
            else:
                cur_type = 'Z'
            if cur_type not in manual_count_dict:
                manual_count_dict[cur_type] = np.int64(0)
            manual_count_dict[cur_type] += 1
            if is_visualize:
                print('[validate_all_tetrad_types] tetrad_id: {}, pollen id: {}, type: {}'.format(tetrad_id, local_pollen_id, cur_type))
    dump_data_to_pickle(bright_tetrad_typing_path, manual_count_dict)
    return manual_count_dict

def collect_all_tetrad_types_enlarge(a_prefix, path_dict, root_path, physical_channels, intensity_thr, debug_set, desired_min_area_ratio, is_purge = None, is_capture = None, is_visualize = None):
    if None is is_purge:
        is_purge = False
    if None is is_capture:
        is_capture = False
    if None is is_visualize:
        is_visualize = False
    a_bright_path = path_dict['1']
    
    rep_physical_channel_str = get_rep_physical_channel_str(physical_channels, a_prefix.split(os.sep)[-2])
    bright_tetrad_pollen_output_path = get_pickle_path(a_bright_path, 'tetrad_pollen')
    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens_revised')
    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    bright_tetrad_typing_path = get_pickle_path(a_bright_path, 'tetrad_types')
    if not os.path.exists(bright_tetrad_pollen_output_path) or not os.path.exists(bright_single_pollen_output_path) or not os.path.exists(bright_tetrad_output_path):
        return
    
    if len(debug_set) > 0:
        is_in_debug_set = False
        for a_key in debug_set:
            if a_key in a_bright_path:
                is_in_debug_set = True
                break
        if not is_in_debug_set:
            if os.path.exists(bright_tetrad_typing_path):
                return load_data_from_pickle(bright_tetrad_typing_path)
            return
    
    if os.path.exists(bright_tetrad_typing_path) and not is_purge and not is_capture and not is_visualize:
        return load_data_from_pickle(bright_tetrad_typing_path)
    capturing_type = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    red_output_path = get_pickle_path(a_red_path, 'aligned')
    green_output_path = get_pickle_path(a_green_path, 'aligned')
    blue_output_path = get_pickle_path(a_blue_path, 'aligned')
    
    o_bright_image = load_data_from_pickle(bright_output_path)
    o_red_image = load_data_from_pickle(red_output_path)
    o_green_image = load_data_from_pickle(green_output_path)
    o_blue_image = load_data_from_pickle(blue_output_path)
    
    o_bright_image = get_padded_image_for_deep(o_bright_image)
    o_red_image = get_padded_image_for_deep(o_red_image)
    o_green_image = get_padded_image_for_deep(o_green_image)
    o_blue_image = get_padded_image_for_deep(o_blue_image)
    
    h, w = o_bright_image.shape[0:2]
    o_bright_image =  cv2.resize(o_bright_image, (h * 2, w * 2))
    h, w = o_red_image.shape[0:2]
    o_red_image =  cv2.resize(o_red_image, (h * 2, w * 2))
    h, w = o_green_image.shape[0:2]
    o_green_image =  cv2.resize(o_green_image, (h * 2, w * 2))
    h, w = o_blue_image.shape[0:2]
    o_blue_image =  cv2.resize(o_blue_image, (h * 2, w * 2))
    
    single_pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_output_path)
    tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
    valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
    
    I_bg, J_bg = np.nonzero(single_pollen_mask == 0)
    bg_mean_bright = np.mean(o_bright_image[I_bg, J_bg, 0])
    bg_mean_blue = np.mean(o_blue_image[I_bg, J_bg, 0])
    bg_mean_green = np.mean(o_green_image[I_bg, J_bg, 1])
    bg_mean_red = np.mean(o_red_image[I_bg, J_bg, 2])
    
    tetrad_type_dict = {'A': np.int64(0), 'B': np.int64(0), 'C': np.int64(0), 'D': np.int64(0), 'E': np.int64(0),
                        'F': np.int64(0), 'G': np.int64(0), 'H': np.int64(0), 'I': np.int64(0), 'J': np.int64(0),
                        'K': np.int64(0), 'L': np.int64(0), 'Z': np.int64(0)}
    
    gr_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}
    bg_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}
    br_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}

    is_include_dyad = False
    verbose = False
    if is_capture:
        is_visualize = True
    image_h, image_w = tetrad_mask.shape[0:2]

    if is_visualize:        
        all_tetrad_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
        fig_hist, ax_hist = plt.subplots(num=1)
        
    if is_visualize and not is_capture:
        cv2.namedWindow('bright')
        cv2.namedWindow('red')
        cv2.namedWindow('green')
        cv2.namedWindow('blue')
        cv2.namedWindow('combined')
        cv2.namedWindow('all_pollens')
        cv2.namedWindow('a_pollen')
        cv2.namedWindow('copied_bright_resized')
        
        cv2.moveWindow('bright', 0, 30)
        cv2.moveWindow('red', 330, 30)
        cv2.moveWindow('green', 660, 30)
        cv2.moveWindow('blue', 990, 30)
        cv2.moveWindow('all_pollens', 0, 250)
        cv2.moveWindow('combined', 330, 250)
        cv2.moveWindow('a_pollen', 660, 250)
        cv2.moveWindow('copied_bright_resized', 0, 500)
        plt.ion()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1000, 330, 640, 480)
    
    if verbose:
        bg_mean_intensities = np.array([bg_mean_bright, bg_mean_blue, bg_mean_green, bg_mean_red], dtype=np.float64)
        print('[collect_all_tetrad_types] bg mean: {}'.format(bg_mean_intensities))
    
    all_intensities_tuple_blue, all_intensities_tuple_green, all_intensities_tuple_red = get_all_valid_intensity_statistics(valid_tetrad_dict, single_pollen_mask, o_blue_image, o_green_image, o_red_image, desired_min_area_ratio, is_include_dyad)
    all_intensities_dict_blue, all_mean_blue, all_std_blue = all_intensities_tuple_blue
    all_intensities_dict_green, all_mean_green, all_std_green = all_intensities_tuple_green
    all_intensities_dict_red, all_mean_red, all_std_red = all_intensities_tuple_red
    pollen_colors = np.array(random_colors(4)) * 255
    
    _, channel_order_rev_dict = get_channel_order_dict(rep_physical_channel_str)
        
    the_for_dict_order = (channel_order_rev_dict[rep_physical_channel_str[0]], channel_order_rev_dict[rep_physical_channel_str[1]], channel_order_rev_dict[rep_physical_channel_str[2]])
    valid_dyad_types = set(['A', 'B', 'C', 'D', 'F'])
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        if tetrad_id not in all_intensities_dict_blue:
            continue
        local_intensities_blue = all_intensities_dict_blue[tetrad_id]
        local_intensities_green = all_intensities_dict_green[tetrad_id]
        local_intensities_red = all_intensities_dict_red[tetrad_id]
        
        standardized_blue = (local_intensities_blue - all_mean_blue) / all_std_blue
        standardized_green = (local_intensities_green - all_mean_green) / all_std_green
        standardized_red = (local_intensities_red - all_mean_red) / all_std_red
        
        sorted_intensities_blue = np.sort(standardized_blue)
        sorted_intensities_green = np.sort(standardized_green)
        sorted_intensities_red = np.sort(standardized_red)
        if is_visualize:
            # normalize
            local_intensities_blue = local_intensities_blue/np.max(local_intensities_blue)
            local_intensities_green = local_intensities_green/np.max(local_intensities_green)
            local_intensities_red = local_intensities_red/np.max(local_intensities_red)
            
        repr_auto_tetrad_type = 'Z'
        if (is_valid_intensity(sorted_intensities_blue, intensity_thr) and is_valid_intensity(sorted_intensities_green, intensity_thr)
        and is_valid_intensity(sorted_intensities_red, intensity_thr)):
            repr_auto_tetrad_type = 'N'

        if is_visualize:
            if 'N' != repr_auto_tetrad_type and not is_capture:
                print('========================= STRANGE ==============================')
                print('[collect_all_tetrad_types] tetrad: {}\nblue: {}\ngreen: {}\nred: {}'.format(
                        tetrad_id, standardized_blue, standardized_green, standardized_red))

            x = np.array(list(range(len(local_intensities_blue))))
            
            ax_hist.cla()
            ax_hist.bar(x - 0.2, standardized_blue, width=0.2, align='center', color='blue')
            ax_hist.bar(x, standardized_green, width=0.2, align='center', color='green')
            ax_hist.bar(x + 0.2, standardized_red, width=0.2, align='center', color='red')
            ax_hist.axhline(linewidth=0.5, y=0, color='black')
            ax_hist.set_ylim(-2, 2)
        if 'N' != repr_auto_tetrad_type:
            continue
        
        original_order_channel_arr = [standardized_red, standardized_green, standardized_blue]
        tetrad_typing_order_channel_arr = [original_order_channel_arr[the_for_dict_order[0]],
        original_order_channel_arr[the_for_dict_order[1]],
        original_order_channel_arr[the_for_dict_order[2]]]
        
        median_red = np.median(tetrad_typing_order_channel_arr[0])
        median_green = np.median(tetrad_typing_order_channel_arr[1])
        median_blue = np.median(tetrad_typing_order_channel_arr[2])
        
        red_on_ids = np.nonzero(tetrad_typing_order_channel_arr[0] > median_red)[0]
        green_on_ids = np.nonzero(tetrad_typing_order_channel_arr[1] > median_green)[0]
        blue_on_ids = np.nonzero(tetrad_typing_order_channel_arr[2] > median_blue)[0]
        
        gr_ids = np.intersect1d(green_on_ids, red_on_ids)
        bg_ids = np.intersect1d(blue_on_ids, green_on_ids)
        br_ids = np.intersect1d(blue_on_ids, red_on_ids)

        bgr_ids = reduce(np.intersect1d, (blue_on_ids, green_on_ids, red_on_ids))
        current_type = get_tetrad_type_three_channels(bgr_ids, gr_ids, bg_ids, br_ids)
        if 3 == len(pollen_ids) and current_type not in valid_dyad_types:
            current_type = 'Z'
        if 'Z' != current_type:
            tetrad_type_dict[current_type] += 1
            if 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 2 == gr_ids.shape[0]:
                gr_current_type = 'P'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 1 == gr_ids.shape[0]:
                gr_current_type = 'T'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 0 == gr_ids.shape[0]:
                gr_current_type = 'NPD'
            else:
                gr_current_type = 'Z'
            
            if 2 == green_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 2 == bg_ids.shape[0]:
                bg_current_type = 'P'
            elif 2 == green_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 1 == bg_ids.shape[0]:
                bg_current_type = 'T'
            elif 2 == green_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 0 == bg_ids.shape[0]:
                bg_current_type = 'NPD'
            else:
                bg_current_type = 'Z'
    
            if 2 == red_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 2 == br_ids.shape[0]:
                br_current_type = 'P'
            elif 2 == red_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 1 == br_ids.shape[0]:
                br_current_type = 'T'
            elif 2 == red_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 0 == br_ids.shape[0]:
                br_current_type = 'NPD'
            else:
                br_current_type = 'Z'
    
            gr_tetrad_type_dict[gr_current_type] += 1
            bg_tetrad_type_dict[bg_current_type] += 1
            br_tetrad_type_dict[br_current_type] += 1
        
        if is_visualize:
            if not is_capture:
                print('[collect_all_tetrad_types] Auto {}, {}'.format(repr_auto_tetrad_type, current_type))
            single_pollen_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
            tetrad_mask_subsampled = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1]), dtype=np.uint8)
     
            I, J = np.nonzero(tetrad_id == tetrad_mask)
            copied_all_tetrad_disp = all_tetrad_disp.copy()
            copied_all_tetrad_disp[I, J] = (0, 255, 255)
            copied_bright = o_bright_image.copy()
            copied_red = o_red_image.copy()
            copied_green = o_green_image.copy()
            copied_blue = o_blue_image.copy()
             
            for idx, pollen_id in enumerate(pollen_ids):
                I_pollen, J_pollen = np.nonzero(single_pollen_mask == pollen_id)
                single_pollen_disp[I_pollen, J_pollen] = pollen_colors[idx]
            tetrad_mask_subsampled[I, J] = 255
            min_y = max(np.min(I) - 64, 0)
            max_y = min(np.max(I) + 64, image_h)
            min_x = max(np.min(J) - 64, 0)
            max_x = min(np.max(J) + 64, image_w)
             
            copied_all_tetrad_disp = copied_all_tetrad_disp[min_y:max_y,min_x:max_x,:]
            copied_bright_resized = copied_bright.copy()
            copied_bright_resized = cv2.rectangle(copied_bright_resized, (min_x, min_y), (max_x, max_y), (0, 0, 255), 5)
            
            
            copied_bright = copied_bright[min_y:max_y,min_x:max_x,...]
            copied_red = copied_red[min_y:max_y,min_x:max_x,...]
            copied_green = copied_green[min_y:max_y,min_x:max_x,...]
            copied_blue = copied_blue[min_y:max_y,min_x:max_x,...]
             
            combined_image = np.zeros_like(copied_bright)
            combined_image[...,0] = copied_blue[...,0]
            combined_image[...,1] = copied_green[...,1]
            combined_image[...,2] = copied_red[...,2]
             
            single_pollen_disp = single_pollen_disp[min_y:max_y,min_x:max_x]
            tetrad_mask_subsampled = tetrad_mask_subsampled[min_y:max_y,min_x:max_x]

            combined_image = cv2.bitwise_and(combined_image, combined_image, mask=tetrad_mask_subsampled)
            copied_red = cv2.bitwise_and(copied_red, copied_red, mask=tetrad_mask_subsampled)
            copied_green = cv2.bitwise_and(copied_green, copied_green, mask=tetrad_mask_subsampled)
            copied_blue = cv2.bitwise_and(copied_blue, copied_blue, mask=tetrad_mask_subsampled)
            if not is_capture:
                copied_bright_resized = cv2.resize(copied_bright_resized, (640, 480))
                cv2.imshow('all_pollens', copied_all_tetrad_disp)
                cv2.imshow('a_pollen', single_pollen_disp)
                cv2.imshow('bright', copied_bright)
                cv2.imshow('copied_bright_resized', copied_bright_resized)
                cv2.imshow('red', copied_red)
                cv2.imshow('green', copied_green)
                cv2.imshow('blue', copied_blue)
                cv2.imshow('combined', combined_image)
            if verbose:
                print('[select_pollen_and_tetrads_from_combined_masks] select the type of the tetrad (A-L)')
            if is_capture:
                if current_type in capturing_type:
                    last_prefix = a_prefix.split(os.sep)[-2]
                    file_name_prefix = a_prefix.split(os.sep)[-1]
                    
                    capture_dir = '{}/captures/{}/{}'.format(root_path, last_prefix, current_type)
                    if not os.path.exists(capture_dir):
                        os.makedirs(capture_dir)
                    
                    copied_bright_resized = cv2.resize(copied_bright_resized, (1024, 1024))
                    
                    copied_bright = cv2.cvtColor(copied_bright, cv2.COLOR_BGR2RGB)
                    single_pollen_disp = cv2.cvtColor(single_pollen_disp, cv2.COLOR_BGR2RGB)
                    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
                    copied_blue = cv2.cvtColor(copied_blue, cv2.COLOR_BGR2RGB)
                    copied_red = cv2.cvtColor(copied_red, cv2.COLOR_BGR2RGB)
                    copied_green = cv2.cvtColor(copied_green, cv2.COLOR_BGR2RGB)
                    
                    fig_capture = plt.figure(figsize=(24, 8))
                    
                    gs_main = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
                    gs_third = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[0], wspace=0.05, hspace=0.05)
                    
                    ax_bright_cropped_result = plt.Subplot(fig_capture, gs_third[0, 0])
                    fig_capture.add_subplot(ax_bright_cropped_result)
                    
                    ax_single_pollen_result = plt.Subplot(fig_capture, gs_third[0, 1])
                    fig_capture.add_subplot(ax_single_pollen_result)
                    
                    ax_combined_result = plt.Subplot(fig_capture, gs_third[0, 2])
                    fig_capture.add_subplot(ax_combined_result)
                    
                    ax_red_result = plt.Subplot(fig_capture, gs_third[1, 0])
                    fig_capture.add_subplot(ax_red_result)
                    
                    ax_green_result = plt.Subplot(fig_capture, gs_third[1, 1])
                    fig_capture.add_subplot(ax_green_result)
                    
                    ax_blue_result = plt.Subplot(fig_capture, gs_third[1, 2])
                    fig_capture.add_subplot(ax_blue_result)
                    
                    ax_bright_cropped_result.imshow(copied_bright)
                    ax_single_pollen_result.imshow(single_pollen_disp)
                    ax_combined_result.imshow(combined_image)
                    
                    ax_red_result.imshow(copied_red)
                    ax_green_result.imshow(copied_green)
                    ax_blue_result.imshow(copied_blue)
                    
                    ax_bright_cropped_result.get_xaxis().set_visible(False)
                    ax_bright_cropped_result.get_yaxis().set_visible(False)
                    ax_single_pollen_result.get_xaxis().set_visible(False)
                    ax_single_pollen_result.get_yaxis().set_visible(False)
                    ax_combined_result.get_xaxis().set_visible(False)
                    ax_combined_result.get_yaxis().set_visible(False)
                    ax_red_result.get_xaxis().set_visible(False)
                    ax_red_result.get_yaxis().set_visible(False)
                    ax_green_result.get_xaxis().set_visible(False)
                    ax_green_result.get_yaxis().set_visible(False)
                    ax_blue_result.get_xaxis().set_visible(False)
                    ax_blue_result.get_yaxis().set_visible(False)
    
                    copied_bright_resized_out_path = '{}/{}{}_0_copied_bright_resized.png'.format(capture_dir, file_name_prefix, tetrad_id)
                    
                    result_out_path = '{}/{}{}_1_combined_mask.png'.format(capture_dir, file_name_prefix, tetrad_id)
                    
                    predict_hist_out_path = '{}/{}{}_2_prediction_hist.png'.format(capture_dir, file_name_prefix, tetrad_id)
                    
                    cv2.imwrite(copied_bright_resized_out_path, copied_bright_resized)
                    fig_capture.savefig(result_out_path, edgecolor='none', bbox_inches='tight', pad_inches=0, dpi=300)
                    fig_hist.savefig(predict_hist_out_path, edgecolor='none', bbox_inches='tight', pad_inches=0, dpi=300)
            else:
                examiner_key = chr(cv2.waitKey(0)).upper()
            
    two_channel_dicts = (gr_tetrad_type_dict, bg_tetrad_type_dict, br_tetrad_type_dict)
    
    if not is_capture or is_purge:
        dump_data_to_pickle(bright_tetrad_typing_path, (tetrad_type_dict, two_channel_dicts))
    
    return (tetrad_type_dict, two_channel_dicts)

def collect_all_tetrad_types(a_prefix, path_dict, root_path, physical_channels, intensity_thr, debug_set, desired_min_area_ratio, is_purge = None, is_capture = None, is_visualize = None):
    if None is is_purge:
        is_purge = False
    if None is is_capture:
        is_capture = False
    if None is is_visualize:
        is_visualize = False
    a_bright_path = path_dict['1']
    
    rep_physical_channel_str = get_rep_physical_channel_str(physical_channels, a_prefix.split(os.sep)[-2])
    bright_tetrad_pollen_output_path = get_pickle_path(a_bright_path, 'tetrad_pollen')
    bright_single_pollen_output_path = get_pickle_path(a_bright_path, 'single_pollens_revised')
    bright_tetrad_output_path = get_pickle_path(a_bright_path, 'tetrads')
    bright_tetrad_typing_path = get_pickle_path(a_bright_path, 'tetrad_types')
    if not os.path.exists(bright_tetrad_pollen_output_path) or not os.path.exists(bright_single_pollen_output_path) or not os.path.exists(bright_tetrad_output_path):
        return
    
    if len(debug_set) > 0:
        is_in_debug_set = False
        for a_key in debug_set:
            if a_key in a_bright_path:
                is_in_debug_set = True
                break
        if not is_in_debug_set:
            if os.path.exists(bright_tetrad_typing_path):
                return load_data_from_pickle(bright_tetrad_typing_path)
            return
    
    if os.path.exists(bright_tetrad_typing_path) and not is_purge and not is_capture and not is_visualize:
        return load_data_from_pickle(bright_tetrad_typing_path)
    capturing_type = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
    a_red_path = path_dict['2']
    a_green_path = path_dict['3']
    a_blue_path = path_dict['4']
    
    bright_output_path = get_pickle_path(a_bright_path, 'aligned')
    red_output_path = get_pickle_path(a_red_path, 'aligned')
    green_output_path = get_pickle_path(a_green_path, 'aligned')
    blue_output_path = get_pickle_path(a_blue_path, 'aligned')
    
    o_bright_image = load_data_from_pickle(bright_output_path)
    o_red_image = load_data_from_pickle(red_output_path)
    o_green_image = load_data_from_pickle(green_output_path)
    o_blue_image = load_data_from_pickle(blue_output_path)
    
    o_bright_image = get_padded_image_for_deep(o_bright_image)
    o_red_image = get_padded_image_for_deep(o_red_image)
    o_green_image = get_padded_image_for_deep(o_green_image)
    o_blue_image = get_padded_image_for_deep(o_blue_image)
    
    single_pollen_mask, n_single_pollens = load_data_from_pickle(bright_single_pollen_output_path)
    tetrad_mask, n_tetrads = load_data_from_pickle(bright_tetrad_output_path)
    valid_tetrad_dict = load_data_from_pickle(bright_tetrad_pollen_output_path)
    
    I_bg, J_bg = np.nonzero(single_pollen_mask == 0)
    bg_mean_bright = np.mean(o_bright_image[I_bg, J_bg, 0])
    bg_mean_blue = np.mean(o_blue_image[I_bg, J_bg, 0])
    bg_mean_green = np.mean(o_green_image[I_bg, J_bg, 1])
    bg_mean_red = np.mean(o_red_image[I_bg, J_bg, 2])
    
    tetrad_type_dict = {'A': np.int64(0), 'B': np.int64(0), 'C': np.int64(0), 'D': np.int64(0), 'E': np.int64(0),
                        'F': np.int64(0), 'G': np.int64(0), 'H': np.int64(0), 'I': np.int64(0), 'J': np.int64(0),
                        'K': np.int64(0), 'L': np.int64(0), 'Z': np.int64(0)}
    
    gr_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}
    bg_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}
    br_tetrad_type_dict = {'P': np.int64(0), 'T': np.int64(0), 'NPD': np.int64(0), 'Z': np.int64(0)}

    is_include_dyad = False
    verbose = False
    if is_capture:
        is_visualize = True
    image_h, image_w = tetrad_mask.shape[0:2]

    if is_visualize:        
        all_tetrad_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
        fig_hist, ax_hist = plt.subplots(num=1)
        
    if is_visualize and not is_capture:
        cv2.namedWindow('bright')
        cv2.namedWindow('red')
        cv2.namedWindow('green')
        cv2.namedWindow('blue')
        cv2.namedWindow('combined')
        cv2.namedWindow('all_pollens')
        cv2.namedWindow('a_pollen')
        cv2.namedWindow('copied_bright_resized')
        
        cv2.moveWindow('bright', 0, 30)
        cv2.moveWindow('red', 330, 30)
        cv2.moveWindow('green', 660, 30)
        cv2.moveWindow('blue', 990, 30)
        cv2.moveWindow('all_pollens', 0, 250)
        cv2.moveWindow('combined', 330, 250)
        cv2.moveWindow('a_pollen', 660, 250)
        cv2.moveWindow('copied_bright_resized', 0, 500)
        plt.ion()
        mngr = plt.get_current_fig_manager()
        mngr.window.setGeometry(1000, 330, 640, 480)
    
    if verbose:
        bg_mean_intensities = np.array([bg_mean_bright, bg_mean_blue, bg_mean_green, bg_mean_red], dtype=np.float64)
        print('[collect_all_tetrad_types] bg mean: {}'.format(bg_mean_intensities))
    
    all_intensities_tuple_blue, all_intensities_tuple_green, all_intensities_tuple_red = get_all_valid_intensity_statistics(valid_tetrad_dict, single_pollen_mask, o_blue_image, o_green_image, o_red_image, desired_min_area_ratio, is_include_dyad)
    all_intensities_dict_blue, all_mean_blue, all_std_blue = all_intensities_tuple_blue
    all_intensities_dict_green, all_mean_green, all_std_green = all_intensities_tuple_green
    all_intensities_dict_red, all_mean_red, all_std_red = all_intensities_tuple_red
    pollen_colors = np.array(random_colors(4)) * 255
    
    _, channel_order_rev_dict = get_channel_order_dict(rep_physical_channel_str)
        
    the_for_dict_order = (channel_order_rev_dict[rep_physical_channel_str[0]], channel_order_rev_dict[rep_physical_channel_str[1]], channel_order_rev_dict[rep_physical_channel_str[2]])
    valid_dyad_types = set(['A', 'B', 'C', 'D', 'F'])
    for tetrad_id, pollen_ids in valid_tetrad_dict.items():
        if tetrad_id not in all_intensities_dict_blue:
            continue
        local_intensities_blue = all_intensities_dict_blue[tetrad_id]
        local_intensities_green = all_intensities_dict_green[tetrad_id]
        local_intensities_red = all_intensities_dict_red[tetrad_id]
        
        standardized_blue = (local_intensities_blue - all_mean_blue) / all_std_blue
        standardized_green = (local_intensities_green - all_mean_green) / all_std_green
        standardized_red = (local_intensities_red - all_mean_red) / all_std_red
        
        sorted_intensities_blue = np.sort(standardized_blue)
        sorted_intensities_green = np.sort(standardized_green)
        sorted_intensities_red = np.sort(standardized_red)
        if is_visualize:
            # normalize
            local_intensities_blue = local_intensities_blue/np.max(local_intensities_blue)
            local_intensities_green = local_intensities_green/np.max(local_intensities_green)
            local_intensities_red = local_intensities_red/np.max(local_intensities_red)
            
        repr_auto_tetrad_type = 'Z'
        if (is_valid_intensity(sorted_intensities_blue, intensity_thr) and is_valid_intensity(sorted_intensities_green, intensity_thr)
        and is_valid_intensity(sorted_intensities_red, intensity_thr)):
            repr_auto_tetrad_type = 'N'

        if is_visualize:
            if 'N' != repr_auto_tetrad_type and not is_capture:
                print('========================= STRANGE ==============================')
                print('[collect_all_tetrad_types] tetrad: {}\nblue: {}\ngreen: {}\nred: {}'.format(
                        tetrad_id, standardized_blue, standardized_green, standardized_red))

            x = np.array(list(range(len(local_intensities_blue))))
            
            ax_hist.cla()
            ax_hist.bar(x - 0.2, standardized_blue, width=0.2, align='center', color='blue')
            ax_hist.bar(x, standardized_green, width=0.2, align='center', color='green')
            ax_hist.bar(x + 0.2, standardized_red, width=0.2, align='center', color='red')
            ax_hist.axhline(linewidth=0.5, y=0, color='black')
            ax_hist.set_ylim(-2, 2)
        if 'N' != repr_auto_tetrad_type:
            continue
        
        original_order_channel_arr = [standardized_red, standardized_green, standardized_blue]
        tetrad_typing_order_channel_arr = [original_order_channel_arr[the_for_dict_order[0]],
        original_order_channel_arr[the_for_dict_order[1]],
        original_order_channel_arr[the_for_dict_order[2]]]
        
        median_red = np.median(tetrad_typing_order_channel_arr[0])
        median_green = np.median(tetrad_typing_order_channel_arr[1])
        median_blue = np.median(tetrad_typing_order_channel_arr[2])
        
        red_on_ids = np.nonzero(tetrad_typing_order_channel_arr[0] > median_red)[0]
        green_on_ids = np.nonzero(tetrad_typing_order_channel_arr[1] > median_green)[0]
        blue_on_ids = np.nonzero(tetrad_typing_order_channel_arr[2] > median_blue)[0]
        
        gr_ids = np.intersect1d(green_on_ids, red_on_ids)
        bg_ids = np.intersect1d(blue_on_ids, green_on_ids)
        br_ids = np.intersect1d(blue_on_ids, red_on_ids)

        bgr_ids = reduce(np.intersect1d, (blue_on_ids, green_on_ids, red_on_ids))
        current_type = get_tetrad_type_three_channels(bgr_ids, gr_ids, bg_ids, br_ids)
        if 3 == len(pollen_ids) and current_type not in valid_dyad_types:
            current_type = 'Z'
        if 'Z' != current_type:
            tetrad_type_dict[current_type] += 1
            if 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 2 == gr_ids.shape[0]:
                gr_current_type = 'P'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 1 == gr_ids.shape[0]:
                gr_current_type = 'T'
            elif 2 == green_on_ids.shape[0] and 2 == red_on_ids.shape[0] and 0 == gr_ids.shape[0]:
                gr_current_type = 'NPD'
            else:
                gr_current_type = 'Z'
            
            if 2 == green_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 2 == bg_ids.shape[0]:
                bg_current_type = 'P'
            elif 2 == green_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 1 == bg_ids.shape[0]:
                bg_current_type = 'T'
            elif 2 == green_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 0 == bg_ids.shape[0]:
                bg_current_type = 'NPD'
            else:
                bg_current_type = 'Z'
    
            if 2 == red_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 2 == br_ids.shape[0]:
                br_current_type = 'P'
            elif 2 == red_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 1 == br_ids.shape[0]:
                br_current_type = 'T'
            elif 2 == red_on_ids.shape[0] and 2 == blue_on_ids.shape[0] and 0 == br_ids.shape[0]:
                br_current_type = 'NPD'
            else:
                br_current_type = 'Z'
    
            gr_tetrad_type_dict[gr_current_type] += 1
            bg_tetrad_type_dict[bg_current_type] += 1
            br_tetrad_type_dict[br_current_type] += 1
        
        if is_visualize:
            if not is_capture:
                print('[collect_all_tetrad_types] Auto {}, {}'.format(repr_auto_tetrad_type, current_type))
            single_pollen_disp = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1], 3), dtype=np.uint8)
            tetrad_mask_subsampled = np.zeros((tetrad_mask.shape[0], tetrad_mask.shape[1]), dtype=np.uint8)
     
            I, J = np.nonzero(tetrad_id == tetrad_mask)
            copied_all_tetrad_disp = all_tetrad_disp.copy()
            copied_all_tetrad_disp[I, J] = (0, 255, 255)
            copied_bright = o_bright_image.copy()
            copied_red = o_red_image.copy()
            copied_green = o_green_image.copy()
            copied_blue = o_blue_image.copy()
             
            for idx, pollen_id in enumerate(pollen_ids):
                I_pollen, J_pollen = np.nonzero(single_pollen_mask == pollen_id)
                single_pollen_disp[I_pollen, J_pollen] = pollen_colors[idx]
            tetrad_mask_subsampled[I, J] = 255
            min_y = max(np.min(I) - 64, 0)
            max_y = min(np.max(I) + 64, image_h)
            min_x = max(np.min(J) - 64, 0)
            max_x = min(np.max(J) + 64, image_w)
             
            copied_all_tetrad_disp = copied_all_tetrad_disp[min_y:max_y,min_x:max_x,:]
            copied_bright_resized = copied_bright.copy()
            copied_bright_resized = cv2.rectangle(copied_bright_resized, (min_x, min_y), (max_x, max_y), (0, 0, 255), 5)
            
            
            copied_bright = copied_bright[min_y:max_y,min_x:max_x,...]
            copied_red = copied_red[min_y:max_y,min_x:max_x,...]
            copied_green = copied_green[min_y:max_y,min_x:max_x,...]
            copied_blue = copied_blue[min_y:max_y,min_x:max_x,...]
             
            combined_image = np.zeros_like(copied_bright)
            combined_image[...,0] = copied_blue[...,0]
            combined_image[...,1] = copied_green[...,1]
            combined_image[...,2] = copied_red[...,2]
             
            single_pollen_disp = single_pollen_disp[min_y:max_y,min_x:max_x]
            tetrad_mask_subsampled = tetrad_mask_subsampled[min_y:max_y,min_x:max_x]

            combined_image = cv2.bitwise_and(combined_image, combined_image, mask=tetrad_mask_subsampled)
            copied_red = cv2.bitwise_and(copied_red, copied_red, mask=tetrad_mask_subsampled)
            copied_green = cv2.bitwise_and(copied_green, copied_green, mask=tetrad_mask_subsampled)
            copied_blue = cv2.bitwise_and(copied_blue, copied_blue, mask=tetrad_mask_subsampled)
            if not is_capture:
                copied_bright_resized = cv2.resize(copied_bright_resized, (640, 480))
                cv2.imshow('all_pollens', copied_all_tetrad_disp)
                cv2.imshow('a_pollen', single_pollen_disp)
                cv2.imshow('bright', copied_bright)
                cv2.imshow('copied_bright_resized', copied_bright_resized)
                cv2.imshow('red', copied_red)
                cv2.imshow('green', copied_green)
                cv2.imshow('blue', copied_blue)
                cv2.imshow('combined', combined_image)
            if verbose:
                print('[select_pollen_and_tetrads_from_combined_masks] select the type of the tetrad (A-L)')
            if is_capture:
                if current_type in capturing_type:
                    last_prefix = a_prefix.split(os.sep)[-2]
                    file_name_prefix = a_prefix.split(os.sep)[-1]
                    
                    capture_dir = '{}/captures/{}/{}'.format(root_path, last_prefix, current_type)
                    if not os.path.exists(capture_dir):
                        os.makedirs(capture_dir)
                    
                    copied_bright_resized = cv2.resize(copied_bright_resized, (1024, 1024))
                    
                    copied_bright = cv2.cvtColor(copied_bright, cv2.COLOR_BGR2RGB)
                    single_pollen_disp = cv2.cvtColor(single_pollen_disp, cv2.COLOR_BGR2RGB)
                    combined_image = cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB)
                    copied_blue = cv2.cvtColor(copied_blue, cv2.COLOR_BGR2RGB)
                    copied_red = cv2.cvtColor(copied_red, cv2.COLOR_BGR2RGB)
                    copied_green = cv2.cvtColor(copied_green, cv2.COLOR_BGR2RGB)
                    
                    fig_capture = plt.figure(figsize=(24, 8))
                    
                    gs_main = gridspec.GridSpec(1, 1, wspace=0, hspace=0)
                    gs_third = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_main[0], wspace=0.05, hspace=0.05)
                    
                    ax_bright_cropped_result = plt.Subplot(fig_capture, gs_third[0, 0])
                    fig_capture.add_subplot(ax_bright_cropped_result)
                    
                    ax_single_pollen_result = plt.Subplot(fig_capture, gs_third[0, 1])
                    fig_capture.add_subplot(ax_single_pollen_result)
                    
                    ax_combined_result = plt.Subplot(fig_capture, gs_third[0, 2])
                    fig_capture.add_subplot(ax_combined_result)
                    
                    ax_red_result = plt.Subplot(fig_capture, gs_third[1, 0])
                    fig_capture.add_subplot(ax_red_result)
                    
                    ax_green_result = plt.Subplot(fig_capture, gs_third[1, 1])
                    fig_capture.add_subplot(ax_green_result)
                    
                    ax_blue_result = plt.Subplot(fig_capture, gs_third[1, 2])
                    fig_capture.add_subplot(ax_blue_result)
                    
                    ax_bright_cropped_result.imshow(copied_bright)
                    ax_single_pollen_result.imshow(single_pollen_disp)
                    ax_combined_result.imshow(combined_image)
                    
                    ax_red_result.imshow(copied_red)
                    ax_green_result.imshow(copied_green)
                    ax_blue_result.imshow(copied_blue)
                    
                    ax_bright_cropped_result.get_xaxis().set_visible(False)
                    ax_bright_cropped_result.get_yaxis().set_visible(False)
                    ax_single_pollen_result.get_xaxis().set_visible(False)
                    ax_single_pollen_result.get_yaxis().set_visible(False)
                    ax_combined_result.get_xaxis().set_visible(False)
                    ax_combined_result.get_yaxis().set_visible(False)
                    ax_red_result.get_xaxis().set_visible(False)
                    ax_red_result.get_yaxis().set_visible(False)
                    ax_green_result.get_xaxis().set_visible(False)
                    ax_green_result.get_yaxis().set_visible(False)
                    ax_blue_result.get_xaxis().set_visible(False)
                    ax_blue_result.get_yaxis().set_visible(False)
    
                    
                    copied_bright_resized_out_path = '{}/{}{}_0_copied_bright_resized.png'.format(capture_dir, file_name_prefix, tetrad_id)
                    
                    result_out_path = '{}/{}{}_1_combined_mask.png'.format(capture_dir, file_name_prefix, tetrad_id)
                    
                    predict_hist_out_path = '{}/{}{}_2_prediction_hist.png'.format(capture_dir, file_name_prefix, tetrad_id)
                    
                    cv2.imwrite(copied_bright_resized_out_path, copied_bright_resized)
                    fig_capture.savefig(result_out_path, edgecolor='none', bbox_inches='tight', pad_inches=0, dpi=300)
                    fig_hist.savefig(predict_hist_out_path, edgecolor='none', bbox_inches='tight', pad_inches=0, dpi=300)
            else:
                examiner_key = chr(cv2.waitKey(0)).upper()
            
        
    two_channel_dicts = (gr_tetrad_type_dict, bg_tetrad_type_dict, br_tetrad_type_dict)
    
    if not is_capture or is_purge:
        dump_data_to_pickle(bright_tetrad_typing_path, (tetrad_type_dict, two_channel_dicts))
    
    return (tetrad_type_dict, two_channel_dicts)

def get_two_channel_result_from_monad_dict(a_monad_dict):
    result_dict = {'AB': np.int64(0), 'AC_A': np.int64(0), 'AB_B': np.int64(0), 'AB_++': np.int64(0),
                   'BC': np.int64(0), 'BC_B': np.int64(0), 'BC_C': np.int64(0), 'AC_++': np.int64(0),
                   'AC': np.int64(0), 'AB_A': np.int64(0), 'AC_C': np.int64(0), 'BC_++': np.int64(0)
                    }
    valid_monad_type_set = set(['ABC', '+++', 'A++', '+BC', 'A+C', '+B+', 'AB+', '++C'])
    for k in valid_monad_type_set:
        if k not in a_monad_dict:
            a_monad_dict[k] = np.int64(0)
            
    result_dict['AB'] = a_monad_dict['ABC'] + a_monad_dict['AB+']
    result_dict['BC'] = a_monad_dict['ABC'] + a_monad_dict['+BC']
    result_dict['AC'] = a_monad_dict['ABC'] + a_monad_dict['A+C']

    result_dict['AC_A'] = a_monad_dict['A++'] + a_monad_dict['AB+']    
    result_dict['AB_A'] = a_monad_dict['A++'] + a_monad_dict['A+C']
    result_dict['BC_B'] = a_monad_dict['+B+'] + a_monad_dict['AB+']
    
    result_dict['AB_B'] = a_monad_dict['+B+'] + a_monad_dict['+BC']
    result_dict['BC_C'] = a_monad_dict['++C'] + a_monad_dict['A+C']
    result_dict['AC_C'] = a_monad_dict['++C'] + a_monad_dict['+BC']
    
    result_dict['AB_++'] = a_monad_dict['+++'] + a_monad_dict['++C']
    result_dict['AC_++'] = a_monad_dict['+++'] + a_monad_dict['+B+']
    result_dict['BC_++'] = a_monad_dict['+++'] + a_monad_dict['A++']
    return result_dict

def calculate_intervals_from_dict_monad_mode(tetrad_type_dict):
    result_dict = {'ABC': np.int64(0), '+++': np.int64(0), 'A++': np.int64(0), '+BC': np.int64(0), 'A+C': np.int64(0), '+B+': np.int64(0), 'AB+': np.int64(0), '++C': np.int64(0)}
    valid_tetrad_type_set = set(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
    for k in valid_tetrad_type_set:
        if k not in tetrad_type_dict:
            tetrad_type_dict[k] = np.int64(0)
    result_dict['ABC'] = tetrad_type_dict['A'] * 2 + tetrad_type_dict['B'] + tetrad_type_dict['C'] + tetrad_type_dict['D'] + tetrad_type_dict['E']
    result_dict['+++'] = tetrad_type_dict['A'] * 2 + tetrad_type_dict['B'] + tetrad_type_dict['C'] + tetrad_type_dict['D'] + tetrad_type_dict['F']
    result_dict['A++'] = tetrad_type_dict['B'] + tetrad_type_dict['E'] + tetrad_type_dict['G'] + tetrad_type_dict['H'] * 2 + tetrad_type_dict['J']
    result_dict['+BC'] = tetrad_type_dict['B'] + tetrad_type_dict['F'] + tetrad_type_dict['G'] + tetrad_type_dict['H'] * 2 + tetrad_type_dict['J']
    result_dict['A+C'] = tetrad_type_dict['D'] + tetrad_type_dict['F'] + tetrad_type_dict['J'] + tetrad_type_dict['K'] + tetrad_type_dict['L'] * 2
    result_dict['+B+'] = tetrad_type_dict['D'] + tetrad_type_dict['E'] + tetrad_type_dict['J'] + tetrad_type_dict['K'] + tetrad_type_dict['L'] * 2
    result_dict['AB+'] = tetrad_type_dict['C'] + tetrad_type_dict['F'] + tetrad_type_dict['G'] + tetrad_type_dict['I'] * 2 + tetrad_type_dict['K']
    result_dict['++C'] = tetrad_type_dict['C'] + tetrad_type_dict['E'] + tetrad_type_dict['G'] + tetrad_type_dict['I'] * 2 + tetrad_type_dict['K']
    # R25=ABC, R26=+B+, R27=A++, R28=AB+, R29=++C, R30=+BC, R31=A+C, R32=+++
    R25 = result_dict['ABC']
    R26 = result_dict['+B+']
    R27 = result_dict['A++']
    R28 = result_dict['AB+']
    R29 = result_dict['++C']
    R30 = result_dict['+BC']
    R31 = result_dict['A+C']
    R32 = result_dict['+++']
    n_total = R25 + R26 + R27 + R28 + R29 + R30 + R31 + R32
    first_interval = (R26+R27+R30+R31)/n_total * 100
    second_interval = (R26+R28+R29+R31)/n_total * 100
    third_interval = (R27+R28+R29+R30)/n_total * 100
    expected_DCOs = first_interval * second_interval /10000 * n_total
    CoC = (R26+R31)/expected_DCOs
    return first_interval, second_interval, third_interval, 1-CoC, result_dict, n_total

def calculate_intervals_from_dict(tetrad_type_dict):
    n_total = 0
    for k, v in tetrad_type_dict.items():
        if 'Z' == k:
            continue
        n_total = n_total + v
    
    multiplier_npd = 3
    interval_denom = float(n_total)
    if 0 == interval_denom:
        rg_interval = 0
        rc_interval = 0
        gc_interval = 0
    else:
        rg_interval = (0.5 * (tetrad_type_dict['B'] + tetrad_type_dict['D'] + tetrad_type_dict['E']
                              + tetrad_type_dict['F'] + tetrad_type_dict['G'] + tetrad_type_dict['K'])
                              + multiplier_npd * (tetrad_type_dict['H'] + tetrad_type_dict['L'] + tetrad_type_dict['J'])) / \
                              float(n_total) * 100
        gc_interval = (0.5 * (tetrad_type_dict['C'] + tetrad_type_dict['D'] + tetrad_type_dict['E']
                          + tetrad_type_dict['F'] + tetrad_type_dict['G'] + tetrad_type_dict['J'])
                          + multiplier_npd * (tetrad_type_dict['I'] + tetrad_type_dict['K'] + tetrad_type_dict['L'])) / \
                          float(n_total) * 100
                          
        rc_interval = (0.5 * (tetrad_type_dict['B'] + tetrad_type_dict['C'] + tetrad_type_dict['E']
                              + tetrad_type_dict['F'] + tetrad_type_dict['J'] + tetrad_type_dict['K'])
                              + multiplier_npd * (tetrad_type_dict['G'] + tetrad_type_dict['H'] + tetrad_type_dict['I'])) / \
                              float(n_total) * 100

    without_adj_CO_denom = float(tetrad_type_dict['A'] + tetrad_type_dict['B'] + tetrad_type_dict['H'])
    if 0 == without_adj_CO_denom:
        without_adj_CO = 0
    else:
        without_adj_CO = ((0.5 * tetrad_type_dict['B']) + multiplier_npd * tetrad_type_dict['H']) / \
        without_adj_CO_denom
    with_adj_CO_denom = float(tetrad_type_dict['C'] + tetrad_type_dict['D'] +
                                tetrad_type_dict['E'] + tetrad_type_dict['F'] +
                                tetrad_type_dict['G'] + tetrad_type_dict['I'] +
                                tetrad_type_dict['J'] + tetrad_type_dict['K'] +
                                tetrad_type_dict['L'])
    if 0 == with_adj_CO_denom:
        with_adj_CO = 0
    else:
        with_adj_CO = (0.5 * (tetrad_type_dict['D'] + tetrad_type_dict['E'] + tetrad_type_dict['F']
                           + tetrad_type_dict['G'] + tetrad_type_dict['K']) +
                           multiplier_npd * (tetrad_type_dict['J'] + tetrad_type_dict['L'])) / \
                           with_adj_CO_denom

    if 0 == with_adj_CO:
        interference_ratio = 0
    else:                                                      
#         interference_ratio = without_adj_CO/with_adj_CO
        interference_ratio = with_adj_CO/without_adj_CO
    return (rg_interval, rc_interval, gc_interval, interference_ratio, with_adj_CO, without_adj_CO, n_total)

def calculate_intervals_from_dict_two_in_monad_mode(result_dict):
#     print('[calculate_intervals_from_dict_in_monad_mode] physical channel: {}'.format(rep_physical_channel_str))
    valid_keys = set(['G', 'R', 'GR', 'None'])
    for v in valid_keys:
        if v not in result_dict:
            result_dict[v] = 0
    n_total = 0
    for key, n_count in result_dict.items():
        if key not in valid_keys:
            continue
        n_total = n_total + n_count
    rg_interval = (result_dict['R'] + result_dict['G']) / float(n_total) * 100
    return rg_interval, n_total

def calculate_intervals_from_dict_two(tetrad_type_dict):
    valid_keys = set(['P', 'T', 'NPD', 'None'])
    for v in valid_keys:
        if v not in tetrad_type_dict:
            tetrad_type_dict[v] = 0
    n_total = 0
    for k, v in tetrad_type_dict.items():
        if k not in valid_keys:
            continue
        n_total = n_total + v
    
    multiplier_npd = 3
    interval_denom = float(n_total)
    
    if 0 == interval_denom:
        rg_interval = 0
    else:
        rg_interval = (0.5 * (tetrad_type_dict['T'])
                              + multiplier_npd * tetrad_type_dict['NPD']) / \
                              interval_denom * 100

    manual_count_dict = {'G': np.int64(0), 'R': np.int64(0), 'GR': np.int64(0), 'None': np.int64(0)}
    manual_count_dict['GR'] += tetrad_type_dict['P'] * 2 + tetrad_type_dict['T']
    manual_count_dict['G'] += tetrad_type_dict['NPD'] * 2 + tetrad_type_dict['T']
    manual_count_dict['R'] += tetrad_type_dict['NPD'] * 2 + tetrad_type_dict['T']
    manual_count_dict['None'] += tetrad_type_dict['P'] * 2 + tetrad_type_dict['T']
    
    n_total_monad_mode = manual_count_dict['GR'] + manual_count_dict['G'] + manual_count_dict['R'] + manual_count_dict['None']
    
    rg_interval_dm = (manual_count_dict['G'] + manual_count_dict['R']) / float(n_total_monad_mode) * 100
    return (rg_interval, rg_interval_dm, manual_count_dict, n_total, n_total_monad_mode)

def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    np.random.shuffle(colors)
    return colors

def get_rep_physical_channel_str(physical_channels, target_str):
    rep_physical_channel_str = ''
    for line_name, physical_channel_str in physical_channels.items():
        if line_name in target_str:
            rep_physical_channel_str = physical_channel_str
            break
    return rep_physical_channel_str

def get_rep_line_name(physical_channels, target_str):
    rep_line_name = ''
    for line_name, physical_channel_str in physical_channels.items():
        if line_name in target_str:
            rep_line_name = line_name
            break
    return rep_line_name

def get_rgb_str(a_str):
    a_str = a_str.upper()
    return a_str.replace('Y', 'G').replace('C', 'B')

def calculate_intervals_from_dict_in_monad_mode(result_dict, rep_physical_channel_str=None):
    if None is rep_physical_channel_str:
        rep_physical_channel_str = 'RGB'
    n_total = 0
    valid_keys = set(['G', 'R', 'B', 'GR', 'BG', 'BR', 'RGB', 'None'])
    for v in valid_keys:
        if v not in result_dict:
            result_dict[v] = 0
    for key, n_count in result_dict.items():
        if key not in valid_keys:
            continue
        n_total = n_total + n_count
#     n_total = n_total - rgb_delta
    if 0 == n_total:
        return (0, 0, 0, 0, 0)
    else:
        rg_interval = (result_dict['G'] + result_dict['BG'] + result_dict['R'] + result_dict['BR']) / float(n_total) * 100.0
        rb_interval = (result_dict['R'] + result_dict['GR'] + result_dict['B'] + result_dict['BG']) / float(n_total) * 100.0
        gb_interval = (result_dict['G'] + result_dict['GR'] + result_dict['B'] + result_dict['BR']) / float(n_total) * 100.0
        
        the_first_interval_str = get_representative_interval_str(rep_physical_channel_str[0:2])
        the_second_interval_str = get_representative_interval_str(rep_physical_channel_str[1:3])
#         the_third_interval_str = get_representative_interval_str(rep_physical_channel_str[0] + rep_physical_channel_str[2])
        value_dict = {}
        value_dict['RG'] = rg_interval
        value_dict['GC'] = gb_interval
        value_dict['RC'] = rb_interval
        rep_physical_rgb_str = get_rgb_str(rep_physical_channel_str)
        if the_first_interval_str not in value_dict:
            the_first_interval_str = the_first_interval_str[1] + the_first_interval_str[0]
        first_interval = value_dict[the_first_interval_str]
        if the_second_interval_str not in value_dict:
            the_second_interval_str = the_second_interval_str[1] + the_second_interval_str[0]
        second_interval = value_dict[the_second_interval_str]
        
        expected_dcos = first_interval * second_interval * n_total / float(10000)
        n_b_only = 0
        if rep_physical_rgb_str[1] in result_dict:
            n_b_only = result_dict[rep_physical_rgb_str[1]]
        the_ac_str = get_rgb_str(rep_physical_rgb_str[0] + rep_physical_rgb_str[2])
        if the_ac_str not in result_dict:
            the_ac_str = the_ac_str[1] + the_ac_str[0]
        n_ac = 0
        if the_ac_str in result_dict:
            n_ac = result_dict[the_ac_str]

        the_ab_str = rep_physical_rgb_str[0] + rep_physical_rgb_str[1]
        if the_ab_str not in result_dict:
            the_ab_str = the_ab_str[1] + the_ab_str[0]
        
        # calculation based on the High-throughput analysis of meiotic crossover frequency and
        # interference via flowcytometry of fluorescent pollen in Arabidopsis thaliana
        observed_dcos = n_b_only + n_ac
        coefficient_of_coincidence = observed_dcos / expected_dcos
        interference = 1 - coefficient_of_coincidence

        return (rg_interval, rb_interval, gb_interval, interference, n_total)
    
def image_segmentation_three_channels(valid_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    is_align_purge = False
    if 'align' in purge_dict:
        is_align_purge = purge_dict['align']
    for a_prefix in valid_prefixes:
        get_pickle_path(a_prefix, 'aligned')
    
    print('[image_segmentation_three_channels] align images')
    Parallel(n_jobs=-1, backend="multiprocessing")(delayed(align_all_images)(a_prefix, False, is_align_purge, False) for a_prefix in valid_prefixes)
    
    print('[image_segmentation_three_channels] find single pollens')
    is_pollen_purge = False
    if 'pollen' in purge_dict:
        is_pollen_purge = purge_dict['pollen']
        
    keras_backend.clear_session()
    pollen_model = load_pollen_prediction_model()
    for a_prefix in valid_prefixes:
        collect_all_single_pollens(a_prefix, pollen_model, is_pollen_purge)

    print('[image_segmentation_three_channels] find tetrads')
    is_tetrad_purge = False
    if 'tetrad' in purge_dict:
        is_tetrad_purge = purge_dict['tetrad']
            
    keras_backend.clear_session()
    tetrad_model = load_tetrad_prediction_model()
    for a_prefix in valid_prefixes:
        collect_all_tetrads(a_prefix, tetrad_model, is_tetrad_purge)

    keras_backend.clear_session()

def get_representative_line_name(a_line_name, name_id):
    m = re.search(r'\d+$', a_line_name)
    # ends with a number
    if m is not None:
        return a_line_name
    rep_line_name = a_line_name
    if 0 == name_id:
        rep_line_name = rep_line_name[:-1]
    elif 1 == name_id:
        rep_line_name = rep_line_name[:-2] + rep_line_name[-1]
    elif 2 == name_id:
        rep_line_name = rep_line_name[:-1] + '-' + rep_line_name[-1]
    return rep_line_name

def detect_tetrads_from_all_three_channel_samples(valid_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    if 0 == len(valid_prefixes):
        return (intensity_thr, '', '')
    image_segmentation_three_channels(valid_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
    
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
    
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
    
    print('[detect_tetrads_from_all_three_channel_samples] determine tetrad types')
    detected_tetrads = []
    if is_merge_visualize or is_count_visualize:
        for a_prefix in valid_prefixes:
            detected_tetrad = determine_tetrad_type_in_a_single_pass(a_prefix, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
            detected_tetrads.append(detected_tetrad)
    else:
        detected_tetrads = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(determine_tetrad_type_in_a_single_pass)(a_prefix, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio) for a_prefix in valid_prefixes)

    per_sample_grouped_result = {}
    per_sample_gr_grouped_result = {}
    per_sample_bg_grouped_result = {}
    per_sample_br_grouped_result = {}
    
    per_line_grouped_result = {}
    per_line_gr_grouped_result = {}
    per_line_bg_grouped_result = {}
    per_line_br_grouped_result = {}
    all_file_interval_output_path = '{}/DeepTetrad_each_file_data_three_channels.csv'.format(root_path)
    with open(all_file_interval_output_path, 'w') as out:
        out.write('line\tfile_name\tn_total\tfirst_interval\tsecond_interval\tthird_interval\tIFR\twith_adj_CO\twithout_adj_CO\tprovider\n')
        for a_detected_tetrad_group in detected_tetrads:
            a_prefix, (result, two_channel_results) = a_detected_tetrad_group
            gr_result, bg_result, br_result = two_channel_results
            individual_name = a_prefix.split(os.sep)[-2]
            file_name = a_prefix.split(os.sep)[-1]
            rep_line_name = get_rep_line_name(physical_channels, individual_name)
            (rg_interval, rc_interval, gc_interval, interference_ratio, with_adj_CO, without_adj_CO, n_total) = calculate_intervals_from_dict(result)
            first_interval, second_interval, third_interval, interference_COC, result_dm, n_total_dm = calculate_intervals_from_dict_monad_mode(result)
            two_rg_interval, two_rg_interval_dm, manual_count_dict_rg, n_two_rg_total, n_total_monad_rg = calculate_intervals_from_dict_two(gr_result)
            two_gc_interval, two_gc_interval_dm, manual_count_dict_gc, n_two_gc_total, n_total_monad_gc = calculate_intervals_from_dict_two(bg_result)
            two_rc_interval, two_rc_interval_dm, manual_count_dict_rc, n_two_rc_total, n_total_monad_rc = calculate_intervals_from_dict_two(br_result)
            
            rep_physical_channel_str = get_rep_physical_channel_str(physical_channels, individual_name)
            order_dict, rev_order_dict = get_channel_order_dict(rep_physical_channel_str)
            
            rg_str = get_converted_interval_str(order_dict, 'RG')
            gc_str = get_converted_interval_str(order_dict, 'GC')
            rc_str = get_converted_interval_str(order_dict, 'RC')
            
            value_dict = {}
            value_dict[rg_str] = rg_interval
            value_dict[gc_str] = gc_interval
            value_dict[rc_str] = rc_interval
            
            two_value_dict = {}
            two_value_dict[rg_str] = two_rg_interval
            two_value_dict[gc_str] = two_gc_interval
            two_value_dict[rc_str] = two_rc_interval
            
            the_first_interval_str = get_representative_interval_str(rep_physical_channel_str[0:2])
            the_second_interval_str = get_representative_interval_str(rep_physical_channel_str[1:3])
            the_third_interval_str = get_representative_interval_str(rep_physical_channel_str[0] + rep_physical_channel_str[2])
            
            out.write('{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\tDeepTetrad\n'.format(
            rep_line_name, file_name, n_total,
            value_dict[the_first_interval_str],
            value_dict[the_second_interval_str],
            value_dict[the_third_interval_str],
            interference_ratio, with_adj_CO, without_adj_CO))
            
            out.write('{}\t{}\t{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\tDeepMonad\n'.format(
            rep_line_name, file_name, n_total_dm,
            first_interval,
            second_interval,
            third_interval,
            interference_COC, 0, 0))
            
            if individual_name not in per_sample_grouped_result:
                per_sample_grouped_result[individual_name] = Counter()
            if individual_name not in per_sample_gr_grouped_result:
                per_sample_gr_grouped_result[individual_name] = Counter()
            if individual_name not in per_sample_bg_grouped_result:
                per_sample_bg_grouped_result[individual_name] = Counter()
            if individual_name not in per_sample_br_grouped_result:
                per_sample_br_grouped_result[individual_name] = Counter()
            
            per_sample_grouped_result[individual_name] = per_sample_grouped_result[individual_name] + Counter(result)
            per_sample_gr_grouped_result[individual_name] = per_sample_gr_grouped_result[individual_name] + Counter(gr_result) 
            per_sample_bg_grouped_result[individual_name] = per_sample_bg_grouped_result[individual_name] + Counter(bg_result)
            per_sample_br_grouped_result[individual_name] = per_sample_br_grouped_result[individual_name] + Counter(br_result)
    
            if rep_line_name not in per_line_grouped_result:
                per_line_grouped_result[rep_line_name] = Counter()
            if rep_line_name not in per_line_gr_grouped_result:
                per_line_gr_grouped_result[rep_line_name] = Counter()
            if rep_line_name not in per_line_bg_grouped_result:
                per_line_bg_grouped_result[rep_line_name] = Counter()
            if rep_line_name not in per_line_br_grouped_result:
                per_line_br_grouped_result[rep_line_name] = Counter()
    
            per_line_grouped_result[rep_line_name] = per_line_grouped_result[rep_line_name] + Counter(result)
            per_line_gr_grouped_result[rep_line_name] = per_line_gr_grouped_result[rep_line_name] + Counter(gr_result)
            per_line_bg_grouped_result[rep_line_name] = per_line_bg_grouped_result[rep_line_name] + Counter(bg_result)
            per_line_br_grouped_result[rep_line_name] = per_line_br_grouped_result[rep_line_name] + Counter(br_result)
        
    all_interval_output_path = '{}/DeepTetrad_individual_intervals_three_channels.csv'.format(root_path)
    all_interferences_output_path = '{}/DeepTetrad_individual_interferences_three_channels.csv'.format(root_path)

    per_sample_types_dt_output_path = '{}/DeepTetrad_individual_types_three_channels.deeptetrad.csv'.format(root_path)
    per_sample_types_dm_output_path = '{}/DeepTetrad_individual_types_three_channels.deepmonad.csv'.format(root_path)
    
    with open(all_interval_output_path, 'w') as all_rf_out:
        with open(all_interferences_output_path, 'w') as all_if_out:
            with open(per_sample_types_dt_output_path, 'w') as out_type_dt:
                with open(per_sample_types_dm_output_path, 'w') as out_type_dm:
                
                    all_rf_out.write('line\tind_name\tn_total\tinterval\tprovider\n')
                    all_if_out.write('line\tind_name\tn_total\tinterference_ratio\tprovider\n')
                    out_type_dt.write('line\tind_name\tA\tB\tC\tD\tE\tF\tG\tH\tI\tJ\tK\tL\tn_total\tprovider\n')
                    out_type_dm.write('line\tind_name\t+++\tA++\t+BC\tA+C\t+B+\tAB+\t++C\tn_total\tprovider\n')
                    for individual_name, result in per_sample_grouped_result.items():
                        rep_physical_channel_str = get_rep_physical_channel_str(physical_channels, individual_name)
                        rep_line_name = get_rep_line_name(physical_channels, individual_name)
                        order_dict, rev_order_dict = get_channel_order_dict(rep_physical_channel_str)
                        (rg_interval, rc_interval, gc_interval, interference_ratio, with_adj_CO, without_adj_CO, n_total) = calculate_intervals_from_dict(result)
                        first_interval_dm, second_interval_dm, third_interval_dm, interference_COC, result_dm, n_total_dm = calculate_intervals_from_dict_monad_mode(result)
                        
                        gr_result = per_sample_gr_grouped_result[individual_name]
                        bg_result = per_sample_bg_grouped_result[individual_name]
                        br_result = per_sample_br_grouped_result[individual_name]
                        two_rg_interval, two_rg_interval_dm, manual_count_dict_rg, n_two_rg_total, n_total_monad_rg = calculate_intervals_from_dict_two(gr_result)
                        two_gc_interval, two_gc_interval_dm, manual_count_dict_gc, n_two_gc_total, n_total_monad_gc = calculate_intervals_from_dict_two(bg_result)
                        two_rc_interval, two_rc_interval_dm, manual_count_dict_rc, n_two_rc_total, n_total_monad_rc = calculate_intervals_from_dict_two(br_result)
                
                        rg_str = get_converted_interval_str(order_dict, 'RG')
                        rc_str = get_converted_interval_str(order_dict, 'RC')
                        gc_str = get_converted_interval_str(order_dict, 'GC')
                        value_dict = {}
                        value_dict[rg_str] = rg_interval
                        value_dict[rc_str] = rc_interval
                        value_dict[gc_str] = gc_interval
                        
                        two_value_dict = {}
                        two_value_dict[rg_str] = two_rg_interval
                        two_value_dict[rc_str] = two_rc_interval
                        two_value_dict[gc_str] = two_gc_interval
                
                        the_first_interval_str = get_representative_interval_str(rep_physical_channel_str[0:2])
                        the_second_interval_str = get_representative_interval_str(rep_physical_channel_str[1:3])
                        the_third_interval_str = get_representative_interval_str(rep_physical_channel_str[0] + rep_physical_channel_str[2])
                        
                        print('{} {}\t# total: {}\t{}: {:.3f}({:.3f})\t{}: {:.3f}({:.3f})\t{}: {:.3f}({:.3f})\tIFR: {:.3f}\tIFR_COC: {:.3f}\twith_adj_CO: {:.3f}\twithout_adj_CO: {:.3f}\tDeepTetrad'.format(
                            individual_name, rep_physical_channel_str, n_total,
                            the_first_interval_str, value_dict[the_first_interval_str], two_value_dict[the_first_interval_str],
                            the_second_interval_str, value_dict[the_second_interval_str], two_value_dict[the_second_interval_str],
                            the_third_interval_str, value_dict[the_third_interval_str], two_value_dict[the_third_interval_str],
                            interference_ratio, interference_COC, with_adj_CO, without_adj_CO))
                        
                        out_type_dt.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tDeepTetrad\n'.format(
                            rep_line_name, individual_name, result['A'], result['B'], result['C'], result['D'], result['E'], result['F'], result['G'],
                            result['H'], result['I'], result['J'], result['K'], result['L'], n_total))
                        
                        out_type_dm.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\tDeepMonad\n'.format(
                            rep_line_name, individual_name, result_dm['+++'], result_dm['A++'], result_dm['+BC'], result_dm['A+C'],
                            result_dm['+B+'], result_dm['AB+'], result_dm['++C'], n_total_dm))
                        
                        rep_line_name_first = get_representative_line_name(rep_line_name, 0)
                        rep_line_name_second = get_representative_line_name(rep_line_name, 1)
                        rep_line_name_third = get_representative_line_name(rep_line_name, 2)
                        all_rf_out.write('{}\t{}\t{}\t{}\tDeepTetrad\n'.format(rep_line_name_first, individual_name, n_total, value_dict[the_first_interval_str]))
                        all_rf_out.write('{}\t{}\t{}\t{}\tDeepTetrad\n'.format(rep_line_name_second, individual_name, n_total, value_dict[the_second_interval_str]))
                        all_rf_out.write('{}\t{}\t{}\t{}\tDeepTetrad\n'.format(rep_line_name_third, individual_name, n_total, value_dict[the_third_interval_str]))
                        
                        all_rf_out.write('{}\t{}\t{}\t{}\tDeepMonad\n'.format(rep_line_name_first, individual_name, n_total_dm, first_interval_dm))
                        all_rf_out.write('{}\t{}\t{}\t{}\tDeepMonad\n'.format(rep_line_name_second, individual_name, n_total_dm, second_interval_dm))
                        all_rf_out.write('{}\t{}\t{}\t{}\tDeepMonad\n'.format(rep_line_name_third, individual_name, n_total_dm, third_interval_dm))
                        
                        all_if_out.write('{}\t{}\t{}\t{:.3f}\tDeepTetrad\n'.format(rep_line_name, individual_name, n_total, interference_ratio))
                        all_if_out.write('{}\t{}\t{}\t{:.3f}\tDeepMonad\n'.format(rep_line_name, individual_name, n_total_dm, interference_COC))

    for line_name, result in per_line_grouped_result.items():
        print('{}\t{}'.format(line_name, sorted(result.items())))
    
    summed_dt_output_path = '{}/DeepTetrad_line_intervals_three_channels.deeptetrad.csv'.format(root_path)
    summed_dm_output_path = '{}/DeepTetrad_line_intervals_three_channels.deepmonad.csv'.format(root_path)
    with open(summed_dt_output_path, 'w') as out_summed_rf_dt:
        with open(summed_dm_output_path, 'w') as out_summed_rf_dm:
            out_summed_rf_dt.write('line\tn_total\tinterval\tprovider\n')
            out_summed_rf_dm.write('line\tn_total\tinterval\tprovider\n')
            for line_name, result in per_line_grouped_result.items():
                rep_physical_channel_str = get_rep_physical_channel_str(physical_channels, line_name)
                order_dict, rev_order_dict = get_channel_order_dict(rep_physical_channel_str)
                (rg_interval, rc_interval, gc_interval, interference_ratio, with_adj_CO, without_adj_CO, n_total) = calculate_intervals_from_dict(result)
                first_interval_dm, second_interval_dm, third_interval_dm, interference_COC, result_dm, n_total_dm = calculate_intervals_from_dict_monad_mode(result)
        
                gr_result = per_line_gr_grouped_result[line_name]
                bg_result = per_line_bg_grouped_result[line_name]
                br_result = per_line_br_grouped_result[line_name]
                two_rg_interval, two_rg_interval_dm, manual_count_dict_rg, n_two_rg_total, n_total_monad_rg = calculate_intervals_from_dict_two(gr_result)
                two_gc_interval, two_gc_interval_dm, manual_count_dict_gc, n_two_gc_total, n_total_monad_gc = calculate_intervals_from_dict_two(bg_result)
                two_rc_interval, two_rc_interval_dm, manual_count_dict_rc, n_two_rc_total, n_total_monad_rc = calculate_intervals_from_dict_two(br_result)
        
                rg_str = get_converted_interval_str(order_dict, 'RG')
                rc_str = get_converted_interval_str(order_dict, 'RC')
                gc_str = get_converted_interval_str(order_dict, 'GC')
                value_dict = {}
                value_dict[rg_str] = rg_interval
                value_dict[rc_str] = rc_interval
                value_dict[gc_str] = gc_interval
                
                two_value_dict = {}
                two_value_dict[rg_str] = two_rg_interval
                two_value_dict[rc_str] = two_rc_interval
                two_value_dict[gc_str] = two_gc_interval
        
                the_first_interval_str = get_representative_interval_str(rep_physical_channel_str[0:2])
                the_second_interval_str = get_representative_interval_str(rep_physical_channel_str[1:3])
                the_third_interval_str = get_representative_interval_str(rep_physical_channel_str[0] + rep_physical_channel_str[2])
                rep_line_name_first = get_representative_line_name(line_name, 0)
                rep_line_name_second = get_representative_line_name(line_name, 1)
                rep_line_name_third = get_representative_line_name(line_name, 2)
                print('{} {}\t# total: {}\t{}: {:.3f}({:.3f})\t{}: {:.3f}({:.3f})\t{}: {:.3f}({:.3f})\tIFR: {:.3f}\tIRF_COC: {:.3f}\twith_adj_CO: {:.3f}\twithout_adj_CO: {:.3f}\tDeepTetrad'.format(
                    line_name, rep_physical_channel_str, n_total,
                    the_first_interval_str, value_dict[the_first_interval_str], first_interval_dm,
                    the_second_interval_str, value_dict[the_second_interval_str], second_interval_dm,
                    the_third_interval_str, value_dict[the_third_interval_str], third_interval_dm,
                    interference_ratio, interference_COC, with_adj_CO, without_adj_CO))
                out_summed_rf_dt.write('{}\t{}\t{:.3f}\tDeepTetrad\n'.format(rep_line_name_first, n_total, value_dict[the_first_interval_str]))
                out_summed_rf_dt.write('{}\t{}\t{:.3f}\tDeepTetrad\n'.format(rep_line_name_second, n_total, value_dict[the_second_interval_str]))
                out_summed_rf_dt.write('{}\t{}\t{:.3f}\tDeepTetrad\n'.format(rep_line_name_third, n_total, value_dict[the_third_interval_str]))
                
                out_summed_rf_dm.write('{}\t{}\t{:.3f}\tDeepMonad\n'.format(rep_line_name_first, n_total_dm, first_interval_dm))
                out_summed_rf_dm.write('{}\t{}\t{:.3f}\tDeepMonad\n'.format(rep_line_name_second, n_total_dm, second_interval_dm))
                out_summed_rf_dm.write('{}\t{}\t{:.3f}\tDeepMonad\n'.format(rep_line_name_third, n_total_dm, third_interval_dm))
            
    return (per_sample_grouped_result, per_line_grouped_result)

def detect_tetrads_from_all_two_channel_samples(valid_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    if 0 == len(valid_prefixes):
        return (intensity_thr, '', '')
    
    # align all images in the prefixes
    is_align_purge = False
    if 'align' in purge_dict:
        is_align_purge = purge_dict['align']
        
    Parallel(n_jobs=-1, backend="multiprocessing")(delayed(align_all_images_two)(a_prefix, False, is_align_purge) for a_prefix in valid_prefixes)
    
    print('[detect_tetrads_from_all_two_channel_samples] find pollens')
    is_pollen_purge = False
    if 'pollen' in purge_dict:
        is_pollen_purge = purge_dict['pollen']
    # find all pollens
    keras_backend.clear_session()
    pollen_model = load_pollen_prediction_model()
    for a_prefix in valid_prefixes:
        collect_all_single_pollens(a_prefix, pollen_model, is_pollen_purge)
    print('[detect_tetrads_from_all_two_channel_samples] find tetrads')
    is_tetrad_purge = False
    if 'tetrad' in purge_dict:
        is_tetrad_purge = purge_dict['tetrad']    
    
    # find all tetrads
    keras_backend.clear_session()
    tetrad_model = load_tetrad_prediction_model()
    
    for a_prefix in valid_prefixes:
        collect_all_tetrads(a_prefix, tetrad_model, is_tetrad_purge)
    
    print('[detect_tetrads_from_all_two_channel_samples] determine tetrad types')
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
    
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
    detected_tetrads = []
    if is_merge_visualize or is_count_visualize:
        for a_prefix in valid_prefixes:
            detected_tetrad = determine_tetrad_type_in_a_single_pass_two(a_prefix, root_path, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
            if None is detected_tetrad[1]:
                continue
            detected_tetrads.append(detected_tetrad)
    else:
        detected_tetrads = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(determine_tetrad_type_in_a_single_pass_two)(a_prefix, root_path, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio) for a_prefix in valid_prefixes)

    per_sample_grouped_result = {}
    per_line_grouped_result = {}
    each_file_interval_output_path = '{}/DeepTetrad_each_file_data_two_channels.csv'.format(root_path)
    with open(each_file_interval_output_path, 'w') as out:
        out.write('line\tfile_name\tn_total\tinterval\provider\n')
        for a_detected_tetrad_group in detected_tetrads:
            a_prefix, (result, _) = a_detected_tetrad_group
            individual_name = a_prefix.split(os.sep)[-2]
            file_name = a_prefix.split(os.sep)[-1]
            rep_line_name = get_rep_line_name(physical_channels, individual_name)
            if rep_line_name not in per_line_grouped_result:
                per_line_grouped_result[rep_line_name] = Counter()
            if individual_name not in per_sample_grouped_result:
                per_sample_grouped_result[individual_name] = Counter()
            per_sample_grouped_result[individual_name] = per_sample_grouped_result[individual_name] + Counter(result)
            per_line_grouped_result[rep_line_name] = per_line_grouped_result[rep_line_name] + Counter(result)
            rg_interval, rg_interval_dm, result_dm, n_total, n_total_dm = calculate_intervals_from_dict_two(result)
            out.write('{}\t{}\t{}\t{:.3f}\tDeepTetrad\n').format(rep_line_name, file_name, n_total, rg_interval)
            out.write('{}\t{}\t{}\t{:.3f}\tDeepMonad\n').format(rep_line_name, file_name, n_total_dm, rg_interval_dm)
    
    all_rf_output_path = '{}/DeepTetrad_individual_intervals_two_channels.csv'.format(root_path)
    per_sample_types_dt_output_path = '{}/DeepTetrad_individual_types_two_channels.deeptetrad.csv'.format(root_path)
    per_sample_types_dm_output_path = '{}/DeepTetrad_individual_types_two_channels.deepmonad.csv'.format(root_path)
    with open(all_rf_output_path, 'w') as out_rf:
        with open(per_sample_types_dt_output_path, 'w') as out_types_dt:
            with open(per_sample_types_dm_output_path, 'w') as out_types_dm:
                out_rf.write('line\tind_name\tn_total\tfirst_interval\tprovider\n')
                out_types_dt.write('line\tind_name\tNPD\tT\tn_total\tprovider\n')
                out_types_dm.write('line\tind_name\tAB\t++\tA\tB\tn_total\tprovider\n')
                for individual_name, result in per_sample_grouped_result.items():
                    rep_line_name = get_rep_line_name(physical_channels, individual_name)
                    rg_interval, rg_interval_dm, result_dm, n_total, n_total_dm = calculate_intervals_from_dict_two(result)
                    out_rf.write('{}\t{}\t{}\t{}\tDeepTetrad\n'.format(rep_line_name, individual_name, n_total, rg_interval))
                    out_rf.write('{}\t{}\t{}\t{}\tDeepMonad\n'.format(rep_line_name, individual_name, n_total_dm, rg_interval_dm))
                    out_types_dt.write('{}\t{}\t{}\t{}\t{}\tDeepTetrad\n'.format(rep_line_name, individual_name, result['NPD'], result['T'], n_total))
                    out_types_dm.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\tDeepMonad\n'.format(rep_line_name, individual_name, result_dm['GR'], result_dm['None'], result_dm['G'], result_dm['R'], n_total_dm))
                    print('{}\t{:.3f}({:.3f})\t# tetrads: {}\tDeepTetrad'.format(individual_name, rg_interval, rg_interval_dm, n_total))
            
    for line_name, result in per_line_grouped_result.items():
        print('{}\t{}'.format(line_name, sorted(result.items())))
            
    for line_name, result in per_line_grouped_result.items():
        rg_interval, rg_interval_dm, result_dm, n_total, n_total_dm = calculate_intervals_from_dict_two(result)
        print('{}\t{:.3f}({:.3f})\t# tetrads: {}\tDeepTetrad'.format(line_name, rg_interval, rg_interval_dm, n_total))
        
    return (per_sample_grouped_result, per_line_grouped_result)

def count_all_tetrads(valid_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    valid_two_channel_prefixes, valid_three_channel_prefixes = valid_prefixes
    print('[count_all_tetrads] three channels')
    detect_tetrads_from_all_three_channel_samples(valid_three_channel_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
    print('[count_all_tetrads] two channels')
    detect_tetrads_from_all_two_channel_samples(valid_two_channel_prefixes, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
    
def detect_valid_tetrads(a_prefix, root_path):
#     print('[detect_valid_tetrads] {}'.format(a_prefix))
    path_dict = get_path_dict(a_prefix)
    collect_all_valid_tetrads(path_dict, root_path)
    
def determine_tetrad_types(a_prefix, root_path, intensity_thr):
#     print('[determine_tetrad_types] {}'.format(a_prefix))
    path_dict = get_path_dict(a_prefix)
    return (a_prefix, collect_all_tetrad_types(a_prefix, path_dict, root_path, intensity_thr, True))

def load_pollen_prediction_model():
    config = PollenInferenceConfig()
    logs_dir = './pollen/logs/'
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs_dir)
    cur_parent = os.path.split(__file__)[0]
    weights_path = '{}{}'.format(cur_parent, '/pollen/single_pollen_weights.h5')
    if not os.path.exists(weights_path):
        download_weight_file(weights_path)
    model.load_weights(weights_path, by_name=True)
    return model

def load_tetrad_prediction_model():
    config = TetradInferenceConfig()
    logs_dir = './pollen/logs/'
    model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=logs_dir)
    cur_parent = os.path.split(__file__)[0]
    weights_path = '{}{}'.format(cur_parent, '/pollen/tetrad_weights.h5')
    if not os.path.exists(weights_path):
        download_weight_file(weights_path)
    model.load_weights(weights_path, by_name=True)
    return model

def download_weight_file(a_local_path):
    TETRAD_MODEL_URL = 'https://github.com/abysslover/deeptetrad/raw/master/src/deeptetrad/pollen/tetrad_weights.h5'
    POLLEN_MODEL_URL = 'https://github.com/abysslover/deeptetrad/raw/master/src/deeptetrad/pollen/single_pollen_weights.h5'
    if 'tetrad_weights' in a_local_path:
        target_model_URL = TETRAD_MODEL_URL
        print('[download_weight_file] start downloading tetrad weights')
    elif 'pollen_weights' in a_local_path:
        target_model_URL = POLLEN_MODEL_URL
        print('[download_weight_file] start downloading pollen weights')
    parent_dir = os.path.split(a_local_path)[0]
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    with urllib.request.urlopen(target_model_URL) as resp, open(a_local_path, 'wb') as out:
        shutil.copyfileobj(resp, out)

def reset_keras_session():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
    print(gc.collect())
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "-1"
    set_session(tf.Session(config=config))

def validate_tetrad_type_in_a_single_pass_two(a_prefix, root_path, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    path_dict = get_path_dict_two(a_prefix)
    is_merge_purge = False
    if 'merge' in purge_dict:
        is_merge_purge = purge_dict['merge']
        
    is_merge_capture = False
    if 'merge' in capture_dict:
        is_merge_capture = capture_dict['merge']
    
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
        
    collect_all_valid_tetrads(a_prefix, root_path, debug_set, is_merge_purge, is_merge_capture, is_merge_visualize)
    is_count_purge = False
    if 'count' in purge_dict:
        is_count_purge = purge_dict['count']
    
    is_count_capture = False
    if 'count' in capture_dict:
        is_count_capture = capture_dict['count']
        
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
        
    return (a_prefix, validate_all_tetrad_types_two(a_prefix, path_dict, root_path, intensity_thr, debug_set, desired_min_area_ratio, is_count_purge, is_count_capture, is_count_visualize))

def determine_tetrad_type_in_a_single_pass_two_enlarge(a_prefix, root_path, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    path_dict = get_path_dict_two(a_prefix)
    is_merge_purge = False
    if 'merge' in purge_dict:
        is_merge_purge = purge_dict['merge']
        
    is_merge_capture = False
    if 'merge' in capture_dict:
        is_merge_capture = capture_dict['merge']
    
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
        
    collect_all_valid_tetrads(a_prefix, root_path, debug_set, is_merge_purge, is_merge_capture, is_merge_visualize)
    is_count_purge = False
    if 'count' in purge_dict:
        is_count_purge = purge_dict['count']
    
    is_count_capture = False
    if 'count' in capture_dict:
        is_count_capture = capture_dict['count']
        
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
        
    return (a_prefix, collect_all_tetrad_types_two_enlarge(a_prefix, path_dict, root_path, intensity_thr, debug_set, desired_min_area_ratio, is_count_purge, is_count_capture, is_count_visualize))

def determine_tetrad_type_in_a_single_pass_two(a_prefix, root_path, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    path_dict = get_path_dict_two(a_prefix)
    is_merge_purge = False
    if 'merge' in purge_dict:
        is_merge_purge = purge_dict['merge']
        
    is_merge_capture = False
    if 'merge' in capture_dict:
        is_merge_capture = capture_dict['merge']
    
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
        
    collect_all_valid_tetrads(a_prefix, root_path, debug_set, is_merge_purge, is_merge_capture, is_merge_visualize)
    is_count_purge = False
    if 'count' in purge_dict:
        is_count_purge = purge_dict['count']
    
    is_count_capture = False
    if 'count' in capture_dict:
        is_count_capture = capture_dict['count']
        
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
        
    return (a_prefix, collect_all_tetrad_types_two(a_prefix, path_dict, root_path, intensity_thr, debug_set, desired_min_area_ratio, is_count_purge, is_count_capture, is_count_visualize))


def validate_tetrad_type_in_a_single_pass(a_prefix, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    path_dict = get_path_dict(a_prefix)
    is_merge_purge = False
    if 'merge' in purge_dict:
        is_merge_purge = purge_dict['merge']
    is_merge_capture = False
    if 'merge' in capture_dict:
        is_merge_capture = capture_dict['merge']
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
    collect_all_valid_tetrads(a_prefix, root_path, debug_set, is_merge_purge, is_merge_capture, is_merge_visualize)
    is_count_purge = False
    if 'count' in purge_dict:
        is_count_purge = purge_dict['count']
    
    is_count_capture = False
    if 'count' in capture_dict:
        is_count_capture = capture_dict['count']
    
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
    
    return (a_prefix, validate_all_tetrad_types(a_prefix, path_dict, root_path, physical_channels, intensity_thr, debug_set, desired_min_area_ratio, is_count_purge, is_count_capture, is_count_visualize))

def determine_tetrad_type_in_a_single_pass(a_prefix, root_path, physical_channels, intensity_thr, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    path_dict = get_path_dict(a_prefix)
    is_merge_purge = False
    if 'merge' in purge_dict:
        is_merge_purge = purge_dict['merge']
    is_merge_capture = False
    if 'merge' in capture_dict:
        is_merge_capture = capture_dict['merge']
    is_merge_visualize = False
    if 'merge' in visualize_dict:
        is_merge_visualize = visualize_dict['merge']
    collect_all_valid_tetrads(a_prefix, root_path, debug_set, is_merge_purge, is_merge_capture, is_merge_visualize)
    is_count_purge = False
    if 'count' in purge_dict:
        is_count_purge = purge_dict['count']
    
    is_count_capture = False
    if 'count' in capture_dict:
        is_count_capture = capture_dict['count']
    
    is_count_visualize = False
    if 'count' in visualize_dict:
        is_count_visualize = visualize_dict['count']
    
    return (a_prefix, collect_all_tetrad_types(a_prefix, path_dict, root_path, physical_channels, intensity_thr, debug_set, desired_min_area_ratio, is_count_purge, is_count_capture, is_count_visualize))
    
@timeit
def run_tetrad_count_all_process(root_path, physical_channels, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio):
    tf_config = tf.ConfigProto()
    tf_config.inter_op_parallelism_threads = 1
    keras_backend.set_session(tf.Session(config=tf_config))
    prefixes = get_all_prefixes(root_path)
    valid_two_channel_prefixes, valid_three_channel_prefixes = get_valid_prefixes(prefixes)
    
    valid_prefixes = [valid_two_channel_prefixes, valid_three_channel_prefixes]
    print('[run_tetrad_count_all_process] three channels: {}'.format(valid_three_channel_prefixes))
    print('[run_tetrad_count_all_process] two channels: {}'.format(valid_two_channel_prefixes))
    count_all_tetrads(valid_prefixes, root_path, physical_channels, 0.4, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
    
def load_physical_locations(a_path):
    result_dict = {}
    with open(a_path) as fin:
        for line in fin:
            line = line.strip()
            tokens = line.split('\t')
            result_dict[tokens[0]] = tokens[1] 
    return result_dict

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Run DeepTetrad')
    parser.add_argument('--physical_loc', required=False,
                        default=None,
                        metavar="/path/to/physical_loc.txt",
                        help='Directory of the physical locations')
    parser.add_argument('--path', required=True,
                        metavar="path to fluorescent images",
                        help='Images to run DeepTetrad')
    
    args = parser.parse_args()
    root_path = args.path
    if None is args.physical_loc:
        physical_channels = {'I1bc': 'GRC', 'I1fg': 'GCR', 'I2ab': 'CGR', 'I2fg': 'RGC', 'I3bc': 'CGR', 'I5ab': 'RGC', 'CEN3': 'RG'}
    else:
        physical_channels = load_physical_locations(args.physical_loc)
    desired_min_area_ratio = 4
    purge_dict = {'align': False, 'pollen': True, 'tetrad': True, 'merge': True, 'count': True}
    capture_dict = {'merge': False, 'count': False}
    visualize_dict = {'merge': False, 'count': False}
    debug_set = set()
    run_tetrad_count_all_process(root_path, physical_channels, purge_dict, capture_dict, visualize_dict, debug_set, desired_min_area_ratio)
    
if __name__ == "__main__":
    main()
    
