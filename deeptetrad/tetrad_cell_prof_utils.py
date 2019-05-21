'''
Created on Feb 25, 2019

@author: vincent
'''
import pyopencl
import silx
import cv2
import numpy as np
import math

from centrosome.cpmorphology import grey_erosion, grey_dilation
from centrosome.filter import convex_hull_transform
from centrosome.filter import stretch
import centrosome.smooth
import centrosome.propagate

import matplotlib.pyplot as plt
import matplotlib.cm
import skimage.morphology
import skimage.filters
import scipy.ndimage as scind
import scipy.sparse
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
from silx.image import sift

SM_CONVEX_HULL = "Convex Hull"
SM_FIT_POLYNOMIAL = "Fit Polynomial"
SM_MEDIAN_FILTER = "Median Filter"
SM_GAUSSIAN_FILTER = "Gaussian Filter"
SM_TO_AVERAGE = "Smooth to Average"
SM_SPLINES = "Splines"
FI_AUTOMATIC = "Automatic"
FI_OBJECT_SIZE = "Object size"
DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"
ROBUST_FACTOR = .02  # For rescaling, take 2nd percentile value
A_SIMILARLY = 'Similarly'
A_SEPARATELY = 'Separately'
C_CROP = "Crop to aligned region"
C_PAD = "Pad images"

UN_INTENSITY = "Intensity"
UN_SHAPE = "Shape"
UN_LOG = "Laplacian of Gaussian"
UN_NONE = "None"

FH_NEVER = "Never"
FH_THRESHOLDING = "After both thresholding and declumping"
FH_DECLUMP = "After declumping only"

WA_INTENSITY = "Intensity"
WA_SHAPE = "Shape"
WA_PROPAGATE = "Propagate"
WA_NONE = "None"

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

"""subplot_imshow cplabels dictionary key: color to use for outlines"""
CPLD_OUTLINE_COLOR = "outline_color"
"""subplot_imshow cplabels dictionary key: display mode - outlines or alpha"""
CPLD_MODE = "mode"
"""subplot_imshow cplabels mode value: show outlines of objects"""
CPLDM_OUTLINES = "outlines"
"""subplot_imshow cplabels mode value: show objects as an alpha-transparent color overlay"""
CPLDM_ALPHA = "alpha"
"""subplot_imshow cplabels mode value: draw outlines using matplotlib plot"""
CPLDM_LINES = "lines"
"""subplot_imshow cplabels mode value: don't show these objects"""
CPLDM_NONE = "none"
"""subplot_imshow cplabels dictionary key: line width of outlines"""
CPLD_LINE_WIDTH = "line_width"
"""subplot_imshow cplabels dictionary key: color map to use in alpha mode"""
CPLD_ALPHA_COLORMAP = "alpha_colormap"
"""subplot_imshow cplabels dictionary key: alpha value to use in overlay mode"""
CPLD_ALPHA_VALUE = "alpha_value"
"""subplot_imshow cplabels dictionary key: show (TRUE) or hide (False)"""
CPLD_SHOW = "show"
# maximum_object_count = 500
# smoothing_filter_size = 10
# suppress_diameter = 10
import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print ('[{}]  {:.2f} s'.format(method.__name__, (te - ts)))
        return result
    return timed

def remove_other_channels(a_path, contributions):
    input_image = cv2.imread(a_path).astype(np.float64)
    channels, contributions = zip(*channels_and_contributions(contributions))
    channels = np.array(channels, int)
    contributions = np.array(contributions)
    selected_id = np.argwhere(contributions == 1)
    selected_id = selected_id[0][0]
    output_image = np.sum(input_image[:, :, channels] *
                              contributions[np.newaxis, np.newaxis, :], 2)
#     output_image[output_image < 15] = 0
#     output_image[output_image > 220] = 0
    result_image = np.zeros_like(input_image)
    result_image[:,:,selected_id] = output_image
    
    return result_image

def remove_noise(file_path, contribution, suppress_diameter, is_bright):
    if is_bright:
        gray_img = color_to_gray(file_path, (1.0, 1.0, 1.0))
        
        img_after_math = suppress_by_radius(gray_img, suppress_diameter)
#         img_after_math = gray_img
    else:
#         remove_contribution = np.array(contribution)
#         remove_contribution[remove_contribution == 0] = -1.0
#         org_img = remove_other_channels(file_path, remove_contribution)
#         gray_img = color_to_gray_from_image(org_img, contribution)
        gray_img = color_to_gray(file_path, contribution)
        img_after_math = gray_img
#         img_after_math = np_standardized_array(gray_img)
#         illum_img = correct_illumination_calculate(gray_img)
#         corr_img = correct_illumination_apply(DOS_DIVIDE, gray_img, illum_img)
#         img_after_math = apply_simple_math(gray_img, corr_img)
#         img_after_math = corr_img
#         img_after_math = illum_img
    return img_after_math

def np_standardized_array(arr):
    return np_divide_by_nonzero(arr - np.mean(arr), np.std(arr))
    
def np_divide_by_nonzero(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def np_normalized_array(a):
    min_a = np.min(a)
    max_a = np.max(a)
    return np_divide_by_nonzero(a - min_a, max_a - min_a)
    
def channels_and_contributions(contributions):
    """Return tuples of channel indexes and their relative contributions

    Used when combining channels to find the channels to combine
    """
    return [(i, contribution) for i, contribution in enumerate(
            (contributions[0], contributions[1], contributions[2]))]

def color_to_gray_from_image(input_image, contributions):
    input_image = input_image.astype(np.float64)
    input_image /= 255.0
    
#     print('[color_to_gray] input.shape: {}, dtype: {}, min: {}, max: {}'.format(input_image.shape, input_image.dtype, np.min(input_image), np.max(input_image)))
    denominator = sum(contributions)
    channels, contributions = zip(*channels_and_contributions(contributions))
    channels = np.array(channels, int)
    contributions = np.array(contributions) / denominator
    
    output_image = np.sum(input_image[:, :, channels] *
                              contributions[np.newaxis, np.newaxis, :], 2)
    output_image = output_image.astype(np.float64)
    output_image[output_image < 0] = 0
    return output_image

def color_to_gray(a_path, contributions):
    input_image = cv2.imread(a_path).astype(np.float64)
    input_image /= 255.0
    
#     print('[color_to_gray] input.shape: {}, dtype: {}, min: {}, max: {}'.format(input_image.shape, input_image.dtype, np.min(input_image), np.max(input_image)))
    denominator = sum(contributions)
    channels, contributions = zip(*channels_and_contributions(contributions))
    channels = np.array(channels, int)
    contributions = np.array(contributions) / denominator
    
    output_image = np.sum(input_image[:, :, channels] *
                              contributions[np.newaxis, np.newaxis, :], 2)
    output_image = output_image.astype(np.float64)
    output_image[output_image < 0] = 0
    return output_image

def suppress_by_radius(image, object_size):
    radius = object_size / 2
#     print('[suppress_by_radius] Suppress: {}'.format(radius))
#     print('[suppress_by_radius] before suppress input.shape: {}, min: {}, max: {}'.format(image.shape, np.min(image), np.max(image)))
    mask = np.ones(image.shape[:2], bool)
    data = get_mask(image, mask)
    selem = get_structuring_element(radius, False)
    result = skimage.morphology.opening(data, selem)
    result = unmask(result, image, mask)
#     print('[suppress_by_radius] after suppress result.shape: {}, min: {}, max: {}'.format(result.shape, np.min(result), np.max(result)))
    return result

def get_mask(pixel_data, mask):
    data = np.zeros_like(pixel_data)
    data[mask] = pixel_data[mask]
    return data

def get_structuring_element(radius, volumetric):
#     print('[get_structuring_element] volumetric {}'.format(volumetric))
    if volumetric:
        return skimage.morphology.ball(radius)
    return skimage.morphology.disk(radius)

def unmask(data, pixel_data, mask):
    data[~mask] = pixel_data[~mask]
    return data

def correct_illumination_calculate(orig_image):
    mask = np.ones(orig_image.shape[:2], bool)
    smooth_function_image = smooth_with_convex_hull(orig_image, mask)
    scaled = apply_scaling(smooth_function_image)
    # for illumination correction, we want the smoothed function to extend beyond the mask.
#     output_image.mask = np.ones(output_image.pixel_data.shape[:2], bool)
    return scaled

def smooth_with_convex_hull(pixel_data, mask):
    '''Use the convex hull transform to smooth the image'''
    #
    # Apply an erosion, then the transform, then a dilation, heuristically
    # to ignore little spikey noisy things.
#     local_mask = np.ones(pixel_data[:2], bool)
    image = grey_erosion(pixel_data, 2, mask)
    image = convex_hull_transform(image, mask=mask)
    image = grey_dilation(image, 2, mask)
    
    return image

def apply_scaling(image):
    def scaling_fn_2d(pixel_data):
        sorted_pixel_data = pixel_data[pixel_data > 0]
        if sorted_pixel_data.shape[0] == 0:
            return pixel_data
        sorted_pixel_data.sort()
        idx = int(sorted_pixel_data.shape[0] * ROBUST_FACTOR)
        robust_minimum = sorted_pixel_data[idx]
        pixel_data = pixel_data.copy()
        pixel_data[pixel_data < robust_minimum] = robust_minimum
        if robust_minimum == 0:
            return pixel_data
        return pixel_data / robust_minimum

    if image.ndim == 2:
        output_pixels = scaling_fn_2d(image)
    else:
        output_pixels = np.dstack([scaling_fn_2d(x) for x in image.transpose(2, 0, 1)])
    return output_pixels

def correct_illumination_apply(divide_or_subtract, orig_image, illum_function):
    if divide_or_subtract == DOS_DIVIDE:
        output_pixels = orig_image / illum_function
    elif divide_or_subtract == DOS_SUBTRACT:
        output_pixels = orig_image - illum_function
        output_pixels[output_pixels < 0] = 0
    return output_pixels

def apply_simple_math(first_image, second_image):
    output_pixel_data = first_image * 0.5 + second_image
    output_pixel_data[output_pixel_data < 0] = 0
    output_pixel_data[output_pixel_data > 1] = 1
    return output_pixel_data

# @timing
def identify_primary_objects(image_and_mask, threshold_smoothing_scale, size_range_min, size_range_max, smoothing_filter_size, maximum_object_count, border_exclude = None, size_exclude = None, verbose=None):
    if None is verbose:
        verbose = False
    if None is border_exclude:
        border_exclude = True
    if None is size_exclude:
        size_exclude = True
    automatic_smoothing = True
    fill_holes = FH_THRESHOLDING
    limit_choice = LIMIT_NONE
    
    unclump_method = UN_SHAPE
    image = image_and_mask
    image_mask = np.ones(image_and_mask.shape[:2], bool)
    
    binary_image, global_threshold, sigma = threshold_image(image, image_mask, threshold_smoothing_scale)
    #
    # Fill background holes inside foreground objects
    #
    def size_fn(size, is_foreground=None):
        if is_foreground is None:
            is_foreground = False
        return size < size_range_max * size_range_max

    binary_image = centrosome.cpmorphology.fill_labeled_holes(binary_image, size_fn=size_fn)
    
#     plt.figure()
#     plt.title('filled holes')
#     plt.imshow(binary_image, cmap='gray')

    labeled_image, object_count = scind.label(binary_image, np.ones((3, 3), bool))
#     print('[identify_primary_objects] object_count: {}'.format(object_count))
#     plt.figure()
#     plt.title('labeled')
#     plt.imshow(labeled_image, cmap='Greys_r')
    
    labeled_image, object_count, maxima_suppression_size = separate_neighboring_objects(
        labeled_image, image, object_count, automatic_smoothing, size_range_min, smoothing_filter_size, fill_holes, unclump_method)
#     print('[identify_primary_objects] suppressed object_count: {}'.format(object_count))
    
#     unedited_labels = labeled_image.copy()

    # Filter out objects touching the border or mask
    if border_exclude:
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = filter_on_border(image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0

#     plt.figure()
#     plt.title('labeled border')
#     plt.imshow(labeled_image, cmap='Greys_r')

    # Filter out small and large objects
    if size_exclude:
#         size_excluded_labeled_image = labeled_image.copy()
        areas = scind.measurements.sum(np.ones(labeled_image.shape),
                                       labeled_image,
                                       np.array(range(0, object_count + 1), dtype=np.int32))
        
        median_areas = np.median(areas)
#         for i in range(3):
#             max_area = np.max(areas)
#             areas[areas == max_area] = 0
#         sorted_areas = np.array(sorted(areas))
        
#         print(sorted_areas.tolist())
#         dynamic_min_size = size_range_min
#         dynamic_max_size = size_range_min
#         if median_areas < 20:
#             median_areas = 20
            
        dynamic_min_size = np.sqrt((median_areas * 0.1 * 4) / np.pi)
        dynamic_max_size = np.sqrt((np.max(areas) * 2 * 4) / np.pi)
        # ignores too small median areas
#         if dynamic_min_size < 15:
#             dynamic_min_size = 15
#             dynamic_max_size = 20
        print('[identify_primary_objects] median_areas: {}, min: {}, max: {}'.format(median_areas, dynamic_min_size, dynamic_max_size))
#         print(dynamic_min_size, dynamic_max_size)
        
#         sorted_areas = savgol_filter(sorted_areas, 5, 1, mode='nearest')
#         max_slope = np.abs([x - z for x, z in zip(sorted_areas[:-1], sorted_areas[1:])])
#         print(max_slope.tolist())
#         min_idx = -1
#         for idx, a_slope in enumerate(max_slope):
#             if 0 == a_slope:
#                 continue
#             if min_idx == -1 and a_slope < 2:
#                 min_idx = idx
#                 break
#         max_idx = -1
#         for idx in range(len(max_slope) - 1, 0, -1):
#             a_slope = max_slope[idx]
#             if -1 == max_idx and a_slope < 2:
#                 max_idx = idx
#                 break
            
        
#         print('[identify_primary_objects] min: {}({}), max: {}({})'.format(areas[min_idx], min_idx, areas[max_idx], max_idx))
#         
#         dynamic_min_size = np.sqrt((sorted_areas[min_idx] * 4) / np.pi)
#         dynamic_max_size = np.sqrt((sorted_areas[max_idx] * 4) / np.pi)
#         plt.plot(sorted_areas)
#         plt.show()
        labeled_image, small_removed_labels = filter_on_size(labeled_image, object_count, dynamic_min_size, dynamic_max_size, size_exclude)
#         size_excluded_labeled_image[labeled_image > 0] = 0
#         if 0 == np.count_nonzero(processed_labeled_image > 0):
#             areas = scind.measurements.sum(np.ones(labeled_image.shape),
#                                                    labeled_image,
#                                                    np.array(range(0, object_count + 1), dtype=np.int32))
#             the_min_size = np.sqrt(np.quantile(areas, 0.10) * np.pi)
#             the_max_size = np.sqrt(np.quantile(areas, 0.90) * np.pi)
#             processed_labeled_image, small_removed_labels = filter_on_size(labeled_image, object_count, the_min_size, the_max_size, size_exclude)
#         labeled_image = processed_labeled_image 

#     plt.figure()
#     plt.title('labeled size')
#     plt.imshow(size_excluded_labeled_image, cmap='Greys_r')
    
    #
    # Fill holes again after watershed
    #
    if fill_holes != FH_NEVER:
        labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

    # Relabel the image
    labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)
    
#     plt.figure()
#     plt.title('relabeled')
#     plt.imshow(labeled_image, cmap='Greys_r')

#     if limit_choice == LIMIT_ERASE:
#         if object_count > maximum_object_count:
#             labeled_image = np.zeros(labeled_image.shape, int)
#             if border_exclude:
#                 border_excluded_labeled_image = np.zeros(labeled_image.shape, int)
#             if size_exclude:
#                 size_excluded_labeled_image = np.zeros(labeled_image.shape, int)
#             object_count = 0

    # Make an outline image
#     outline_image = centrosome.outline.outline(labeled_image)
#     if size_exclude:
#         outline_size_excluded_image = centrosome.outline.outline(size_excluded_labeled_image)
#     if border_exclude:
#         outline_border_excluded_image = centrosome.outline.outline(border_excluded_labeled_image)
#     statistics = []
    
#     statistics.append(["# of accepted objects", "%d" % object_count])
#     if object_count > 0:
#         areas = scind.sum(
#             np.ones(labeled_image.shape),
#             labeled_image, np.arange(1, object_count + 1)
#         )
#         areas.sort()
#         low_diameter = (math.sqrt(float(areas[int(object_count / 10)]) / np.pi) * 2)
#         median_diameter = (math.sqrt(float(areas[int(object_count / 2)]) / np.pi) * 2)
#         high_diameter = (math.sqrt(float(areas[int(object_count * 9 / 10)]) / np.pi) * 2)
#         statistics.append(["10th pctile diameter", "%.1f pixels" % low_diameter])
#         statistics.append(["Median diameter", "%.1f pixels" % median_diameter])
#         statistics.append(["90th pctile diameter", "%.1f pixels" % high_diameter])
#         object_area = np.sum(areas)
#         total_area = np.product(labeled_image.shape[:2])
#         statistics.append(["Area covered by objects", "%.1f %%" % (100.0 * float(object_area) / float(total_area))])
#         statistics.append(["Thresholding filter size", "%.1f" % sigma])
#         statistics.append(["Threshold", "%0.3g" % global_threshold])
#         if unclump_method != UN_NONE:
#             statistics.append(["Declumping smoothing filter size",
#                                "%.1f" % (calc_smoothing_filter_size(automatic_smoothing, size_range_min, smoothing_filter_size))])
#             statistics.append(["Maxima suppression size",
#                                "%.1f" % maxima_suppression_size])
#         print(statistics)
        #     print(np.count_nonzero(1==aligned_bright_mask))
    
# #     unique_labels = np.unique(labeled_image[labeled_image!=0])
# #     print(unique_labels)
#     if verbose:
#     #     cplabels = [
#     #                 dict(name="Pollen",
#     #                      labels=[labeled_image]),
#     #                 dict(name="Objects filtered out by size",
#     #                      labels=[size_excluded_labeled_image]),
#     #                 dict(name="Objects touching border",
#     #                      labels=[border_excluded_labeled_image])]
#     #     cplabels = get_corrected_cplabels(cplabels)
#     #     outline_image, olocal_cmap = get_labeld_color_map(outline_image)
#         decorated_labeled_image, local_cmap = get_labeld_color_map(labeled_image)
# 
#         f, axarr = plt.subplots(2, 2)
#         axarr[0, 0].imshow(image, cmap='Greys_r')
#         axarr[0, 0].set_title('Input')
#         axarr[0, 1].imshow(labeled_image, cmap=local_cmap)
#         axarr[0, 1].set_title('Pollen')
#         axarr[1, 0].imshow(decorated_labeled_image, cmap=local_cmap)
#         if size_exclude:
#             axarr[1, 0].imshow(size_excluded_labeled_image, cmap="Greys_r", alpha=0.7)
#         if border_exclude:
#             axarr[1, 0].imshow(border_excluded_labeled_image, cmap="Greys_r", alpha=0.7)
#         axarr[1, 0].set_title('outline')
#     #     axarr[1, 1].imshow(aligned_blue, cmap='gray')
#     #     axarr[1, 1].set_title('Aligned Blue')
#         plt.show()
    return (labeled_image, object_count)

# @timing
def identify_secondary_objects(image_and_mask, threshold_smoothing_scale, size_range_min, size_range_max, smoothing_filter_size, maximum_object_count, border_exclude = None, size_exclude = None, verbose=None):
    if None is verbose:
        verbose = False
    if None is border_exclude:
        border_exclude = True
    if None is size_exclude:
        size_exclude = True
    automatic_smoothing = True
    fill_holes = FH_THRESHOLDING
    limit_choice = LIMIT_NONE
    
    unclump_method = UN_SHAPE
    image = image_and_mask
    image_mask = np.ones(image_and_mask.shape[:2], bool)
    
    binary_image, global_threshold, sigma = threshold_image(image, image_mask, threshold_smoothing_scale)
    #
    # Fill background holes inside foreground objects
    #
    def size_fn(size, is_foreground=None):
        if is_foreground is None:
            is_foreground = False
        return size < size_range_max * size_range_max

    binary_image = centrosome.cpmorphology.fill_labeled_holes(binary_image, size_fn=size_fn)
    

    labeled_image, object_count = scind.label(binary_image, np.ones((3, 3), bool))
    
    labeled_image, object_count, maxima_suppression_size = separate_secondary_neighboring_objects(
        labeled_image, image, object_count, automatic_smoothing, size_range_min, smoothing_filter_size, fill_holes, unclump_method)
# #     print('[identify_primary_objects] suppressed object_count: {}'.format(object_count))
#     
# #     unedited_labels = labeled_image.copy()
# 
#     # Filter out objects touching the border or mask
#     if border_exclude:
#         border_excluded_labeled_image = labeled_image.copy()
#         labeled_image = filter_on_border(image, labeled_image)
#         border_excluded_labeled_image[labeled_image > 0] = 0
# 
# #     plt.figure()
# #     plt.title('labeled border')
# #     plt.imshow(labeled_image, cmap='Greys_r')
# 
#     # Filter out small and large objects
#     if size_exclude:
# #         size_excluded_labeled_image = labeled_image.copy()
#         areas = scind.measurements.sum(np.ones(labeled_image.shape),
#                                        labeled_image,
#                                        np.array(range(0, object_count + 1), dtype=np.int32))
#         
#         median_areas = np.median(areas)
# #         for i in range(3):
# #             max_area = np.max(areas)
# #             areas[areas == max_area] = 0
# #         sorted_areas = np.array(sorted(areas))
#         
# #         print(sorted_areas.tolist())
# #         dynamic_min_size = size_range_min
# #         dynamic_max_size = size_range_min
# #         if median_areas < 20:
# #             median_areas = 20
#             
#         dynamic_min_size = np.sqrt((median_areas * 0.1 * 4) / np.pi)
#         dynamic_max_size = np.sqrt((np.max(areas) * 2 * 4) / np.pi)
#         # ignores too small median areas
# #         if dynamic_min_size < 15:
# #             dynamic_min_size = 15
# #             dynamic_max_size = 20
#         print('[identify_primary_objects] median_areas: {}, min: {}, max: {}'.format(median_areas, dynamic_min_size, dynamic_max_size))
# #         print(dynamic_min_size, dynamic_max_size)
#         
# #         sorted_areas = savgol_filter(sorted_areas, 5, 1, mode='nearest')
# #         max_slope = np.abs([x - z for x, z in zip(sorted_areas[:-1], sorted_areas[1:])])
# #         print(max_slope.tolist())
# #         min_idx = -1
# #         for idx, a_slope in enumerate(max_slope):
# #             if 0 == a_slope:
# #                 continue
# #             if min_idx == -1 and a_slope < 2:
# #                 min_idx = idx
# #                 break
# #         max_idx = -1
# #         for idx in range(len(max_slope) - 1, 0, -1):
# #             a_slope = max_slope[idx]
# #             if -1 == max_idx and a_slope < 2:
# #                 max_idx = idx
# #                 break
#             
#         
# #         print('[identify_primary_objects] min: {}({}), max: {}({})'.format(areas[min_idx], min_idx, areas[max_idx], max_idx))
# #         
# #         dynamic_min_size = np.sqrt((sorted_areas[min_idx] * 4) / np.pi)
# #         dynamic_max_size = np.sqrt((sorted_areas[max_idx] * 4) / np.pi)
# #         plt.plot(sorted_areas)
# #         plt.show()
#         labeled_image, small_removed_labels = filter_on_size(labeled_image, object_count, dynamic_min_size, dynamic_max_size, size_exclude)
# #         size_excluded_labeled_image[labeled_image > 0] = 0
# #         if 0 == np.count_nonzero(processed_labeled_image > 0):
# #             areas = scind.measurements.sum(np.ones(labeled_image.shape),
# #                                                    labeled_image,
# #                                                    np.array(range(0, object_count + 1), dtype=np.int32))
# #             the_min_size = np.sqrt(np.quantile(areas, 0.10) * np.pi)
# #             the_max_size = np.sqrt(np.quantile(areas, 0.90) * np.pi)
# #             processed_labeled_image, small_removed_labels = filter_on_size(labeled_image, object_count, the_min_size, the_max_size, size_exclude)
# #         labeled_image = processed_labeled_image 
# 
# #     plt.figure()
# #     plt.title('labeled size')
# #     plt.imshow(size_excluded_labeled_image, cmap='Greys_r')
#     
#     #
#     # Fill holes again after watershed
#     #
#     if fill_holes != FH_NEVER:
#         labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)
# 
#     # Relabel the image
#     labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)
    
    return (labeled_image, object_count)

def threshold_image(image, mask, threshold_smoothing_scale, automatic=None):
    if None is automatic:
        automatic = False

    local_threshold, global_threshold = get_threshold_li(image, mask, automatic)
    binary_image, sigma = apply_threshold(image, local_threshold, threshold_smoothing_scale, automatic)

    return binary_image, global_threshold, sigma

def get_threshold_li(image, mask, automatic):
#     threshold = _global_threshold(image, mask, skimage.filters.threshold_li)
    threshold = _global_threshold(image, mask, skimage.filters.threshold_otsu)
    if automatic:
        return threshold, threshold

    threshold = _correct_global_threshold(threshold)

    return threshold, threshold

def _global_threshold(image, mask, threshold_fn):
    data = image

    if len(data[mask]) == 0:
        t_global = 0.0
    elif np.all(data[mask] == data[mask][0]):
        t_global = data[mask][0]
    else:
        t_global = threshold_fn(data[mask])

    return t_global

def _correct_global_threshold(threshold, threshold_range_min = None, threshold_range_max = None, threshold_correction_factor = None):
    if None is threshold_range_min:
        threshold_range_min = 0
    if None is threshold_range_max:
        threshold_range_max = 1.0
    if None is threshold_correction_factor:
        threshold_correction_factor = 1.0
    threshold *= threshold_correction_factor
#     print('[_correct_global_threshold] thr: {}'.format(threshold))
    return min(max(threshold, threshold_range_min), threshold_range_max)

def apply_threshold(image, threshold, threshold_smoothing_scale, automatic=False):
    data = image


    mask = np.ones(data.shape[:2], bool)

    if automatic:
        sigma = 1
    else:
        # Convert from a scale into a sigma. What I've done here
        # is to structure the Gaussian so that 1/2 of the smoothed
        # intensity is contributed from within the smoothing diameter
        # and 1/2 is contributed from outside.
        sigma = threshold_smoothing_scale / 0.6744 / 2.0

    blurred_image = centrosome.smooth.smooth_with_function_and_mask(
        data,
        lambda x: scind.gaussian_filter(x, sigma, mode="constant", cval=0),
        mask
    )

    return (blurred_image >= threshold) & mask, sigma

def separate_neighboring_objects(labeled_image, image, object_count, automatic_smoothing, size_range_min, smoothing_filter_size, fill_holes, unclump_method):
    """Separate objects based on local maxima or distance transform

    workspace - get the image from here

    labeled_image - image labeled by scipy.ndimage.label

    object_count  - # of objects in image

    returns revised labeled_image, object count, maxima_suppression_size,
    LoG threshold and filter diameter
    """
#     print('[separate_neighboring_objects] smoothing filter size: {}'.format(smoothing_filter_size))
#     image = labeled_image
    mask = np.ones(image.shape[:2], bool)
    # Speed up by using lower-resolution image to find local maxima?'
    low_res_maxima = True
    # automatically calculate the distance between intensity maxima to assist in declumping.
    automatic_suppression = True
    watershed_method = WA_SHAPE
    blurred_image = smooth_image(image, mask, automatic_smoothing, size_range_min, smoothing_filter_size)
#     
#     plt.figure()
#     plt.imshow(blurred_image)
#     plt.title('blurred')
#     plt.show()
    if size_range_min > 10 and (low_res_maxima):
        print('[separate_neighboring_objects] blurred size_range: {}, low_res_maxima: {}'.format(size_range_min, low_res_maxima))
        image_resize_factor = 10.0 / float(size_range_min)
        if automatic_suppression:
            maxima_suppression_size = 7
        else:
            maxima_suppression_size = (maxima_suppression_size * image_resize_factor + .5)
        reported_maxima_suppression_size = maxima_suppression_size / image_resize_factor
    else:
        image_resize_factor = 1.0
        if automatic_suppression:
            maxima_suppression_size = size_range_min / 1.5
        reported_maxima_suppression_size = maxima_suppression_size
    maxima_mask = centrosome.cpmorphology.strel_disk(max(1, maxima_suppression_size - .5))
    distance_transformed_image = None
    if unclump_method == UN_INTENSITY:
        # Remove dim maxima
        maxima_image = get_maxima(blurred_image,
                                       labeled_image,
                                       maxima_mask,
                                       image_resize_factor)
    elif unclump_method == UN_SHAPE:
        if fill_holes == FH_NEVER:
            # For shape, even if the user doesn't want to fill holes,
            # a point far away from the edge might be near a hole.
            # So we fill just for this part.
            foreground = centrosome.cpmorphology.fill_labeled_holes(labeled_image) > 0
        else:
            foreground = labeled_image > 0
        distance_transformed_image = \
            scind.distance_transform_edt(foreground)
        # randomize the distance slightly to get unique maxima
        np.random.seed(0)
        distance_transformed_image += \
            np.random.uniform(0, .001, distance_transformed_image.shape)
        maxima_image = get_maxima(distance_transformed_image,
                                       labeled_image,
                                       maxima_mask,
                                       image_resize_factor)
    else:
        raise ValueError("Unsupported local maxima method: %s" % unclump_method)
    
    # Create the image for watershed
    if watershed_method == WA_INTENSITY:
        # use the reverse of the image to get valleys at peaks
        watershed_image = 1 - image
    elif watershed_method == WA_SHAPE:
        if distance_transformed_image is None:
            distance_transformed_image = \
                scind.distance_transform_edt(labeled_image > 0)
        watershed_image = -distance_transformed_image
        watershed_image = watershed_image - np.min(watershed_image)
    elif watershed_method == WA_PROPAGATE:
        # No image used
        pass
    else:
        raise NotImplementedError("Watershed method %s is not implemented" % watershed_method.value)
    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    labeled_maxima, object_count = \
        scind.label(maxima_image, np.ones((3, 3), bool))
    if watershed_method == WA_PROPAGATE:
        watershed_boundaries, distance =  centrosome.propagate.propagate(np.zeros(labeled_maxima.shape),
                                           labeled_maxima,
                                           labeled_image != 0, 1.0)
    else:
        markers_dtype = (np.int16
                         if object_count < np.iinfo(np.int16).max
                         else np.int32)
        markers = np.zeros(watershed_image.shape, markers_dtype)
        markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
    
        #
        # Some labels have only one maker in them, some have multiple and
        # will be split up.
        #
    
        watershed_boundaries = skimage.morphology.watershed(
            connectivity=np.ones((3, 3), bool),
            image=watershed_image,
            markers=markers,
            mask=labeled_image != 0
        )
    
        watershed_boundaries = -watershed_boundaries
    
    return watershed_boundaries, object_count, reported_maxima_suppression_size

def separate_secondary_neighboring_objects(labeled_image, image, object_count, automatic_smoothing, size_range_min, smoothing_filter_size, fill_holes, unclump_method):
    """Separate objects based on local maxima or distance transform

    workspace - get the image from here

    labeled_image - image labeled by scipy.ndimage.label

    object_count  - # of objects in image

    returns revised labeled_image, object count, maxima_suppression_size,
    LoG threshold and filter diameter
    """
#     print('[separate_neighboring_objects] smoothing filter size: {}'.format(smoothing_filter_size))
#     image = labeled_image
    mask = np.ones(image.shape[:2], bool)
    # Speed up by using lower-resolution image to find local maxima?'
    low_res_maxima = False
    # automatically calculate the distance between intensity maxima to assist in declumping.
    automatic_suppression = True
    watershed_method = WA_SHAPE
    blurred_image = smooth_image(image, mask, automatic_smoothing, size_range_min, smoothing_filter_size)
#     
#     plt.figure()
#     plt.imshow(blurred_image)
#     plt.title('blurred')
#     plt.show()
    if size_range_min > 10 and (low_res_maxima):
        print('[separate_neighboring_objects] blurred size_range: {}, low_res_maxima: {}'.format(size_range_min, low_res_maxima))
        image_resize_factor = 10.0 / float(size_range_min)
        if automatic_suppression:
            maxima_suppression_size = 7
        else:
            maxima_suppression_size = (maxima_suppression_size * image_resize_factor + .5)
        reported_maxima_suppression_size = maxima_suppression_size / image_resize_factor
    else:
        image_resize_factor = 1.0
        if automatic_suppression:
            maxima_suppression_size = size_range_min / 1.5
        reported_maxima_suppression_size = maxima_suppression_size
    maxima_mask = centrosome.cpmorphology.strel_disk(max(1, maxima_suppression_size - .5))
    distance_transformed_image = None
    if unclump_method == UN_INTENSITY:
        # Remove dim maxima
        maxima_image = get_maxima(blurred_image,
                                       labeled_image,
                                       maxima_mask,
                                       image_resize_factor)
    elif unclump_method == UN_SHAPE:
        if fill_holes == FH_NEVER:
            # For shape, even if the user doesn't want to fill holes,
            # a point far away from the edge might be near a hole.
            # So we fill just for this part.
            foreground = centrosome.cpmorphology.fill_labeled_holes(labeled_image) > 0
        else:
            foreground = labeled_image > 0
        distance_transformed_image = \
            scind.distance_transform_edt(foreground)
        # randomize the distance slightly to get unique maxima
        np.random.seed(0)
        distance_transformed_image += \
            np.random.uniform(0, .001, distance_transformed_image.shape)
        maxima_image = get_maxima(distance_transformed_image,
                                       labeled_image,
                                       maxima_mask,
                                       image_resize_factor)
    else:
        raise ValueError("Unsupported local maxima method: %s" % unclump_method)
    
    # Create the image for watershed
    if watershed_method == WA_INTENSITY:
        # use the reverse of the image to get valleys at peaks
        watershed_image = 1 - image
    elif watershed_method == WA_SHAPE:
        if distance_transformed_image is None:
            distance_transformed_image = \
                scind.distance_transform_edt(labeled_image > 0)
        watershed_image = -distance_transformed_image
        watershed_image = watershed_image - np.min(watershed_image)
    elif watershed_method == WA_PROPAGATE:
        # No image used
        pass
    else:
        raise NotImplementedError("Watershed method %s is not implemented" % watershed_method.value)
    #
    # Create a marker array where the unlabeled image has a label of
    # -(nobjects+1)
    # and every local maximum has a unique label which will become
    # the object's label. The labels are negative because that
    # makes the watershed algorithm use FIFO for the pixels which
    # yields fair boundaries when markers compete for pixels.
    #
    labeled_maxima, object_count = \
        scind.label(maxima_image, np.ones((3, 3), bool))
    if watershed_method == WA_PROPAGATE:
        watershed_boundaries, distance =  centrosome.propagate.propagate(np.zeros(labeled_maxima.shape),
                                           labeled_maxima,
                                           labeled_image != 0, 1.0)
    else:
        markers_dtype = (np.int16
                         if object_count < np.iinfo(np.int16).max
                         else np.int32)
        markers = np.zeros(watershed_image.shape, markers_dtype)
        markers[labeled_maxima > 0] = -labeled_maxima[labeled_maxima > 0]
    
        #
        # Some labels have only one maker in them, some have multiple and
        # will be split up.
        #
    
        watershed_boundaries = skimage.morphology.watershed(
            connectivity=np.ones((3, 3), bool),
            image=watershed_image,
            markers=markers,
            mask=labeled_image != 0
        )
    
        watershed_boundaries = -watershed_boundaries
    
    return watershed_boundaries, object_count, reported_maxima_suppression_size

def smooth_image(image, mask, automatic_smoothing, size_range_min, smoothing_filter_size):
    """Apply the smoothing filter to the image"""

    filter_size = calc_smoothing_filter_size(automatic_smoothing, size_range_min, smoothing_filter_size)
#     print('[smooth_image] initial filter size: {}, mask.shape: {}, mask count(1==): {}'.format(filter_size, mask.shape, np.count_nonzero(1 == mask)))
    if filter_size == 0:
        return image
    sigma = filter_size / 2.35
    #
    # We not only want to smooth using a Gaussian, but we want to limit
    # the spread of the smoothing to 2 SD, partly to make things happen
    # locally, partly to make things run faster, partly to try to match
    # the Matlab behavior.
    #
    filter_size = max(int(float(filter_size) / 2.0), 1)
    f = (1 / np.sqrt(2.0 * np.pi) / sigma *
         np.exp(-0.5 * np.arange(-filter_size, filter_size + 1) ** 2 /
                   sigma ** 2))
    
#     print('[smooth_image] filter size: {}, sigma: {}, f: {}'.format(filter_size, sigma, f))

    def fgaussian(image):
        output = scind.convolve1d(image, f, axis=0, mode='constant')
        return scind.convolve1d(output, f, axis=1, mode='constant')

    #
    # Use the trick where you similarly convolve an array of ones to find
    # out the edge effects, then divide to correct the edge effects
    #
    edge_array = fgaussian(mask.astype(float))
    masked_image = image.copy()
    masked_image[~mask] = 0
    smoothed_image = fgaussian(masked_image)
    masked_image[mask] = smoothed_image[mask] / edge_array[mask]
    return masked_image

def calc_smoothing_filter_size(automatic_smoothing, size_range_min, smoothing_filter_size):
    """Return the size of the smoothing filter, calculating it if in automatic mode"""
    if automatic_smoothing:
        return 2.35 * size_range_min / 3.5
    else:
        return smoothing_filter_size

def get_maxima(image, labeled_image, maxima_mask, image_resize_factor):
    if image_resize_factor < 1.0:
        shape = np.array(image.shape) * image_resize_factor
        i_j = (np.mgrid[0:shape[0], 0:shape[1]].astype(float) /
               image_resize_factor)
        resized_image = scind.map_coordinates(image, i_j)
        resized_labels = scind.map_coordinates(
                labeled_image, i_j, order=0).astype(labeled_image.dtype)

    else:
        resized_image = image
        resized_labels = labeled_image
    #
    # find local maxima
    #
    if maxima_mask is not None:
        binary_maxima_image = centrosome.cpmorphology.is_local_maximum(resized_image,
                                                                       resized_labels,
                                                                       maxima_mask)
        binary_maxima_image[resized_image <= 0] = 0
    else:
        binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
    if image_resize_factor < 1.0:
        inverse_resize_factor = (float(image.shape[0]) /
                                 float(binary_maxima_image.shape[0]))
        i_j = (np.mgrid[0:image.shape[0],
               0:image.shape[1]].astype(float) /
               inverse_resize_factor)
        binary_maxima_image = scind.map_coordinates(
                binary_maxima_image.astype(float), i_j) > .5
        assert (binary_maxima_image.shape[0] == image.shape[0])
        assert (binary_maxima_image.shape[1] == image.shape[1])

    # Erode blobs of touching maxima to a single point

    shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
    return shrunk_image

def filter_on_border(image, labeled_image, exclude_border_objects = None):
    """Filter out objects touching the border

    In addition, if the image has a mask, filter out objects
    touching the border of the mask.
    """
    if None is exclude_border_objects:
        exclude_border_objects = True
    if not exclude_border_objects:
        return labeled_image
    border_labels = list(labeled_image[0, :])
    border_labels.extend(labeled_image[:, 0])
    border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
    border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
    border_labels = np.array(border_labels)
    #
    # the following histogram has a value > 0 for any object
    # with a border pixel
    #
    histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
                                         (border_labels,
                                          np.zeros(border_labels.shape))),
                                        shape=(np.max(labeled_image) + 1, 1)).todense()
    histogram = np.array(histogram).flatten()
    if any(histogram[1:] > 0):
        histogram_image = histogram[labeled_image]
        labeled_image[histogram_image > 0] = 0
#     elif image.has_mask:
#         # The assumption here is that, if nothing touches the border,
#         # the mask is a large, elliptical mask that tells you where the
#         # well is. That's the way the old Matlab code works and it's duplicated here
#         #
#         # The operation below gets the mask pixels that are on the border of the mask
#         # The erosion turns all pixels touching an edge to zero. The not of this
#         # is the border + formerly masked-out pixels.
#         mask_border = np.logical_not(scind.binary_erosion(image.mask))
#         mask_border = np.logical_and(mask_border, image.mask)
#         border_labels = labeled_image[mask_border]
#         border_labels = border_labels.flatten()
#         histogram = scipy.sparse.coo_matrix((np.ones(border_labels.shape),
#                                              (border_labels,
#                                               np.zeros(border_labels.shape))),
#                                             shape=(np.max(labeled_image) + 1, 1)).todense()
#         histogram = np.array(histogram).flatten()
#         if any(histogram[1:] > 0):
#             histogram_image = histogram[labeled_image]
#             labeled_image[histogram_image > 0] = 0
    return labeled_image

def filter_on_size(labeled_image, object_count, size_range_min, size_range_max, exclude_size = None):
    """ Filter the labeled image based on the size range

    labeled_image - pixel image labels
    object_count - # of objects in the labeled image
    returns the labeled image, and the labeled image with the
    small objects removed
    """
    if None is exclude_size:
        exclude_size = True
    
    if exclude_size and object_count > 0:
        areas = scind.measurements.sum(np.ones(labeled_image.shape),
                                               labeled_image,
                                               np.array(range(0, object_count + 1), dtype=np.int32))
        areas = np.array(areas, dtype=int)
        min_allowed_area = np.pi * (size_range_min * size_range_min) / 4
        max_allowed_area = np.pi * (size_range_max * size_range_max) / 4
#         print('[filter_on_size] min: {}, max: {}'.format(min_allowed_area, max_allowed_area))
        # area_image has the area of the object at every pixel within the object
        area_image = areas[labeled_image]
        labeled_image[area_image < min_allowed_area] = 0
        small_removed_labels = labeled_image.copy()
        labeled_image[area_image > max_allowed_area] = 0
    else:
        small_removed_labels = labeled_image.copy()
    return labeled_image, small_removed_labels

def get_labeld_color_map(image):
    # Mask the original labels
    label_image = np.ma.masked_where(image == 0, image)
    # Get the colormap from the user preferences
    colormap = matplotlib.cm.get_cmap('jet')
    # Initialize the colormap so we have access to the LUT
    colormap._init()
    # N is the number of "entries" in the LUT. `_lut` goes a little bit beyond that,
    # I think because there are "under" and "over" values. Regardless, we only one this
    # part of the LUT
    n = colormap.N
    # Get the LUT (only the part we care about)
    lut = colormap._lut[:n].copy()
    # Shuffle the colors so adjacently labeled objects are different colors
    np.random.shuffle(lut)
    # Set the LUT
    colormap._lut[:n] = lut
    # Make sure the background is black
    colormap.set_bad(color='black')
    return label_image, colormap

def do_align_without_mask(pixels1, pixels2):
    '''Align the second image with the first using mutual information

    returns the x,y offsets to add to image1's indexes to align it with
    image2

    The algorithm computes the mutual information content of the two
    images, offset by one in each direction (including diagonal) and
    then picks the direction in which there is the most mutual information.
    From there, it tries all offsets again and so on until it reaches
    a local maximum.
    '''
    #
    # TODO: Possibly use all 3 dimensions for color some day
    #
    if pixels1.ndim == 3:
        pixels1 = np.mean(pixels1, 2)
    if pixels2.ndim == 3:
        pixels2 = np.mean(pixels2, 2)

    def mutualinf_without_mask(x, y):
        return entropy(x) + entropy(y) - entropy2(x, y)

    maxshape = np.maximum(pixels1.shape, pixels2.shape)
    pixels1 = reshape_image(pixels1, maxshape)
    pixels2 = reshape_image(pixels2, maxshape)

    best = mutualinf_without_mask(pixels1, pixels2)
    i = 0
    j = 0
    while True:
        last_i = i
        last_j = j
        for new_i in range(last_i - 1, last_i + 2):
            for new_j in range(last_j - 1, last_j + 2):
                if new_i == 0 and new_j == 0:
                    continue
                p2, p1 = offset_slice(pixels2, pixels1, new_i, new_j)
                info = mutualinf_without_mask(p1, p2)
                if info > best:
                    best = info
                    i = new_i
                    j = new_j
#         print('[do_align_without_mask] i: {}, j: {}, best: {}'.format(i, j, best))
        if i == last_i and j == last_j:
            return j, i
        
def do_align(first_image, second_image):
    mask1 = np.ones(first_image.shape[:2], bool)
    mask2 = np.ones(second_image.shape[:2], bool)
    return align_mutual_information(first_image, second_image, mask1, mask2)

def align_mutual_information(pixels1, pixels2, mask1, mask2):
    '''Align the second image with the first using mutual information

    returns the x,y offsets to add to image1's indexes to align it with
    image2

    The algorithm computes the mutual information content of the two
    images, offset by one in each direction (including diagonal) and
    then picks the direction in which there is the most mutual information.
    From there, it tries all offsets again and so on until it reaches
    a local maximum.
    '''
    #
    # TODO: Possibly use all 3 dimensions for color some day
    #
    if pixels1.ndim == 3:
        pixels1 = np.mean(pixels1, 2)
    if pixels2.ndim == 3:
        pixels2 = np.mean(pixels2, 2)

    def mutualinf(x, y, maskx, masky):
        x = x[maskx & masky]
        y = y[maskx & masky]
        return entropy(x) + entropy(y) - entropy2(x, y)

    maxshape = np.maximum(pixels1.shape, pixels2.shape)
    pixels1 = reshape_image(pixels1, maxshape)
    pixels2 = reshape_image(pixels2, maxshape)
    mask1 = reshape_image(mask1, maxshape)
    mask2 = reshape_image(mask2, maxshape)

    best = mutualinf(pixels1, pixels2, mask1, mask2)
    i = 0
    j = 0
    while True:
        last_i = i
        last_j = j
        for new_i in range(last_i - 1, last_i + 2):
            for new_j in range(last_j - 1, last_j + 2):
                if new_i == 0 and new_j == 0:
                    continue
                p2, p1 = offset_slice(pixels2, pixels1, new_i, new_j)
                m2, m1 = offset_slice(mask2, mask1, new_i, new_j)
                info = mutualinf(p1, p2, m1, m2)
                if info > best:
                    best = info
                    i = new_i
                    j = new_j
        if i == last_i and j == last_j:
            return j, i

def entropy(x):
    '''The entropy of x as if x is a probability distribution'''
    histogram = scind.histogram(x.astype(float), np.min(x), np.max(x), 256)
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram != 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0

def entropy2(x, y):
    '''Joint entropy of paired samples X and Y'''
    #
    # Bin each image into 256 gray levels
    #
    x = (stretch(x) * 255).astype(int)
    y = (stretch(y) * 255).astype(int)
    #
    # create an image where each pixel with the same X & Y gets
    # the same value
    #
    xy = 256 * x + y
    xy = xy.flatten()
    sparse = scipy.sparse.coo_matrix((np.ones(xy.shape, dtype=np.int32),
                                      (xy, np.zeros(xy.shape, dtype=np.int32))))
    histogram = sparse.toarray()
    n = np.sum(histogram)
    if n > 0 and np.max(histogram) > 0:
        histogram = histogram[histogram > 0]
        return np.log2(n) - np.sum(histogram * np.log2(histogram)) / n
    else:
        return 0
    
def reshape_image(source, new_shape):
    '''Reshape an image to a larger shape, padding with zeros'''
    if tuple(source.shape) == tuple(new_shape):
        return source

    result = np.zeros(new_shape, source.dtype)
    result[:source.shape[0], :source.shape[1]] = source
    return result

def offset_slice(pixels1, pixels2, i, j):
    '''Return two sliced arrays where the first slice is offset by i,j
    relative to the second slice.

    '''
    if i < 0:
        height = min(pixels1.shape[0] + i, pixels2.shape[0])
        p1_imin = -i
        p2_imin = 0
    else:
        height = min(pixels1.shape[0], pixels2.shape[0] - i)
        p1_imin = 0
        p2_imin = i
    p1_imax = p1_imin + height
    p2_imax = p2_imin + height
    if j < 0:
        width = min(pixels1.shape[1] + j, pixels2.shape[1])
        p1_jmin = -j
        p2_jmin = 0
    else:
        width = min(pixels1.shape[1], pixels2.shape[1] - j)
        p1_jmin = 0
        p2_jmin = j
    p1_jmax = p1_jmin + width
    p2_jmax = p2_jmin + width

    p1 = pixels1[p1_imin:p1_imax, p1_jmin:p1_jmax]
    p2 = pixels2[p2_imin:p2_imax, p2_jmin:p2_jmax]
    return p1, p2

def adjust_offsets(offsets, shapes, crop_mode):
    '''Adjust the offsets and shapes for output

    workspace - workspace passed to "run"

    offsets - i,j offsets for each image

    shapes - shapes of the input images

    names - pairs of input / output names

    Based on the crop mode, adjust the offsets and shapes to optimize
    the cropping.
    '''
    offsets = np.array(offsets)
    shapes = np.array(shapes)
    if crop_mode == C_CROP:
        # modify the offsets so that all are negative
        max_offset = np.max(offsets, 0)
        offsets = offsets - max_offset[np.newaxis, :]
        #
        # Reduce each shape by the amount chopped off
        #
        shapes += offsets
        #
        # Pick the smallest in each of the dimensions and repeat for all
        #
        shape = np.min(shapes, 0)
        shapes = np.tile(shape, len(shapes))
        shapes.shape = offsets.shape
    elif crop_mode == C_PAD:
        #
        # modify the offsets so that they are all positive
        #
        min_offset = np.min(offsets, 0)
        offsets = offsets - min_offset[np.newaxis, :]
        #
        # Expand each shape by the top-left padding
        #
        shapes += offsets
        #
        # Pick the largest in each of the dimensions and repeat for all
        #
        shape = np.max(shapes, 0)
        shapes = np.tile(shape, len(shapes))
        shapes.shape = offsets.shape
    return offsets.tolist(), shapes.tolist()

def apply_alignment(input_image, off_x, off_y, shape):
    '''Apply an alignment to the input image to result in the output image

    workspace - image set's workspace passed to run

    input_image_name - name of the image to be aligned

    output_image_name - name of the resultant image

    off_x, off_y - offset of the resultant image relative to the original

    shape - shape of the resultant image
    '''
    
    pixel_data = input_image
    if pixel_data.ndim == 2:
        output_shape = (shape[0], shape[1], 1)
        planes = [pixel_data]
    else:
        output_shape = (shape[0], shape[1], pixel_data.shape[2])
        planes = [pixel_data[:, :, i] for i in range(pixel_data.shape[2])]
    input_image_mask = np.ones(input_image.shape[:2], bool)
    
    output_pixels = np.zeros(output_shape, pixel_data.dtype)
    for i, plane in enumerate(planes):
        #
        # Copy the input to the output
        #
        p1, p2 = offset_slice(plane, output_pixels[:, :, i], off_y, off_x)
        p2[:, :] = p1[:, :]
    if pixel_data.ndim == 2:
        output_pixels.shape = output_pixels.shape[:2]
    output_mask = np.zeros(shape, bool)
#     print('[apply_alignment] input mask count(==1): {}'.format(np.count_nonzero(1 == input_image_mask)))
    p1, p2 = offset_slice(input_image_mask, output_mask, off_y, off_x)
    p2[:, :] = p1[:, :]
#     if np.all(output_mask):
#         output_mask = None
    crop_mask = np.zeros(input_image.shape, bool)
    p1, p2 = offset_slice(crop_mask, output_pixels, off_y, off_x)
    p1[:, :] = True
    if np.all(crop_mask):
        crop_mask = None
#     print('[apply_alignment] output mask count(==1): {}, crop mask count(==1): {}'.format(np.count_nonzero(1 == output_mask), np.count_nonzero(1 == crop_mask)))
#     if None is output_mask and None is not crop_mask:
#         output_pixels = crop_image(output_pixels, crop_mask)
#         print('[apply_alignment] output: {}, crop: {}'.format(output_pixels.shape, crop_mask.shape))
#         return (output_pixels, crop_mask)
    
    return (output_pixels, output_mask)

def align_images_by_cv2_sift(im1, additional_images, gray_images, crop_mode):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.15
    results = [(im1, None)]
    for im2 in additional_images:
        # Convert images to grayscale
        im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
         
        # Detect ORB features and compute descriptors.
        orb = cv2.ORB_create(MAX_FEATURES)
        keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
         
        # Match features.
        matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
        matches = matcher.match(descriptors1, descriptors2, None)
         
        # Sort matches by score
        matches.sort(key=lambda x: x.distance, reverse=False)
        
        # Remove not so good matches
        numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
        matches = matches[:numGoodMatches]
        
        # Draw top matches
    #     imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    #     cv2.imwrite("matches.jpg", imMatches)
         
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        
        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt
        
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        
        # Use homography
        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, h, (width, height))
        results.append((im1Reg, None))
        print(im1Reg.shape)
        
    return results

def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    grad_x = cv2.Sobel(np.float32(im),cv2.CV_32F,1,0, ksize=31)
    grad_y = cv2.Sobel(np.float32(im),cv2.CV_32F,0,1, ksize=31)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad

def align_a_single_image(im1, im1_gray, im2, im2_gray, warp_mode=None):
    if None is warp_mode:
        warp_mode = cv2.MOTION_TRANSLATION
    # Find size of image1
    sz = im1.shape
     
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
     
    # Specify the number of iterations.
    number_of_iterations = 5000
     
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10
     
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
     
    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC (im1_gray, im2_gray, warp_matrix, warp_mode, criteria)
     
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        # Use warpPerspective for Homography 
        im2_aligned = cv2.warpPerspective (im2, warp_matrix, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned
            
def align_images_by_motion_translation(first_input_image, additional_images, gray_images, crop_mode):
    all_gray_images = []
    for an_img in gray_images:
        a_gray_image = an_img * 255
        a_gray_image[a_gray_image > 255] = 255
        a_gray_image[a_gray_image < 0] = 0
        a_gray_image = a_gray_image.astype(np.uint8)
        all_gray_images.append(a_gray_image)
        
    im1 =  first_input_image
    im1_gray = all_gray_images[0]
    # Define the motion model
#     warp_mode = cv2.MOTION_TRANSLATION
    
    results = [(im1, None)]
    for im2, im2_gray in zip(additional_images, all_gray_images[1:]):
        im2_aligned = align_a_single_image(im1, im1_gray, im2, im2_gray)
        results.append((im2_aligned, None))
    return results

def align_by_gradient(first_input_image, additional_images, gray_images, crop_mode):
    all_images = []
    for an_img in gray_images:
        a_gray_image = an_img * 255
        a_gray_image[a_gray_image > 255] = 255
        a_gray_image[a_gray_image < 0] = 0
        a_gray_image = a_gray_image.astype(np.uint8)
        all_images.append(a_gray_image)
    
    images = [first_input_image]
    images.extend(additional_images)
    first_image = gray_images[0]
    
    # Define motion model
#     warp_mode = cv2.MOTION_AFFINE
    warp_mode = cv2.MOTION_HOMOGRAPHY
 
    # Set the warp matrix to identity.
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
 
    # Set the stopping criteria for the algorithm.
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000,  1e-10)
    h, w = first_input_image.shape[0:2]
    results = [first_input_image, None]
    # Warp the blue and green channels to the red channel
    for additional_image in all_images[0]:
        (cc, warp_matrix) = cv2.findTransformECC (get_gradient(first_image), get_gradient(additional_image), warp_matrix, warp_mode, criteria)
        result_img = np.zeros((h, w), dtype=np.uint8)
        if warp_mode == cv2.MOTION_HOMOGRAPHY :
            # Use Perspective warp when the transformation is a Homography
            result_img = cv2.warpPerspective (additional_image, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else :
            # Use Affine warp when the transformation is not a Homography
            result_img = cv2.warpAffine(additional_image, warp_matrix, (w, h), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        results.append((result_img, None))
    return results

def align_images_by_sift(first_input_image, additional_images, gray_images, crop_mode):
    all_images = []
    for an_img in gray_images:
        a_gray_image = an_img * 255
        a_gray_image[a_gray_image > 255] = 255
        a_gray_image[a_gray_image < 0] = 0
        a_gray_image = a_gray_image.astype(np.uint8)
        all_images.append(a_gray_image)
    
    images = [first_input_image]
    images.extend(additional_images)
    first_image = gray_images[0]
    device_type = "GPU"
    sift_ocl = sift.SiftPlan(template=first_image, devicetype=device_type)
    key_points = sift_ocl(first_image)
    offsets = [(0, 0)]
    
    for an_img in gray_images[1:]:
        shifted_points = sift_ocl(an_img)
        mp = sift.MatchPlan()
        match = mp(key_points, shifted_points)
        print('[align_images_by_sift] Number of Keypoints with for image 1 : {}, For image 2 : {}, Matching keypoints: {}'
              .format(key_points.size, shifted_points.size, match.shape[0]))
        print('[align_images_by_sift] Measured offsets dx: {:.3f}, dy: {:.3f}'.format(np.median(match[:,1].x-match[:,0].x), np.median(match[:,1].y-match[:,0].y)))
        an_offset = [int(np.median(match[:,1].x-match[:,0].x)), int(np.median(match[:,1].y-match[:,0].y))]
        if abs(an_offset[0]) > 10:
            an_offset[0] = 0
        if abs(an_offset[1]) > 10:
            an_offset[1] = 0
        offsets.append(an_offset)
    print('[align_images_by_sift] offsets: {}'.format(offsets))
    shapes = [x.shape[:2] for x in images]
    offsets, shapes = adjust_offsets(offsets, shapes, crop_mode)
    results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(apply_alignment)(an_image, x, y, shape) for an_image, (y, x), shape in zip(images, offsets, shapes))
    return results

def align_images(first_input_image, additional_images, gray_images, crop_mode):
    first_image = gray_images[0]
    images = [first_input_image]
    images.extend(additional_images)
    
    additional_offsets = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(do_align_without_mask)(first_image, an_img) for an_img in gray_images[1:])
    offsets = [(0, 0)]
    offsets.extend(additional_offsets)
#     off_x, off_y = do_align(images[0], images[1])
#     offsets = [(0, 0), (off_y, off_x)]
#     for additional in additional_images[1:]:
#         if align_choice == A_SIMILARLY:
#             a_off_x, a_off_y = off_x, off_y
#         else:
#             a_off_x, a_off_y = do_align(first_input_image, additional)
#         offsets.append((a_off_y, a_off_x))
    print('[align_images] offsets: {}'.format(offsets))
    shapes = [x.shape[:2] for x in images]
    offsets, shapes = adjust_offsets(offsets, shapes, crop_mode)
    results = Parallel(n_jobs=-1, backend="multiprocessing")(delayed(apply_alignment)(an_image, x, y, shape) for an_image, (y, x), shape in zip(images, offsets, shapes))
#     results = []
#     for idx, (an_image, (y, x), shape) in enumerate(zip(images, offsets, shapes)):
# #         print('[align_images] idx: {}, x: {}, y: {}'.format(idx, x, y))
#         result = apply_alignment(an_image, x, y, shape)
#         results.append(result)
    return results

def get_shrinken_contour(I, J, n_iter=None, scale_factor=None, max_shape=None):
    if None is n_iter:
        n_iter = 1
    if None is scale_factor:
        scale_factor = 0.95
    if None is max_shape:
        max_shape = (10000, 10000)
    copied_I = I.astype(np.float64)
    copied_J = J.astype(np.float64)
    median_I = np.median(copied_I)
    median_J = np.median(copied_J)
    
#     print('[get_shrinken_contour] I: {}, J: {}'.format(median_I, median_J))

#     find the center of the contour (moments() or getBoundingRect())
#     subtract it from each point in the contour
#     multiply contour points x,y by a scale factor
#     add the center again to each point
    for _ in range(n_iter):
        copied_I -= median_I
        copied_J -= median_J
        copied_I *= scale_factor
        copied_J *= scale_factor
        copied_I += median_I
        copied_J += median_J
#         copied_I[copied_I < median_I] += 1
#         copied_I[copied_I > median_I] -= 1
#         copied_J[copied_J < median_J] += 1
#         copied_J[copied_J > median_J] -= 1
#         if 0 == copied_I.shape[0]:
#             return I, J
    copied_I = copied_I.astype(np.int64)
    copied_J = copied_J.astype(np.int64)
    
#     if max_shape != (10000, 10000):
#         print(max_shape[0], max_shape[1])
# #         print(copied_I)
#         print('[get_shrinken_contour] before: {}'.format(copied_J))
    copied_I[copied_I >= max_shape[0]] = max_shape[0] - 1
    copied_J[copied_J >= max_shape[1]] = max_shape[1] - 1
    
#     if max_shape != (10000, 10000):
# #         print(copied_I)
#         print('[get_shrinken_contour] after:  {}'.format(copied_J))
    
    shrinken_contours = np.stack((copied_I, copied_J), axis=-1)
    if 0 == shrinken_contours.shape[0]:
        return I, J
    unique_contours = np.unique(shrinken_contours, axis=0)
#     print(np.stack((I, J), axis=-1).tolist())
#     print(shrinken_contours.tolist())
#     print(unique_contours.tolist())
    s_I = unique_contours[:,0]
    s_J = unique_contours[:,1]
    return (s_I, s_J)