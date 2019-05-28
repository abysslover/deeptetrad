'''
Created on Mar 6, 2019

@author: vincent
'''

import cv2
import numpy as np


import matplotlib.cm

C_CROP = "Crop to aligned region"
C_PAD = "Pad images"

# maximum_object_count = 500
# smoothing_filter_size = 10
# suppress_diameter = 10
import time

def pad_image(an_img, shape):
    # shape = (height, width)
    out_img = an_img.copy()
    height_top = (shape[0] - out_img.shape[0]) >> 1
    height_bottom = shape[0] - out_img.shape[0] - height_top
    width_left = (shape[1] - out_img.shape[1]) >> 1
    width_right = shape[1] - out_img.shape[1] - width_left
    return cv2.copyMakeBorder(out_img, height_top, height_bottom, width_left, width_right, cv2.BORDER_CONSTANT)

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