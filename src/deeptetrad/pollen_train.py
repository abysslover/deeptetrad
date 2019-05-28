"""
Mask R-CNN
Train on the toy Pollen dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 pollen.py train --dataset=/path/to/pollen/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 pollen.py train --dataset=/path/to/pollen/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 pollen.py train --dataset=/path/to/pollen/dataset --weights=imagenet

    # Apply color splash to an image
    python3 pollen.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 pollen.py splash --weights=last --video=<URL or path to file>
"""

import sys
import json
import numpy as np
import skimage.draw
import skimage.io
from imgaug import augmenters as iaa
import imgaug as ia

# from sklearn.neighbors import KDTree
from deeptetrad.utils.file_utils import *
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Root directory of the project
ROOT_DIR = './pollen'

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils
import mrcnn.model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

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

class PollenConfig(Config):
    """Configuration for training on the toy  dataset.
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
    STEPS_PER_EPOCH = 50

    # Skip detections with < 90% confidence
#     DETECTION_MIN_CONFIDENCE = 0.9
    DETECTION_MIN_CONFIDENCE = 0
    
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"
#     BACKBONE = "resnet101"

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
#     TRAIN_ROIS_PER_IMAGE = 128
    TRAIN_ROIS_PER_IMAGE = 512
    
    # Image mean (RGB)
    MEAN_PIXEL = np.array([35.694053141276044, 35.332364298502604, 24.645670166015623])
    
    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
#     USE_MINI_MASK = True
#     MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1000
      
#     IMAGE_PADDING = 1

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
#     MASK_SHAPE = [10, 10]
    
    DETECTION_MAX_INSTANCES = 1000


############################################################
#  Dataset
############################################################

class PollenDataset(mrcnn.utils.Dataset):

    def load_pollen(self, dataset_dir, subset):
        """Load a subset of the Pollen dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("pollen", 1, "pollen")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
#         annotations = list(annotations.values())  # don't need the dict keys
        annotations = annotations.values()  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        print('[PollenDataset.load_pollen] # annotations: {}'.format(len(annotations)))
        image_info_dict_path = '{}/image_info_dict.{}.pickle'.format(dataset_dir, subset)
        print('[PollenDataset.load_pollen] {}'.format(image_info_dict_path))
        if not os.path.exists(image_info_dict_path):
            # Add images
            for a in annotations:
    #             print(a['regions'])
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above)
    #             polygons = [r['shape_attributes'] for r in a['regions'].values()]
                polygons = [r['shape_attributes'] for r in a['regions']]
    
                # load_mask() needs the image size to convert polygons to masks.
                # Unfortunately, VIA doesn't include it in JSON, so we must read
                # the image. This is only managable since the dataset is tiny.
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                self.add_image(
                    "pollen",
                    image_id=a['filename'],  # use file name as a unique image id
                    path=image_path,
                    width=width, height=height,
                    polygons=polygons)
                print(image.shape)
                image_mean_r = np.mean(image[:,:,0])
                image_mean_g = np.mean(image[:,:,1])
                image_mean_b = np.mean(image[:,:,2])
                print('[load_pollen] images mean: R: {}, G: {}, B: {}'.format(image_mean_r, image_mean_g, image_mean_b))
            self.dump_data_to_pickle(image_info_dict_path, self.image_info)
        self.image_info = self.load_data_from_pickle(image_info_dict_path)

    def get_pickle_path(self, a_path, prefix):
        parent_path, filename = os.path.split(a_path)
        output_path = os.path.join(parent_path, 'test')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_filename = filename.replace('.jpg', '.{}.pickle'.format(prefix))
        output_path = '{}/{}'.format(output_path, output_filename)
        return output_path

    def dump_data_to_pickle(self, a_path, a_data):
        with open(a_path, 'wb') as out_data:
            pickle.dump(a_data, out_data)
            
    def load_data_from_pickle(self, a_path):
        with open(a_path, 'rb') as in_data:
            a_data = pickle.load(in_data)
        return a_data
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a pollen dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pollen":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
#         mask = np.zeros([info["height"], info["width"], 1],
#                     dtype=np.uint8)
#         print('[PollenDataset.load_mask] {}, # polygon: {}'.format(info["path"], len(info["polygons"])))
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "pollen":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def disp_training_data():
    class_names = ['BG', 'pollen']
    dataset_train = PollenDataset()
    dataset_train.load_pollen(args.dataset, "train")
    dataset_val = PollenDataset()
    dataset_val.load_pollen(args.dataset, "val")
    n_train_masks = 0
    for idx, an_image_dict in enumerate(dataset_train.image_info):
        n_train_masks = n_train_masks + len(an_image_dict['polygons'])
    
    n_val_masks = 0
    for idx, an_image_dict in enumerate(dataset_val.image_info):
        n_val_masks = n_val_masks + len(an_image_dict['polygons'])
    
    print('[disp_training_data] # train: {}, # val: {}'.format(n_train_masks, n_val_masks))
#     for idx, an_image_dict in enumerate(dataset_train.image_info):
#         image = dataset_train.load_image(idx)
#         mask, the_ones = dataset_train.load_mask(idx)
# #         bbox = utils.extract_bboxes(mask)
#         bbox = None
#         class_ids = np.ones(mask.shape[-1], dtype=np.uint32)
#         visualize.display_instances(image, bbox, mask, class_ids, class_names, show_bbox=False, show_captions=False)

def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = PollenDataset()
    dataset_train.load_pollen(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = PollenDataset()
    dataset_val.load_pollen(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Affine(
            scale={"x": (0.5, 1.5), "y": (0.5, 1.5)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
            rotate=(-180, 180), # rotate by -45 to +45 degrees
            shear=(-16, 16), # shear by -16 to +16 degrees
            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
        iaa.PiecewiseAffine(scale=(0.01, 0.05))
    ])

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                augmentation=augmentation,
                layers='all')
    
############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect pollens.')
    parser.add_argument("--command", required=False,
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/pollen/dataset/",
                        help='Directory of the Pollen dataset')
    parser.add_argument('--weights', required=False,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()
    
#     args.command = "train"
#     args.command = "splash"
#     args.command = 'disp_training'
    args.dataset = './pollen'
    args.weights = 'coco'
#     args.weights = 'last'
    args.logs = './pollen/logs/'
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command != 'disp_training':
        if args.command == "train":
            config = PollenConfig()
        else:
            class InferenceConfig(PollenConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                IMAGE_RESIZE_MODE = "pad64"
                GPU_COUNT = 1
                IMAGES_PER_GPU = 7
                DETECTION_MIN_CONFIDENCE = 0
                DETECTION_MAX_INSTANCES = 500
            config = InferenceConfig()
        config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    elif args.command == 'disp_training':
        pass
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)
    if args.command != 'disp_training':
        # Select weights file to load
        if args.weights.lower() == "coco":
            weights_path = COCO_WEIGHTS_PATH
            # Download weights file
            if not os.path.exists(weights_path):
                mrcnn.utils.download_trained_weights(weights_path)
        elif args.weights.lower() == "last":
            # Find last trained weights
            weights_path = model.find_last()
        elif args.weights.lower() == "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        else:
            weights_path = args.weights

        # Load weights
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "disp_training":
        disp_training_data()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))