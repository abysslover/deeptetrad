"""
Mask R-CNN
Train on the toy Tetrad dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 tetrad.py train --dataset=/path/to/tetrad/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 tetrad.py train --dataset=/path/to/tetrad/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 tetrad.py train --dataset=/path/to/tetrad/dataset --weights=imagenet

    # Apply color splash to an image
    python3 tetrad.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 tetrad.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw
import skimage.io
import matplotlib
matplotlib.use('Agg')
from imgaug import augmenters as iaa
import pickle
import tensorflow
import keras.backend as k_backend
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Root directory of the project
ROOT_DIR = './tetrad'

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


class TetradConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tetrad"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
#     NUM_CLASSES = 1 + 4  # Background + monad + dyad + triad + tetrad
    NUM_CLASSES = 1 + 1  # Background + non_tetrad + tetrad

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
    
#     RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    BACKBONE_STRIDES = [2, 4, 8, 16, 32]
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.8
    
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
#     USE_MINI_MASK = False
#     MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1000
      
#     IMAGE_PADDING = 1

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
#     MASK_SHAPE = [25, 25]
    

############################################################
#  Dataset
############################################################

class TetradDataset(mrcnn.utils.Dataset):

    def load_tetrad(self, dataset_dir, subset):
        """Load a subset of the Tetrad dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
#         self.add_class("pollen", 1, "monad")
#         self.add_class("pollen", 2, "dyad")
#         self.add_class("pollen", 3, "triad")
#         self.add_class("pollen", 4, "tetrad")
#         self.add_class("pollen", 1, "non_tetrad")
#         self.add_class("pollen", 2, "tetrad")
        self.add_class("pollen", 1, "tetrad")

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
        print('[TetradDataset.load_tetrad] # annotations: {}'.format(len(annotations)))
        image_info_dict_path = '{}/image_info_dict.{}.pickle'.format(dataset_dir, subset)
        print('[TetradDataset.load_tetrad] {}'.format(image_info_dict_path))
        
        self.image_mask_dict = {}
        if not os.path.exists(image_info_dict_path):
            # Add images
            for a in annotations:
    #             print(a['regions'])
                # Get the x, y coordinaets of points of the polygons that make up
                # the outline of each object instance. There are stores in the
                # shape_attributes (see json format above)
    #             polygons = [r['shape_attributes'] for r in a['regions'].values()]
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                polygons = [r['shape_attributes'] for r in a['regions']]
                region_attributes = [r['region_attributes'] for r in a['regions']]
                local_annotations = {}
                for a_poly, a_region_attr in zip(polygons, region_attributes):
                    the_pollen_type = a_region_attr['pollen_type']
                    #TODO: comment out two following statements if you want to add all original classes
                    if 'tetrad' != the_pollen_type:
                        the_pollen_type = 'tetrad'
                    if the_pollen_type not in local_annotations:
                        local_annotations[the_pollen_type] = []
                    local_annotations[the_pollen_type].append(a_poly)
                image_path = os.path.join(dataset_dir, a['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                an_img_id = '{}'.format(a['filename'])
                self.add_image(
                        "pollen",
                        image_id=an_img_id,  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        annotations=local_annotations)
    #             for a_class_info_dict in self.class_info:
    #                 if 0 == a_class_info_dict['id']:
    #                     continue
    #                 pollen_type = a_class_info_dict['name']
    #                 local_polygons = annotations[pollen_type]
    #                 an_img_id = '{}_{}'.format(a['filename'], pollen_type)
    # #                 print('[load_tetrad] pollen_type: {}, image_id: {}, path: {}'.format(pollen_type, an_img_id, image_path))
    #                 self.add_image(
    #                     pollen_type,
    #                     image_id=an_img_id,  # use file name as a unique image id
    #                     path=image_path,
    #                     width=width, height=height,
    #                     polygons=local_polygons)
                instance_masks = []
                class_ids = []
                
                for a_class_info_dict in self.class_info:
                    cur_class_id = a_class_info_dict['id']
                    if 0 == a_class_info_dict['id']:
                        continue
                    
                    pollen_type = a_class_info_dict['name']
                    if 'tetrad' != pollen_type:
                        pollen_type = 'tetrad'
                    local_polygons = local_annotations[pollen_type]
                
                    for p in local_polygons:
                        mask = np.zeros([height, width], dtype=np.uint8)
                    # Get indexes of pixels inside the polygon and set them to 1
                        rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                        mask[rr, cc] = 1
            #                 print('[tetrad_train.load_mask] class id: {}, # polygons: {}'.format(cur_class_id, len(local_polygons)))
                        class_ids.append(cur_class_id)
                        instance_masks.append(mask)
                if class_ids:
                    mask = np.stack(instance_masks, axis=2).astype(np.bool)
                    class_ids = np.array(class_ids, dtype=np.int32)
                    print('[load_tetrad] mask_shape: {}, class_id shape: {}'.format(mask.shape, class_ids.shape))
                else:
                    mask = np.zeros([height, width], dtype=np.uint8)
                    class_ids = []
                self.image_mask_dict[an_img_id] = (mask, class_ids)
                
                image_mean_r = np.mean(image[:,:,0])
                image_mean_g = np.mean(image[:,:,1])
                image_mean_b = np.mean(image[:,:,2])
                print('[load_tetrad] images mean: R: {}, G: {}, B: {}'.format(image_mean_r, image_mean_g, image_mean_b))
                
            self.dump_data_to_pickle(image_info_dict_path, (self.image_info, self.image_mask_dict))
            
#         for a in annotations:
# #             print(a['regions'])
#             # Get the x, y coordinaets of points of the polygons that make up
#             # the outline of each object instance. There are stores in the
#             # shape_attributes (see json format above)
# #             polygons = [r['shape_attributes'] for r in a['regions'].values()]
#             polygons = [r['shape_attributes'] for r in a['regions']]
#             region_attributes = [r['region_attributes'] for r in a['regions']]
#             print(len(polygons))
#             print(len(region_attributes))
#             
# 
#             # load_mask() needs the image size to convert polygons to masks.
#             # Unfortunately, VIA doesn't include it in JSON, so we must read
#             # the image. This is only managable since the dataset is tiny.
#             image_path = os.path.join(dataset_dir, a['filename'])
#             image = skimage.io.imread(image_path)
#             height, width = image.shape[:2]
#             self.add_image(
#                 "tetrad",
#                 image_id=a['filename'],  # use file name as a unique image id
#                 path=image_path,
#                 width=width, height=height,
#                 polygons=polygons)
# #                 print(image.shape)
#             image_mean_r = np.mean(image[:,:,0])
#             image_mean_g = np.mean(image[:,:,1])
#             image_mean_b = np.mean(image[:,:,2])
#             print('[load_tetrad] images mean: R: {}, G: {}, B: {}'.format(image_mean_r, image_mean_g, image_mean_b))
#             self.dump_data_to_pickle(image_info_dict_path, self.image_info)
        (self.image_info, self.image_mask_dict) = self.load_data_from_pickle(image_info_dict_path)
        for an_image_info_dict in self.image_info:
            print('[load_tetrad] id: {}, name: {}'.format(an_image_info_dict['id'], an_image_info_dict['source']))

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
            pickle.dump(a_data, out_data, protocol=4)
            
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
        # If not a tetrad dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "pollen":
#         if image_info["source"] != "monad" and image_info["source"] != "dyad" and image_info["source"] != "triad" and \
#             image_info["source"] != "tetrad":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        
        return self.image_mask_dict[info["id"]]
#         annotations = info["annotations"]
#         instance_masks = []
#         class_ids = []
        
#         for a_class_info_dict in self.class_info:
#             cur_class_id = a_class_info_dict['id']
#             if 0 == a_class_info_dict['id']:
#                 continue
#             
#             pollen_type = a_class_info_dict['name']
#             local_polygons = annotations[pollen_type]
#         
#             for p in local_polygons:
#                 mask = np.zeros([info["height"], info["width"]], dtype=np.uint8)
#             # Get indexes of pixels inside the polygon and set them to 1
#                 rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
#                 mask[rr, cc] = 1
# #                 print('[tetrad_train.load_mask] class id: {}, # polygons: {}'.format(cur_class_id, len(local_polygons)))
#                 class_ids.append(cur_class_id)
#                 instance_masks.append(mask)
# #         mask = np.zeros([info["height"], info["width"], 1],
# #                     dtype=np.uint8)
# #         print('[TetradDataset.load_mask] {}, # polygon: {}'.format(info["path"], len(info["polygons"])))
# #         for i, p in enumerate(info["polygons"]):
# #             # Get indexes of pixels inside the polygon and set them to 1
# #             rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
# #             mask[rr, cc, i] = 1
# 
#         # Return mask, and array of class IDs of each instance. Since we have
#         # one class ID only, we return an array of 1s
# #         return mask, np.ones([mask.shape[-1]], dtype=np.int32)
#     # Pack instance masks into an array
#         if class_ids:
#             mask = np.stack(instance_masks, axis=2).astype(np.bool)
#             class_ids = np.array(class_ids, dtype=np.int32)
#             print('[load_mask] mask_shape: {}, class_id shape: {}'.format(mask.shape, class_ids.shape))
#             return mask, class_ids
#         else:
#             # Call super class to return an empty mask
#             return super(TetradDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
#         if info["source"] == "monad" or info["source"] == "dyad" or info["source"] == "triad" or info["source"] == "tetrad":
        if info["source"] == "pollen":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
def disp_training_data():
    class_names = ['BG', 'non-tetrad', 'tetrad']
    dataset_train = TetradDataset()
    dataset_train.load_tetrad(args.dataset, "train")
    dataset_val = TetradDataset()
    dataset_val.load_tetrad(args.dataset, "val")
    n_train_masks = 0
    for idx, an_image_dict in enumerate(dataset_train.image_info):
        for annotation_key, annotation_arr in an_image_dict['annotations'].items():
            n_train_masks = n_train_masks + len(annotation_arr)
    
    n_val_masks = 0
    for idx, an_image_dict in enumerate(dataset_val.image_info):
        for annotation_key, annotation_arr in an_image_dict['annotations'].items():
            n_val_masks = n_val_masks + len(annotation_arr)
    
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
    dataset_train = TetradDataset()
    dataset_train.load_tetrad(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TetradDataset()
    dataset_val.load_tetrad(args.dataset, "val")
    dataset_val.prepare()

    augmentation_first = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(scale={"x": (0.3, 2), "y": (0.3, 2)}),
        iaa.OneOf([iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Affine(shear=(-5, 5)), # shear by -16 to +16 degrees)
        iaa.Multiply((0.8, 1.5)),
        iaa.PiecewiseAffine(scale=(0.01, 0.05))
    ])
    
    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                augmentation=augmentation_first,
                layers='all')
    

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect tetrads.')
    parser.add_argument("--command", required=False,
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/tetrad/dataset/",
                        help='Directory of the Tetrad dataset')
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
#     args.command = "load"
    args.command = "train"
#     args.command = "splash"
#     args.command = 'disp_training'
    args.dataset = './tetrad'
    args.weights = 'coco'
#     args.weights = 'last'
    args.logs = './tetrad/logs/'
    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    if args.command != 'disp_training':
        # Configurations
        if args.command == "train":
            config = TetradConfig()
        else:
            class TetradInferenceConfig(TetradConfig):
                # Set batch size to 1 since we'll be running inference on
                # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
                IMAGE_RESIZE_MODE = "pad64"
                GPU_COUNT = 1
                IMAGES_PER_GPU = 21
                DETECTION_MIN_CONFIDENCE = 0
                DETECTION_MAX_INSTANCES = 200
    #             DETECTION_MAX_INSTANCES = DETECTION_MAX_INSTANCES * IMAGES_PER_GPU
    
            config = TetradInferenceConfig()
        config.display()
    if args.command != 'disp_training':
        tf_config = tensorflow.ConfigProto()
        tf_config.inter_op_parallelism_threads = 1
        k_backend.set_session(tensorflow.Session(config=tf_config))
        # Create model
        if args.command == "train":
            model = modellib.MaskRCNN(mode="training", config=config,
                                      model_dir=args.logs)
        else:
            model = modellib.MaskRCNN(mode="inference", config=config,
                                      model_dir=args.logs)

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
    elif args.command == 'disp_training':
        disp_training_data()
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))