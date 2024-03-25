import random
import numpy as np
import math
import cv2
from os.path import exists
from PIL import Image
import torch

from torch.utils.data import DataLoader

from vision.utils.misc import get_annotations_from_string


class FDDBDataset:

    def __init__(self, data_path, split, transform=None, target_transform=None, debug_mode=False):
        print('Setting up {} dataset:'.format(split))

        self.ann_path = data_path + "/" + 'FDDB-folds/' + split + ".txt"
        self.img_path = data_path + "/" + 'originalPics'

        self.transform = transform
        self.target_transform = target_transform
        self.debug_mode = debug_mode

        if not exists(self.ann_path):
            print(self.ann_path)

            if split.find('mini') != -1:
                self.prepare_dataset(data_path, mini=True)
            else:
                self.prepare_dataset(data_path)

        self.path_names = []
        self.objects = []

        with open(self.ann_path, 'r') as f:
            lines = f.readlines()

            j = 0

            while j < len(lines):
                path, nr_objs, str_objs = lines[j], int(lines[j+1]), lines[j+2]

                path = path.replace('\n', '')

                objects = get_annotations_from_string(str_objs, nr_objs)
                new_objects = self.convert_annotations_to_bbox(objects)

                self.path_names.append(path)
                self.objects.append(new_objects)

                j = j + 3

        self.class_names = ('BACKGROUND',
                            'human')
        print('Loaded {} samples for {} split'.format(
            len(self.path_names), split))

    def prepare_dataset(self, data_path, mini=False, split={'train': .7, 'val': .2, 'test': .1}):
        """
        Defines the train/val/test split and creates corresponding annotations files

        The dataset has 10 annotations file with the following format:
            path_name
            nr_objects
            object_1_annotation
            object_2_annotation
            ...
            object_k_annotation
        """
        ann_folder_path = data_path + '/FDDB-folds'

        paths = []
        annotations = []
        # Loop through all 10 annotations files
        for i in range(1, 11):
            ann_path = ann_folder_path + \
                '/FDDB-fold-{:02d}-ellipseList.txt'.format(i)

            # Open annotation file
            with open(ann_path, mode='r') as ann_file:
                lines = ann_file.readlines()

                j = 0

                while j < len(lines):
                    # Append current sample
                    path, nr_objs = lines[j], int(lines[j+1])

                    objs = [obj.replace('\n', '')
                            for obj in lines[j + 2: j + nr_objs + 2]]

                    paths.append(path)
                    annotations.append(objs)

                    # Go to next sample
                    j = j + nr_objs + 2

        # Compute the dimensions of train, val, test based on split percentage
        size = len(paths)

        if mini:
            size = 128

        train_size, val_size = int(
            size * split['train']), int(size * split['val'])

        # Make sure samples aren't lost by conversion to int
        test_size = size - train_size - val_size

        print('The dataset size {} splitted in: '.format(size))
        print('train: ', train_size)
        print('val: ', val_size)
        print('test: ', test_size)

        # Divide into the train, val, test
        indices = [x for x in range(0, len(paths))]

        random.shuffle(indices)

        train, val, test = indices[0: train_size], indices[train_size: train_size +
                                                           val_size], indices[size - test_size: size]

        # Create annotation files
        def write_split_to_file(file_name, samples):
            with open(ann_folder_path + file_name, 'w') as f:
                for idx in samples:
                    f.write(paths[idx])
                    f.write(str(len(annotations[idx])) + '\n')
                    f.write(str(annotations[idx]) + '\n')

        if mini:
            write_split_to_file('/mini_train.txt', train)
            write_split_to_file('/mini_val.txt', val)
            write_split_to_file('/mini_test.txt', test)
        else:
            write_split_to_file('/train.txt', train)
            write_split_to_file('/val.txt', val)
            write_split_to_file('/test.txt', test)

    def convert_annotations_to_bbox(self, anns):
        """
        Default annotations are defined as a an ellipses: 
            major_axis_radius, minor_axis_radius, angle, center_x, center_y, 1

        We convert them to bounding boxes in corner form
        TODO: re-check the math
        """
        new_anns = []
        for ann in anns:
            r_max, r_min, angle, center_x, center_y = ann[0], ann[1], ann[2], ann[3], ann[4]

            cos_sq = math.cos(angle) ** 2
            sin_sq = math.sin(angle) ** 2
            r_min_sq = r_min ** 2
            r_max_sq = r_max ** 2

            width = math.sqrt(r_min_sq * cos_sq + r_max_sq * sin_sq)
            height = math.sqrt(r_min_sq * sin_sq + r_max_sq * cos_sq)

            #new_anns.append([center_x, center_y, width, height])
            new_anns.append([center_x-width/2, center_y-height/2, center_x + width/2, center_y + height/2])

        return new_anns

    def debug_output_dataloader(self, images, boxes, labels):
        from torchvision.utils import draw_bounding_boxes
        import torchvision
        import time
        boxes = boxes.type(torch.uint8)
        #images = images.type(torch.uint8)

        images = images.mul(255).add_(0.5).clamp_(
            0, 255).to("cpu", torch.uint8)
        boxes = boxes.cpu()

        print(images.shape, boxes.shape, labels.shape)
        nr_faces = 0
        for i in range(0, len(labels)):
            if labels[i] == 1:
                nr_faces += 1
        img = draw_bounding_boxes(
            images, boxes[0:nr_faces], width=3, colors=(255, 255, 0))
        img = torchvision.transforms.ToPILImage()(img)
        img.show()
    
    def display_item(self, image, boxes):
        boxes = boxes.astype(np.uint8)
        for box in boxes:
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0))
        
        cv2.imshow('out.png', image)

    def __getitem__(self, idx):
        """
        Function that is called by the dataloader in order to return a sample at a specified index
        """
        image = cv2.imread(
            str(self.img_path + "/" + self.path_names[idx] + ".jpg"))
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = np.array(self.objects[idx])
        labels = np.array([1] * len(boxes))

        if self.debug_mode:
            self.display_item(image, boxes)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        #if self.debug_mode:
        #    self.debug_output_dataloader(image, boxes, labels)

        return image, boxes, labels

    def __len__(self):
        """
        Function that returns the size of the dataset
        """
        return len(self.path_names)
