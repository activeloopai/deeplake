import argparse
import os
import struct
from array import array as pyarray

import numpy as np
import cv2
import hub
from hub.collections import dataset, tensor
from xml.etree import cElementTree as ElementTree

class XmlListConfig(list):
    def __init__(self, aList):
        for element in aList:
            if element:
                # treat like dict
                if len(element) == 1 or element[0].tag != element[1].tag:
                    self.append(XmlDictConfig(element))
                # treat like list
                elif element[0].tag == element[1].tag:
                    self.append(XmlListConfig(element))
            elif element.text:
                text = element.text.strip()
                if text:
                    self.append(text)


class XmlDictConfig(dict):
    '''
    Example usage:

    >>> tree = ElementTree.parse('your_file.xml')
    >>> root = tree.getroot()
    >>> xmldict = XmlDictConfig(root)

    Or, if you want to use an XML string:

    >>> root = ElementTree.XML(xml_string)
    >>> xmldict = XmlDictConfig(root)

    And then use xmldict for what it is... a dict.
    '''
    def __init__(self, parent_element):
        if parent_element.items():
            self.update(dict(parent_element.items()))
        for element in parent_element:
            if element:
                # treat like dict - we assume that if the first two tags
                # in a series are different, then they are all different.
                if len(element) == 1 or element[0].tag != element[1].tag:
                    aDict = XmlDictConfig(element)
                # treat like list - we assume that if the first two tags
                # in a series are the same, then the rest are the same.
                else:
                    # here, we put the list in dictionary; the key is the
                    # tag name the list elements all share in common, and
                    # the value is the list itself 
                    aDict = {element[0].tag: XmlListConfig(element)}
                # if the tag has attributes, add those to the dict
                if element.items():
                    aDict.update(dict(element.items()))
                self.update({element.tag: aDict})
            # this assumes that if you've got an attribute in a tag,
            # you won't be having any text. This may or may not be a 
            # good idea -- time will tell. It works for the way we are
            # currently doing XML configuration files...
            elif element.items():
                self.update({element.tag: dict(element.items())})
            # finally, if there are no child tags and no attributes, extract
            # the text
            else:
                self.update({element.tag: element.text})
                
                
                
def transform_img(img):
    img = cv2.imread(img)
    return cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

def load_pascal_detection(path):

    # get train path
    fname_img_train = os.path.join(path, "train/Images")
    fname_lbl_train = os.path.join(path, "train/DetectionLabels")
    # get test path
    fname_img_test = os.path.join(path, "test/Images")
    fname_lbl_test = os.path.join(path, "test/DetectionLabels")

    train_imgs = sorted(
        [
            os.path.join(fname_img_train, f) for f in os.listdir(fname_img_train)
        ]
    )
    test_imgs = sorted(
        [
            os.path.join(fname_img_test, f) for f in os.listdir(fname_img_test)
        ]
    )
    
    train_lbls = sorted(
        [
            os.path.join(fname_lbl_train, f) for f in os.listdir(fname_lbl_train)
        ]
    )
    test_lbls = sorted(
        [
            os.path.join(fname_lbl_test, f) for f in os.listdir(fname_lbl_test)
        ]
    )
    
    train_imgs = [transform_img(img) for img in train_imgs]
    test_imgs = [transform_img(img) for img in test_imgs]
    
    
    train_lbls = [ElementTree.parse(lbl) for lbl in train_lbls]
    test_lbls = [ElementTree.parse(lbl) for lbl in test_lbls]
    
    train_lbls = [XmlDictConfig(lbl.getroot()) for lbl in train_lbls]
    test_lbls = [XmlDictConfig(lbl.getroot()) for lbl in test_lbls]
    
    print('a',train_lbls[0])
    
    
    
    #We must change bounding box coordinates because image have been resized
    for i in range(len(train_lbls)):
        train_lbls[i]['object']['bndbox']['xmin'] = str(float(train_lbls[i]['object']['bndbox']['xmin'])*256/float(train_lbls[i]['size']['width']))
        train_lbls[i]['object']['bndbox']['ymin'] = str(float(train_lbls[i]['object']['bndbox']['ymin'])*256/float(train_lbls[i]['size']['height']))
        train_lbls[i]['object']['bndbox']['xmax'] = str(float(train_lbls[i]['object']['bndbox']['xmax'])*256/float(train_lbls[i]['size']['width']))
        train_lbls[i]['object']['bndbox']['ymax'] = str(float(train_lbls[i]['object']['bndbox']['ymax'])*256/float(train_lbls[i]['size']['height']))
        train_lbls[i]['size']['width'] = '256'
        train_lbls[i]['size']['height'] = '256'
    
    for i in range(len(test_lbls)):
        test_lbls[i]['object']['bndbox']['xmin'] = str(float(test_lbls[i]['object']['bndbox']['xmin'])*256/float(test_lbls[i]['size']['width']))
        test_lbls[i]['object']['bndbox']['ymin'] = str(float(test_lbls[i]['object']['bndbox']['ymin'])*256/float(test_lbls[i]['size']['height']))
        test_lbls[i]['object']['bndbox']['xmax'] = str(float(test_lbls[i]['object']['bndbox']['xmax'])*256/float(test_lbls[i]['size']['width']))
        test_lbls[i]['object']['bndbox']['ymax'] = str(float(test_lbls[i]['object']['bndbox']['ymax'])*256/float(test_lbls[i]['size']['height']))
        test_lbls[i]['size']['width'] = '256'
        test_lbls[i]['size']['height'] = '256'
    
    train_lbls = np.array(list(train_lbls[0].items()))
    test_lbls = np.array(list(test_lbls[0].items()))
    
    train = {"imgs": train_imgs, "lbls": train_lbls}
    test = {"imgs": test_imgs, "lbls": test_lbls}
    
    return train, test     



def main():
    

    train, test = load_pascal_detection(path)
    
    train_images = np.concatenate([img for img in train["imgs"]])
    train_labels = np.concatenate([lbl for lbl in train["lbls"]])
    
    test_images = np.concatenate([img for img in test["imgs"]])
    test_labels = np.concatenate([lbl for lbl in test["lbls"]])

    
    train_images = tensor.from_array(train_images, dtag="imgs")
    train_labels = tensor.from_array(train_labels, dtag="lbls")
    
    test_images = tensor.from_array(test_images, dtag="imgs")
    test_labels = tensor.from_array(test_labels, dtag="lbls")

    train_ds = dataset.from_tensors({"data": train_images, "labels": train_labels})
    test_ds = dataset.from_tensors({"data": test_images, "labels": test_labels})
    
    train_ds.store('arenbeglaryan/vocdetection')


if __name__ == "__main__":
    main()

