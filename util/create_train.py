# Author: Wang Yongjie
# Email:  yongjie.wang@ntu.edu.sg
# Description: create training examples

import os
import random

def create_pair(items, num):
    """
    items:  image subdirectory lists
    num:    number of image pairs
    """

    image_list = []
    label_list = []

    for i in range(num):
        # generate positive pair
        tmp = []
        index = random.randint(len(items))
        anchor = random.randint(len(items[index]))
        positive = random.randint(len(items[index]))
        while anchor == positive:
            positive = random.randint(len(items[index]))
        tmp.append(items[index][anchor])
        tmp.append(items[index][positive])
        image_list.append(tmp)
        label_list.append(1)

        # generate negative pair
        for j in range(3):
            tmp = []
            index1 = random.randint(len(items))
            while index == index1:
                index1 = random.randint(len(items))
            
            tmp.append(items[index][anchor])
            tmp.append(items[index1][negative])
            image_list.append(tmp)
            label_list.append(0)


    return image_list, label_list



