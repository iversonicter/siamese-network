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
        index = random.randrange(len(items))
        anchor = random.randrange(len(items[index]))
        positive = random.randrange(len(items[index]))
        while anchor == positive:
            positive = random.randrange(len(items[index]))
        tmp.append(items[index][anchor])
        tmp.append(items[index][positive])
        image_list.append(tmp)
        label_list.append(1)

        # generate negative pair
        for j in range(3):
            tmp = []
            index1 = random.randrange(len(items))
            while index == index1:
                index1 = random.randrange(len(items))

            negative = random.randrange(len(items[index1]))
            tmp.append(items[index][anchor])
            tmp.append(items[index1][negative])
            image_list.append(tmp)
            label_list.append(0)


    return image_list, label_list

def create_filelist(directory):
    file_list = []
    for i in os.listdir(directory):
        subdir = os.path.join(directory, i)
        tmp = []
        for j in os.listdir(subdir):
            files = os.path.join(subdir, j)
            tmp.append(files)
        file_list.append(tmp)

    return file_list

if __name__ == "__main__":
    file_list = create_filelist("/home/yongjie/github/siamese-network/data/orl_faces")
    image_list, label_list = create_pair(file_list, 5)
    print(image_list)
    print(label_list)

