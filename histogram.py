import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import glob


def pr_cruve(answer_image_number, my_answer_image_number):
    TP = 0
    FP = 0
    FN = 0

    np_answer_image_number = np.array(answer_image_number)
    np_my_answer_image_number = np.array(my_answer_image_number)

    TP_array_result = np.in1d(answer_image_number, my_answer_image_number)
    TP_array = np_answer_image_number[TP_array_result]
    # print(TP_array)
    TP = len(TP_array)
    FP = len(my_answer_image_number) - TP
    FN = len(answer_image_number) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return (precision, recall)


def colorDegreeCalculate(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])
    degree = 0

    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            # print(hist1[i], hist2[i])

            degree = degree + \
                (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree+1
    degree = degree/len(hist1)
    return degree


def RGB_Historgram(img1, img2, size):
    # compare the similarity between img1 and img2
    img1 = cv2.resize(img1, size)
    img2 = cv2.resize(img2, size)

    rgb_img1 = cv2.split(img1)
    rgb_img2 = cv2.split(img2)
    similarity = 0

    # calculate every tunnel similarity
    for i1, i2 in zip(rgb_img1, rgb_img2):
        similarity = similarity + colorDegreeCalculate(i1, i2)
    similarity = similarity / 3
    return similarity


if __name__ == '__main__':
    # For news dataset
    fileName = "./news_out"
    filePath = "news_out"

    # for ngc dataset
    fileName = "./ngc_out"
    filePath = "ngc_out"
    files = os.listdir(fileName)
    # ans_scenes = [73, 235, 301, 370, 452, 861, 1281]
    ans_scenes = [[127, 164], [196, 253], 285, 340, 383, [384, 444], 456, [516, 535], [540, 573], [573, 622], [622, 664], 683, 703, 722, [728, 748], [
        760, 816], [816, 838], [840, 851], 859, 868, 876, 885, 897, 909, 921, 933, 943, 958, 963, 965, 969, 976, 986, [1003, 1009], 1038, [1048, 1059]]
    ans_arr = []
    my_ans_arr = []
    for scene in ans_scenes:
        if isinstance(scene, int):
            ans_arr.append(scene)
        else:
            for i in range(scene[0], scene[1]+1):
                ans_arr.append(i)
    # print(ans_arr)
    for i in range(len(os.listdir(fileName))-1):
        path = os.path.join(filePath, files[i])
        img1 = cv2.imread(path)

        path = os.path.join(filePath, files[i+1])
        img2 = cv2.imread(path)
        res = RGB_Historgram(img1, img2, (300, 300))
        if(res < 0.8):
            my_ans_arr.append(i+1)
            print("Scene:", str(i+1), "Similarity:", '%.3f' % res)

    precision, recall = pr_cruve(ans_arr, my_ans_arr)
    print('precision:', precision)
    print('recall:', recall)
