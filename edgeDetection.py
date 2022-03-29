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


def edgeCompare(img1, img2):
    weight1 = img1.shape[1]
    height1 = img1.shape[0]

    same = 0
    for y in range(0, height1-1):
        for x in range(0, weight1-1):
            arrImg1 = img1[y, x, :]
            arrImg2 = img2[y, x, :]

            if (np.array_equal(arrImg1, arrImg2)):
                same += 1
    return same


if __name__ == '__main__':
    fileName = "./ftfm_out"
    filePath = "ftfm_out"
    files = os.listdir(fileName)
    #ans_scenes = [73, 235, 301, 370, 452, 861, 1281]
    ans_scenes = [[1, 8], 29, 49, 66, 90, 134, [148, 157], 178, 206, 225, [298, 305], 331,
                  355, 372, 394, 429, [446, 450], 483, 518, 549, 576, [594, 601], 630, 655, 674, 692, 730]
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
        edge_img1 = cv2.Canny(img1, 100, 200)
        edge_img2 = cv2.Canny(img2, 100, 200)
        res = edgeCompare(img1, img2)

        if(i == 10):
            cv2.imshow("original", img1)
            cv2.imshow("edge", edge_img1)
            cv2.waitKey(0)

        if(res <= 650):
            print("Scene:", str(i+1), "Same number of pixels", res)
            ans_arr.append(i+1)
    print(ans_arr)
    precision, recall = pr_cruve(ans_arr, my_ans_arr)
    print('precision:', precision)
    print('recall:', recall)
