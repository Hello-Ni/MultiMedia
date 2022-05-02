'''
N96091172 HSING-YUN, TSAI
Multimedia HW1 (shot change boundary detection)
'''
import os
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt


def get_EuclideanDistance(x, y):
    myX = np.array(x)
    myY = np.array(y)
    return np.sqrt(np.sum((myX - myY) * (myX - myY)))

# returns the edge-change-ratio
# the dilate_rate parameter controls the distance of the pixels between the frame


def compare_Edge(img1, img2):
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


if __name__ == '__main__':
    starttime = datetime.datetime.now()
    # fileName = "./soccer_out"
    # filePath = "soccer_out"
    # imgTitle = "soccer"

    fileName = "./ftfm_out"
    filePath = "ftfm_out"
    imgTitle = "ngc-"

    # 產生對比線條
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    centercolor = np.array([125, 125, 125])

    '''Read file'''
    ansNum = []
    for num in range(len(os.listdir(fileName))-1):
        strNum = str(num)
        zeroNum = 4 - len(strNum)
        imgName = imgTitle + "0"*zeroNum + strNum + ".jpeg"
        path = os.path.join(filePath, imgName)
        img1 = cv2.imread(path)

        strNum2 = str(num+1)
        zeroNum = 4 - len(strNum2)
        imgName = imgTitle + "0" * zeroNum + strNum2 + ".jpeg"
        path = os.path.join(filePath, imgName)
        img2 = cv2.imread(path)
        edge_img1 = cv2.Canny(img1, 100, 200)
        edge_img2 = cv2.Canny(img2, 100, 200)
        result = compare_Edge(img1, img2)
        if (result <= 650):
            print("Number:", strNum, "相同像素量:", result)
            ansNum.append(int(strNum))
        # else: print("XXXX Number:", strNum, "相同像素量:", result)

    ans_scenes = [[1, 8], 29, 49, 66, 90, 134, [148, 157], 178, 206, 225, [298, 305], 331,
                  355, 372, 394, 429, [446, 450], 483, 518, 549, 576, [594, 601], 630, 655, 674, 692, 730]
    ans_arr = []
    for scene in ans_scenes:
        if isinstance(scene, int):
            ans_arr.append(scene)
        else:
            for i in range(scene[0], scene[1]+1):
                ans_arr.append(i)
    soccer_out_precision, soccer_out_recall = pr_cruve(ans_arr, ansNum)
    print('soccer out precision: ', soccer_out_precision)
    print('soccer out recall: ', soccer_out_recall)

    endtime = datetime.datetime.now()
    print("執行時間：", (endtime - starttime))
