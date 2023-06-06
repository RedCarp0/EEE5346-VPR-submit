import csv
import math
import os
import numpy as np
import random
import cv2


class GroundTruth:
    def __init__(self):
        self.dis_list = []
        self.north_list = []
        self.east_list = []
        self.yaw_list = []
        self.csv_name_list = []
        self.dis_list1 = []
        self.north_list1 = []
        self.east_list1 = []
        self.yaw_list1 = []
        self.data_list = []
        self.data_list1 = []
        self.graph_file_name = []

    def read_csv(self, csvFile_path):
        """
        this function read the corresponding GPS csv file to get the easting, northing, and yaw value.
        :param csvFile_path: the path of the GPS csv file
        :return: None
        """
        csvFile = open(csvFile_path, "r")
        reader = csv.reader(csvFile)

        for item in reader:
            if reader.line_num == 1:
                continue
            dis = self.cal_dis(float(item[5]), float(item[6]))
            self.dis_list.append(dis)
            self.north_list.append(float(item[5]))
            self.east_list.append(float(item[6]))
            self.yaw_list.append(float(item[14]))
            self.csv_name_list.append(int(item[0].rstrip(' ')))

        self.data_list = list(zip(self.csv_name_list, self.north_list, self.east_list, self.dis_list, self.yaw_list))

    def read_graph(self, path):
        """
        This function get the all name of the graphs (without .png)
        :param path: the path of the file with all graphs
        :return: names of the graphs (without .png) in list
        """
        # 获取所有图片的名字
        # path = "/home/pinoc/Desktop/navigation_data/2015-03-17-11-08-44/2015-03-17-11-08-44_01/stereo/centre"
        file_name_list = os.listdir(path)
        file_name = str(file_name_list)
        file_name = file_name.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
        file_name = file_name.split('.png,')[:-2]
        return file_name
        # print(file_name) #['1426590796320640', '1426590753451643', '1426590907305488'...]

    def align(self, path):
        """
        This function is to align the timestamp of each graph to the GPS data
        :param path: the path of the file with all graphs
        :return: None
        """
        self.graph_file_name = self.read_graph(path)
        for file in self.graph_file_name:
            file_ts = int(file)
            diff_list = np.array(self.csv_name_list) - np.ones_like(np.array(self.csv_name_list)) * file_ts
            idx = np.argmin(np.abs(diff_list))
            self.dis_list1.append(self.dis_list[idx])
            self.yaw_list1.append(self.yaw_list[idx])
            self.north_list1.append(self.north_list[idx])
            self.east_list1.append(self.east_list[idx])
        self.data_list1 = list(zip(self.graph_file_name, self.north_list1, self.east_list1, self.dis_list1,
                                   self.yaw_list1))  # [('1426590796320640', 5768992.580767565, 2.762767), ('1426590753451643', 5769227.829803004, 2.342556)]

    def cal_dis(self, north_value, east_value):
        """
        this function is to calculate the Euclidean distance
        :param north_value: the northing value in the GPS data
        :param east_value: the easting value in the GPS data
        :return: Euclidean distance
        """
        dis = math.sqrt(north_value ** 2 + east_value ** 2)
        return dis


# def img_show(name, img):
#     cv2.imshow(name, img)
#     cv2.waitKey(0)


if __name__ == '__main__':
    i = 0
    query = GroundTruth()
    query.read_csv("./2014-12-10-18-10-50_gps/gps/ins.csv")
    query.align("./query")
    ref = GroundTruth()
    ref.read_csv("./2015-03-17-11-08-44_gps/gps/ins.csv")
    ref.align("./ref")

    true_pair = []
    false_pair = []
    i = 0
    ref_len = len(ref.graph_file_name)
    for value1, value2, value3, value4 in zip(query.dis_list1, query.yaw_list1, query.north_list1, query.east_list1):
        true_index = 0
        j = 0
        dis_diff = np.array(ref.dis_list1) - np.ones_like(np.array(ref.dis_list1)) * value1  # <50m
        yaw_diff = np.array(ref.yaw_list1) - np.ones_like(np.array(ref.yaw_list1)) * value2  # <70度
        north_diff = np.array(ref.north_list1) - np.ones_like(np.array(ref.north_list1)) * value3  # <50m
        east_diff = np.array(ref.east_list1) - np.ones_like(np.array(ref.east_list1)) * value4  # <70度
        for p, q, r, s in zip(dis_diff, yaw_diff,north_diff,east_diff):
            # print(p,q)
            if abs(p) <= 25 and abs(q) <= 50 and abs(r) <= 25 and abs(s) <= 25:  # criterion to make groundtruth table
                # print(i,j)
                true_pair.append(['./query/'+query.graph_file_name[i]+'.png', './ref/'+ref.graph_file_name[j]+'.png'])
                true_index = 1
                break
            j = j + 1
        if true_index != 1:  # 说明没有任何匹配
            false_pair.append(['./query/'+query.graph_file_name[i]+'.png', './ref/' + ref.graph_file_name[random.randint(0, ref_len-1)]+'.png'])
        i = i + 1

    true_len = len(true_pair)
    false_len = len(false_pair)
    print(true_len, false_len)
    # ratio = math.floor(false_len/true_len)
    f = open('./groundtruth_25.txt', 'w')
    write_true_list = random.sample([i for i in range(0, true_len-1)], false_len)
    for item in write_true_list:
        f.write(true_pair[item][0] + ', ' + true_pair[item][1] + ', ' + '1' + '\n')
    for k in range(false_len):
        f.write(false_pair[k][0] + ', ' + false_pair[k][1] + ', ' + '0' + '\n')
    f.close()

# if len(true_pair)>5000:
#     for k in range(0, len(true_pair), 5000):
#         img1 = cv2.imread(true_pair[k][0])
#         img2 = cv2.imread(true_pair[k][1])
#
#         # 水平组合
#         imghstack = np.hstack((img1, img2))
#         # cv2.namedWindow("truepair")
#         # cv2.imshow("Ture Pair", imghstack)
#         cv2.imwrite('diff_{}.png'.format(k/5000), imghstack, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#         # cv2.waitKey(0)
# elif 500 < len(true_pair) < 5000:
#     for k in range(0, len(true_pair), 500):
#         img1 = cv2.imread(true_pair[k][0])
#         img2 = cv2.imread(true_pair[k][1])
#         # 水平组合
#         imghstack = np.hstack((img1, img2))
#         # cv2.namedWindow("truepair")
#         # cv2.imshow("Ture Pair", imghstack)
#         cv2.imwrite('diff_{}.png'.format(k/500), imghstack, [cv2.IMWRITE_PNG_COMPRESSION, 0])
#         # cv2.waitKey(0)
# elif len(true_pair) < 500:
#     for k in range(0, len(true_pair), 50):
#         img1 = cv2.imread(true_pair[k][0])
#         img2 = cv2.imread(true_pair[k][1])
#         # 水平组合
#         imghstack = np.hstack((img1, img2))
#         # cv2.namedWindow("truepair")
#         # cv2.imshow("Ture Pair", imghstack)
#         cv2.imwrite('diff_{}.png'.format(k/50), imghstack, [cv2.IMWRITE_PNG_COMPRESSION, 0])