import cv2
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
np.random.seed(0)


DIY1_DATA_NUM = 2000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

def compute_orb_keypoints(filename):
    """
    Reads image from filename and computes ORB keypoints
    Returns image, keypoints and descriptors. 
    """
    # load image
    img = cv2.imread(filename)

    # create orb object
    orb = cv2.ORB_create()

    # set parameters 
    orb.setScoreType(cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    orb.setWTA_K(3)

    # detect keypoints
    kp = orb.detect(img, None)

    # for detected keypoints compute descriptors. 
    kp, des = orb.compute(img, kp)

    return img, kp, des


def brute_force_matcher(des1, des2):
    """
    Brute force matcher to match ORB feature descriptors
    """
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    return matches


def compute_fundamental_matrix(filename1, filename2, idx, dataset, inlier_threshold):
    """
    Takes in filenames of two input images 
    Return Fundamental matrix computes 
    using 8 point algorithm
    """
    # compute ORB keypoints and descriptor for each image
    img1, kp1, des1 = compute_orb_keypoints(filename1)
    img2, kp2, des2 = compute_orb_keypoints(filename2)

    # compute keypoint matches using descriptor
    matches = brute_force_matcher(des1, des2)

    # extract points
    pts1 = []
    pts2 = []
    good_matches = []
    for i, (m) in enumerate(matches):
        if m.distance < 20:
            # print(m.distance)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            good_matches.append(matches[i])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)

    # Compute fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, ransacReprojThreshold=0.5, confidence=0.99)

    # We select only inlier points
    if mask is None:
        # print(0, end="\t")
        return 0, len(good_matches), 0
    else:
        inlier_matches = [b for a, b in zip(mask, good_matches) if a]

    # print(len(inlier_matches), end="\n")

    # if idx==100 or 250 or 350 :
    #     final_img1 = cv2.drawMatches(img1, kp1, img2, kp2, matches,None)
    #     cv2.imwrite('./baseline/BF_matches_{}_{}.png'.format(dataset.strip("./txt"),str(np.floor(idx / 100))), final_img1,
    #                 [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     # cv2.imshow("BF Matches", final_img1)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #
    #     final_img2 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches,None)
    #     cv2.imwrite('./baseline/Good_matches_{}_{}.png'.format(dataset.strip("./txt"), str(np.floor(idx / 100))),
    #                 final_img2,
    #                 [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     # cv2.imshow("Good Matches", final_img)
    #     # cv2.waitKey()
    #     # cv2.destroyAllWindows()
    #
    #     final_img3 = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches,None)
    #     cv2.imwrite('./baseline/Inlier_matches_{}_{}.png'.format(dataset.strip("./txt"),str(np.floor(idx / 100))),
    #                 final_img3,[cv2.IMWRITE_PNG_COMPRESSION, 0])
        # cv2.imshow("Inlier Matches", final_img)
        # if (cv2.waitKey(25) & 0xFF) == 27:
        #     cv2.destroyAllWindows()
        #     sys.exit()
        # else:
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        # imghstack = np.hstack((final_img1, final_img2))
        # cv2.imwrite('./baseline/{}_{}.png'.format(dataset_name.strip('.txt'), str(int(idx/100))), imghstack,
        #             [cv2.IMWRITE_PNG_COMPRESSION, 0])

    if len(inlier_matches) > inlier_threshold:  # set this condition to what you need
        return 1, len(good_matches), len(inlier_matches)
    else:
        return 0, len(good_matches), len(inlier_matches)


def evaluate(gt_txt, inlier_threshold):
    id = 0
    match_points = []
    pre_labels = []
    gt_labels = []
    fp = open(gt_txt, "r")
    for line in tqdm(fp):
        line_str = line.split(", ")
        query, reference, gt = line_str[0], line_str[1], int(line_str[2])
        if query[0] == './':
            query.lstrip('./')
        if reference[0] == './':
            reference.lstrip('./')
        root = '/media/pinoc/Nicole/slam_data/navigation_data/img_data/'
        # print(gt_txt)
        try:
            x, y, z = compute_fundamental_matrix(root + query, root + reference, id,gt_txt, inlier_threshold)
            match_points.append(z)
            pre_labels.append(x)
            gt_labels.append(gt)
        except Exception as e:
            print(str(e))
            print(line, id)
        id = id + 1
    return np.array(match_points), np.array(gt_labels), np.array(pre_labels)


def precision_recall_score(pre_label_list, gt_label_list):
    """
    通过预测的label列表和实际的label列表，用TP / (TP + FP)和TP / (TP + FN)计算精度和回调
    :param pre_label_list: 预测出来的label列表
    :param gt_label_list: 实际label列表（groundtruth）
    :return: precision，recall
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    label_len = len(pre_label_list)
    for i in range(label_len):
        if gt_label_list[i] == 1 and pre_label_list[i] == 1:
            TP = TP + 1
        if gt_label_list[i] == 1 and pre_label_list[i] == 0:
            FN = FN + 1
        if gt_label_list[i] == 0 and pre_label_list[i] == 1:
            FP = FP + 1
        if gt_label_list[i] == 0 and pre_label_list[i] == 0:
            TN = TN + 1
    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0

    try:
        recall = TP / (TP + FN)
    except ZeroDivisionError:
        recall = 0
    return precision, recall


def ave_precision_score(pre_prob_list, gt_label_list):
    """
    计算mAP
    :param pre_prob_list: 预测出来的概率(0-1之间)列表
    :param gt_label_list: 实际label列表（groundtruth）
    :return: ave_presicion：平均精度（一个值）
    """
    ave_presicion = average_precision_score(y_true=gt_label_list, y_score=pre_prob_list)
    return ave_presicion


def pre_recall_curve(pre_prob_list, gt_label_list):
    """
    求得PR曲线的precision列表和recall列表
    :param pre_prob_list: 预测出来的概率(0-1之间)列表
    :param gt_label_list: 实际label列表（groundtruth）
    :return: precision_curve: PR曲线中的Precision列表, recall_curve：PR曲线中的Recall列表
    """
    precision_curve, recall_curve, _ = precision_recall_curve(y_true=gt_label_list, probas_pred=pre_prob_list)
    return precision_curve, recall_curve


def main():
    datasets = ["qAutumn_dbNight.txt","qAutumn_dbSunCloud.txt","qAutumn_dbNight_all.txt","qAutumn_dbSunCloud_all.txt"]
    threshold = range(4,14)

    for dataset in datasets:
        precision_list = []
        recall_list = []
        print("-------- Processing {} ----------".format(dataset))
        for th_value in threshold:
            match_points, gt_label, pre_label = evaluate(dataset, th_value)
            # print(match_points, labels)
            scaled_scores = match_points / max(match_points)
            precision, recall = precision_recall_score(pre_label, gt_label)
            precision_list.append(precision)
            recall_list.append(recall)
            print('{}_Precision_{} = {:.3f}'.format(dataset.strip("./txt"),th_value, precision), end = '\n')
            print('{}_Recall_{} = {:.3f}'.format(dataset.strip("./txt"),th_value, recall), end = '\n')
            precision_curve, recall_curve = pre_recall_curve(scaled_scores, gt_label)
            ave_precision = ave_precision_score(scaled_scores, gt_label)
            print('{}_AP_{} = {:.3f}'.format(dataset.strip("./txt"),th_value, ave_precision), end='\n')
        draw_mask = [False if (p==0 or r==0) else True for (p, r) in zip(precision_list, recall_list)]
        draw_precision = np.array(precision_list)[draw_mask]
        draw_recall = np.array(recall_list)[draw_mask]
        recall_ones = recall_list.count(1.0)
        draw_match_thresh_list = np.array(threshold)[draw_mask]

        fig, axes = plt.subplots(2,1,figsize=(10,10))
        axes[0].plot(recall_list, precision_list, '-.')
        axes[0].set_xlabel('recall')
        axes[0].set_ylabel('precision')
        axes[0].set_title('Precision-Recall curve')
        axes[0].set_xlim(0, 1)
        axes[0].set_ylim(0, 1)
        axes[1].plot(threshold, recall_list, label='recall')
        axes[1].plot(threshold, precision_list, label='precision' )
        axes[1].set_xlabel('match threshold')
        axes[1].set_ylabel('precision / recall')
        axes[1].set_title('Precision/Recall-Threshold curve')
        axes[1].legend()
        plt.savefig("curve_{}".format((dataset.strip("./txt"))))
        plt.clf()


if __name__ == '__main__':
    main()
