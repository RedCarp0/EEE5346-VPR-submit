import numpy as np
np.random.seed(0)


def combine_two(path1, path2):
    """
    this function is to combine two datasets (the diff and the easy version)
    :param path1: path for the dataset 1
    :param path2: path for the dataset 2
    :return: the array of the combined dataset
    """
    dataset1 = np.loadtxt(path1, dtype=str, delimiter=' ')
    dataset2 = np.loadtxt(path2, dtype=str, delimiter=' ')
    combine_dataset = np.concatenate((dataset1, dataset2), axis=0)
    print(combine_dataset.shape)
    return combine_dataset


def shuffle_dataset(dataset):
    """
    this function is to shuffle the dataset randomly
    :param dataset: the dataset to be shuffled
    :return: dataset after shuffled in array
    """
    np.random.shuffle(dataset)
    return dataset


def split_test(dataset):
    """
    this function is to split the dataset into train:val:test=8:1:1
    :param dataset: the dataset to be splitted
    :return: the test dataset in list (only 10% of the all data)
    """
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    train_num = int(len(dataset) * TRAIN_RATIO)
    val_num = train_num + int(len(dataset) * VAL_RATIO)

    train_data_list = dataset[:train_num]
    val_data_list = dataset[train_num:val_num]
    test_data_list = dataset[val_num:]
    return test_data_list


def main():
    # combine + shuffle + split
    AN_combine = combine_two("robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt")
    AN_combine = shuffle_dataset(AN_combine)
    AN_test = split_test(AN_combine)
    filename1 = 'qAutumn_dbNight.txt'
    np.savetxt(filename1, AN_test, fmt="%s", delimiter=' ')

    # combine + shuffle + split
    AS_combine = combine_two("robotcar_qAutumn_dbSunCloud_easy_final.txt", "robotcar_qAutumn_dbSunCloud_diff_final.txt")
    AS_combine = shuffle_dataset(AS_combine)
    AS_test = split_test(AS_combine)
    filename1 = 'qAutumn_dbSunCloud.txt'
    np.savetxt(filename1, AS_test, fmt="%s", delimiter=' ')

    # combine + shuffle
    ANall_combine = combine_two("robotcar_qAutumn_dbNight_easy_final.txt", "robotcar_qAutumn_dbNight_diff_final.txt")
    ANall_combine = shuffle_dataset(ANall_combine)
    filename1 = 'qAutumn_dbNight_all.txt'
    np.savetxt(filename1, ANall_combine, fmt="%s", delimiter=' ')

    # combine + shuffle
    ASall_combine = combine_two("robotcar_qAutumn_dbSunCloud_easy_final.txt",
                                "robotcar_qAutumn_dbSunCloud_diff_final.txt")
    ASall_combine = shuffle_dataset(ASall_combine)
    filename1 = 'qAutumn_dbSunCloud_all.txt'
    np.savetxt(filename1, ASall_combine, fmt="%s", delimiter=' ')

    # DIY_dataset = np.loadtxt("./gt_25/groundtruth_25.txt", dtype=str, delimiter=' ')
    # DIY_dataset = shuffle_dataset(DIY_dataset)[:2000]
    # print(len(DIY_dataset))
    # DIY_test = split_test(DIY_dataset)

    # combine_dataset = np.concatenate((DIY_test, AN_test, AS_test), axis=0)
    # print(combine_dataset)
    # filename3 = 'combine.txt'
    # np.savetxt(filename3,combine_dataset, fmt = "%s",delimiter=' ')


if __name__ == '__main__':
    main()
