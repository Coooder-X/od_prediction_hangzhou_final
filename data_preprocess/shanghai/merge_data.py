import h5py
import os
import pickle as pkl
import numpy as np
data_ROOT = r"E:\WST_code\zhijiang_data\shanghai\raw_data\ODLink"
save_ROOT = r"E:\WST_code\zhijiang_data\shanghai\raw_data"
# output_file = "hangzhou_202005.h5"

# 定义要合并的H5文件数量范围
start_index = 1
end_index = 30

with h5py.File(os.path.join(save_ROOT, "hangzhou_202005.h5"), "w") as f:
    num = 1
    for i in range(start_index, end_index + 1):
        if i < 10:
            data_path = os.path.join(data_ROOT, f"HT18040{i}.pkl")
            print(data_path)
        else:
            data_path = os.path.join(data_ROOT, f"HT1804{i}.pkl")
            print(data_path)

        with open(data_path, 'rb') as f1:
            data = pkl.load(f1)
            print(len(data))
        for j in range(len(data)):
            delta_data = 60 * 60 * 24 * (i - 1)
            trip_data = np.array(data[j])
            timestamps = trip_data[:,0]
            trips = trip_data[:,[1,2]]
            f.create_dataset(f"/timestamps/{num}", data=timestamps)
            f.create_dataset(f"/trips/{num}", data=trips)

            if num % 10_000 == 0:
                print(num)

            num += 1
#
        f.attrs["num"] = num-1
        f.close()
        print(np.array(data[i]).shape)


# import h5py
# # 打开一个HDF5文件（如果文件不存在，则会创建它）
# with h5py.File('example.h5', 'w') as f:
#     num = 'some_dataset_name'  # 数据集的名称
#     timestamps = [1, 2, 3, 4, 5]  # 要保存的时间戳数据
#
#     # 创建一个数据集并将数据写入其中
#     f.create_dataset(f"/timestamps/{num}", data=timestamps)


    #     for time_step in range(1, len(h5_input["labels"]) + 1):
    #
    #         delta_data = 60 * 60 * 24 * (i - 1)
    #         labels = h5_input["/labels"][str(time_step)][()]
    #         sources = h5_input["/sources"][str(time_step)][()]
    #         timestamps = h5_input["/timestamps"][str(time_step)][()] + delta_data
    #         trips = h5_input["/trips"][str(time_step)][()]
    #
    #         f.create_dataset(f"/labels/{num}", data=labels)
    #         f.create_dataset(f"/sources/{num}", data=sources)
    #         f.create_dataset(f"/timestamps/{num}", data=timestamps)
    #         f.create_dataset(f"/trips/{num}", data=trips)
    #
    #         if num % 10_000 == 0:
    #             print(num)
    #
    #         num += 1
    #
    #     h5_input.close()
    #
    # num = len(f["/timestamps"])
    # f.attrs["num"] = num
    # print(f"Saved {num} trips.")