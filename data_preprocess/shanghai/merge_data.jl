using CSV, DataFrames, HDF5
# import Libc
# import Base.Iterators: product
data_ROOT = "E:\\WST_code\\zhijiang_data\\shanghai\\raw_data\\ODLink"
# 定义要合并的H5文件数量范围
start_index = 1
end_index = 31

data_path = joinpath(data_ROOT, "shanghai.h5")
h5_input = h5open(data_path, "r")

timestamps = read(h5_input["/timestamps/1"])
trips = read(h5_input["/trips/1"])

