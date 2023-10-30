using Pkg
Pkg.add("PyCall")
using CSV, DataFrames, HDF5
# import Libc
# import Base.Iterators: product

data_ROOT = "E:\\WST_code\\zhijiang_data\\shanghai\\raw_data\\ODLink"
# output_file = "hangzhou_202005.h5"

# 定义要合并的H5文件数量范围
start_index = 1
end_index = 30




h5open(joinpath(data_ROOT, "shanghai_201804.h5"), "w") do f
    num = 1
    for i in range(start_index,stop=end_index)
        if i < 10
            data_path = joinpath(data_ROOT, "HT18040$i" * ".pkl")
            println(data_path)
        else
            data_path = joinpath(data_ROOT, "HT1804$i" * ".pkl")
            println(data_path)
            end
        h5_input = h5open(data_path, "r")
            print(h5_input)
        for time_step in range(1,stop=length(h5_input["labels"]))
            deta_data = Int32(60*60*24)*(i-1)
            labels = read(h5_input["/labels/$time_step"])
            sources = read(h5_input["/sources/$time_step"])
            timestamps = read(h5_input["/timestamps/$time_step"]) .+ deta_data
            trips = read(h5_input["/trips/$time_step"])
            f["/labels/$num"] = labels
            f["/sources/$num"] = sources
            f["/timestamps/$num"] = timestamps
            f["/trips/$num"] = trips
            num % 10_000 == 0 && println("$num")
            num = num + 1
            end
            close(h5_input)
    end
      num = length(f["/timestamps"])
      attributes(f)["num"] = num
      println("Saved $num trips.")
    end
