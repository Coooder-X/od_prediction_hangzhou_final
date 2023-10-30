#=
read_data:
- Julia version: 
- Author: Administrator
- Date: 2023-07-27
=#
using HDF5
h5open("E://WST_code/zhijiang_code/t2vec/t2vec-master/t2vec-master/data/porto/porto.h5", "r") do f
#     trainsrc = open("$datapath/train.src", "w")
    println(length(f["/trip_ID"]))
    TRIP_ID = f["/trip_ID/17041"] |> read
    println(TRIP_ID)
end