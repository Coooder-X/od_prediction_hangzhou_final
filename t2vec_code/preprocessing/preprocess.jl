
using JSON
using DataStructures
using NearestNeighbors
using Serialization, ArgParse
include("SpatialRegionTools.jl")

args = let s = ArgParseSettings()
    @add_arg_table! s begin
        "--datapath"
            arg_type=String
            default="E://WST_code//zhijiang_data//harbin"
    end
    parse_args(s; as_symbols=true)
end

datapath = args[:datapath]
param  = JSON.parsefile("t2vec_code/hyper-parameters.json")
regionps = param["region"]
cityname = regionps["cityname"]
cellsize = regionps["cellsize"]

if !isfile("$datapath/$cityname.h5")
    println("Please provide the correct hdf5 file $datapath/$cityname.h5")
    exit(1)
end

region = SpatialRegion(cityname,
                       regionps["minlon"], regionps["minlat"],
                       regionps["maxlon"], regionps["maxlat"],
                       cellsize, cellsize,
                       regionps["minfreq"], # minfreq
                       40_000, # maxvocab_size
                       10, # k
                       4) # vocab_start

# 需要对划分方法进行修改，
# 目前计划的划分方法：

println("Building spatial region with:
        cityname=$(region.name),
        minlon=$(region.minlon),
        minlat=$(region.minlat),
        maxlon=$(region.maxlon),
        maxlat=$(region.maxlat),
        xstep=$(region.xstep),
        ystep=$(region.ystep),
        minfreq=$(region.minfreq)")

paramfile = "$datapath/$(region.name)-param-cell$(Int(cellsize))"
if isfile(paramfile)
    println("Reading parameter file from $paramfile")
    region = deserialize(paramfile)
else
    println("Creating paramter file $paramfile")
    num_out_region = makeVocab!(region, "$datapath/$cityname.h5")
    serialize(paramfile, region)
end

println("Vocabulary size $(region.vocab_size) with cell size $cellsize (meters)")
println("Creating training and validation datasets...")
createTrainVal(region, "$datapath/$cityname.h5", datapath, downsamplingDistort, 1_170_000, 101_000)
# createTrainVal(region, "$datapath/$cityname.h5", datapath, downsamplingDistort, 10_000, 1_000)
saveKNearestVocabs(region, datapath)
