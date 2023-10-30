using ArgParse

include("utils.jl")

args = let s = ArgParseSettings()
    @add_arg_table! s begin
        "--datapath"
            arg_type=String
            default="E:/WST_code/zhijiang_data/porto_data/porto/porto"
    end
    parse_args(s; as_symbols=true)
end

datapath = args[:datapath]
porto2h5("$datapath/train_new.csv")