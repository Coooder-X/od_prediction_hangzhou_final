# import pandas as pd
# import numpy as np
# import random
# random.seed()
# df = pd.DataFrame({'str':['a', 'a', 'b', 'b', 'a'],
# 'no':['one', 'two', 'one', 'two', 'one'],
# 'data1':np.random.randn(5),
# 'data2':[np.random.randn(5)]*5})
# grouped=df.groupby(["str","no"])
# # for name,group in grouped:
# #     print(name)
# #     print(group["data2"].min())
# #     print(group["data2"].max())
# #     print(group["data2"].mean())
#
# data = df.groupby(["str","no"])["data2"]
# data.agg({'data2': 'sum'})
# print(data)
# data = data.reset_index(name='data2')
# print(data)
# pickup_order_records = df[["str", "no"]].groupby(by=["str", "no"])['str'].size()
# pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')
# for i in range(len(pickup_order_records)):
#     grouped[]
#
import pandas as pd
import numpy as np
# 示例数据，第三列是由列表组成
data = {
    'col1': ['A', 'A', 'B', 'B', 'A'],
    'col2': ['X', 'Y', 'X', 'Y', 'X'],
    'col3': [np.zeros(5), np.ones(5), np.zeros(5), np.zeros(5), np.zeros(5)]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 定义一个自定义函数来合并数列
def merge_lists(lst):
    data = np.stack(lst)
    return np.stack([data.max(axis=0),data.mean(axis=0),data.min(axis=0)])


pickup_order_records = df[['col1', 'col2']].groupby(by=['col1', 'col2'])[
    'col1'].size()


pickup_order_records = pickup_order_records.reset_index(name='pickup_order_num')
df['merged_col3'] = df.groupby(['col1', 'col2'],as_index=False)['col3'].apply(merge_lists)
df.drop('col3', axis=1, inplace=True)
df = df.dropna(axis=0, how='any')
out = pd.merge(pickup_order_records, df, on=['col1', 'col2'])
print(out.shape)

