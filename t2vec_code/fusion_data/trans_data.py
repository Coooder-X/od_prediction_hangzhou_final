import pandas as pd
import numpy as np
df10 = pd.read_csv('../data/temp_data/train_all_temp.csv', header=0)
# print(df10["vec"][0].shape)

# df10["vec"].values[0] = df10["vec"].values[0].replace('\n',' ')
# print(np.fromstring(df10["vec"].values[0], dtype= float , sep='').shape)
# print(len(eval(df10["vec"][0])))
for i in range(len(df10)):
    df10["vec"][i] = np.array(eval(df10["vec"][i]))
    print(df10["vec"][i].shape)
