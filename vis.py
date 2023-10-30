import pandas as pd
import matplotlib.pyplot as plt

def parse_log_file(log_file):
    # 读取训练日志文件
    with open(log_file, 'r') as f:
        lines = f.readlines()
    # 解析日志数据
    epochs = []
    train_mse_values = []
    test_mse_values = []
    for line in lines:
        if "Epoch" in line:
            epoch = int(line.split()[-1])
            epochs.append(epoch)
        elif "训练 MSE" in line:
            train_mse = float(line.split(":")[-2].split(",")[0])
            test_mse = float(line.split(":")[-1].split(",")[0])
            train_mse_values.append(train_mse)
            test_mse_values.append(test_mse)
    # 创建DataFrame来存储数据
    df = pd.DataFrame({'Epoch': epochs, 'Train MSE': train_mse_values, 'Test MSE': test_mse_values})
    return df

def plot_mse_curve(df,df_vec):
    # 绘制MSE曲线图
    plt.figure(figsize=(8, 6))
    l = min(len(df),len(df_vec))
    plt.plot(df['Epoch'].values[:l], df['Train MSE'].values[:l], c = "r",linestyle='--',label='Train MSE')
    plt.plot(df['Epoch'].values[:l], df['Test MSE'].values[:l], c = "r", label='Test MSE')
    plt.plot(df_vec['Epoch'].values[:l], df_vec['Train MSE'].values[:l], c = "b",linestyle='--', label='Train with vec MSE')
    plt.plot(df_vec['Epoch'].values[:l], df_vec['Test MSE'].values[:l], c = "b", label='Test with vec MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Train and Test MSE over Epochs')
    # plt.grid(True)
    plt.show()

if __name__ == "__main__":
    log_file_path_vec = "reult_log/result_with_vec/training_log.txt"  # 指定训练日志文件的路径
    df_vec = parse_log_file(log_file_path_vec)
    log_file_path = "reult_log/result/training_log.txt"  # 指定训练日志文件的路径
    df = parse_log_file(log_file_path)
    plot_mse_curve(df,df_vec)