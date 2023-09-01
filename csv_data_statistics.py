# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルからデータを読み込む
data = pd.read_csv("padded_data_for_1dcnn_length_1024\padded_one_hot_encoding_all_2.csv")

# 各クラスの頻度をカウント
class_0_count = data['Class_0'].sum()
class_1_count = data['Class_1'].sum()
class_2_count = data['Class_2'].sum()

# グラフのプロット（各クラスの頻度を示す棒グラフ）
classes = ['Stage_0', 'Stage_1', 'Stage_2']
frequencies = [class_0_count, class_1_count, class_2_count]

plt.bar(classes, frequencies)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class 2 data_size=512')
plt.show()

# 同様の操作を別のCSVファイルに対して繰り返す
data = pd.read_csv("padded_data_for_1dcnn_length_1024\padded_one_hot_encoding_all_1.csv")

class_0_count = data['Class_0'].sum()
class_1_count = data['Class_1'].sum()
class_2_count = data['Class_2'].sum()

classes = ['Stage_0', 'Stage_1', 'Stage_2']
frequencies = [class_0_count, class_1_count, class_2_count]

plt.bar(classes, frequencies)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Frequency of Each Class 1 data_size=512')
plt.show()

# 箱ひげ図を描画する関数を定義
def plot_boxplot_with_mean(data, title):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    
    # 平均値を計算して表示
    means = data.mean()
    for x, mean in enumerate(means):
        plt.text(x, mean + 0.01, f'{mean:.2f}', ha='center', color='red')
    
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Value')
    plt.xticks(rotation=45)
    plt.show()

# 範囲内のファイルを読み込んでデータを集計する
frequency_table = pd.DataFrame(columns=['Stage_0', 'Stage_1', 'Stage_2'])

# 1から100までのCSVファイルに対して繰り返す
for n in range(1, 101):
    file_path = f'data_for_1dcnn/one_hot_encoding_{n}_2.csv'
    data = pd.read_csv(file_path)
    class_counts = data[['Stage_0', 'Stage_1', 'Stage_2']].sum()
    frequency_table.loc[n] = class_counts

# 箱ひげ図を描画し、平均値も表示
plot_boxplot_with_mean(frequency_table, 'Class Frequency Boxplot with Mean 2')

# 同様の操作を別のCSVファイルに対して繰り返す
for n in range(1, 101):
    file_path = f'data_for_1dcnn/one_hot_encoding_{n}_1.csv'
    data = pd.read_csv(file_path)
    class_counts = data[['Stage_0', 'Stage_1', 'Stage_2']].sum()
    frequency_table.loc[n] = class_counts

# 箱ひげ図を描画し、平均値も表示
plot_boxplot_with_mean(frequency_table, 'Class Frequency Boxplot with Mean 1')

# 各CSVファイルごとにEpochの総数を集計しリストに格納
epoch_totals = []

# 1から100までのCSVファイルに対して繰り返す
for n in range(1, 101):
    file_path = f"{n}_1.csv"
    
    try:
        df = pd.read_csv(file_path)
        epoch_total = df['Epoch'].iloc[-1]  # 最終行のEpochの値を総数として取得
        epoch_totals.append((n, epoch_total))  # (n, Epochの総数)のタプルを格納
    except FileNotFoundError:
        print(f"File {file_path} not found.")

# グラフ作成（横軸：n、縦軸：Epochの総数）
n_values, total_epochs = zip(*epoch_totals)  # タプルを分解してリストに変換
plt.figure(figsize=(10, 5))
plt.plot(n_values, total_epochs, marker='o')
plt.xlabel('n (CSV File Index)')
plt.ylabel('Total Epochs')
plt.title('Epochs per CSV File')
plt.grid(True)
plt.show()

# ヒストグラム作成
plt.figure(figsize=(10, 5))
plt.hist(total_epochs, bins=20, edgecolor='black')
plt.xlabel('Epochs')
plt.ylabel('Frequency')
plt.title('Histogram of Epoch Counts 1')
plt.grid(True)
plt.show()

# 統計値計算
max_n, max_epoch = max(epoch_totals, key=lambda x: x[1])
min_n, min_epoch = min(epoch_totals, key=lambda x: x[1])
mean_epoch = sum(total_epochs) / len(total_epochs)
median_epoch = sorted(total_epochs)[len(total_epochs) // 2]
variance_epoch = sum((x - mean_epoch) ** 2 for x in total_epochs) / len(total_epochs)
std_deviation_epoch = variance_epoch ** 0.5

# 統計値出力
print("Statistics of Epoch Counts:")
print(f"Maximum: n={max_n}, Epochs={max_epoch}")
print(f"Minimum: n={min_n}, Epochs={min_epoch}")
print(f"Mean: {mean_epoch:.2f}")
print(f"Median: {median_epoch}")
print(f"Variance: {variance_epoch:.2f}")
print(f"Standard Deviation: {std_deviation_epoch:.2f}")

# 同様の操作を別のCSVファイルに対して繰り返す（別のCSVファイルセット）
epoch_totals = []

for n in range(1, 101):
    file_path = f"{n}_2.csv"
    
    try:
        df = pd.read_csv(file_path)
        epoch_total = df['Epoch'].iloc[-1]  # 最終行のEpochの値を総数として取得
        epoch_totals.append((n, epoch_total))  # (n, Epochの総数)のタプルを格納
    except FileNotFoundError:
        print(f"File {file_path} not found.")

# グラフ作成（横軸：n、縦軸：Epochの総数）
n_values, total_epochs = zip(*epoch_totals)  # タプルを分解してリストに変換
plt.figure(figsize=(10, 5))
plt.plot(n_values, total_epochs, marker='o')
plt.xlabel('n (CSV File Index)')
plt.ylabel('Total Epochs')
plt.title('Epochs per CSV File')
plt.grid(True)
plt.show()

# ヒストグラム作成
plt.figure(figsize=(10, 5))
plt.hist(total_epochs, bins=20, edgecolor='black')
plt.xlabel('Epochs')
plt.ylabel('Frequency')
plt.title('Histogram of Epoch Counts 2')
plt.grid(True)
plt.show()

# 統計値計算
max_n, max_epoch = max(epoch_totals, key=lambda x: x[1])
min_n, min_epoch = min(epoch_totals, key=lambda x: x[1])
mean_epoch = sum(total_epochs) / len(total_epochs)
median_epoch = sorted(total_epochs)[len(total_epochs) // 2]
variance_epoch = sum((x - mean_epoch) ** 2 for x in total_epochs) / len(total_epochs)
std_deviation_epoch = variance_epoch ** 0.5

# 統計値出力
print("Statistics of Epoch Counts:")
print(f"Maximum: n={max_n}, Epochs={max_epoch}")
print(f"Minimum: n={min_n}, Epochs={min_epoch}")
print(f"Mean: {mean_epoch:.2f}")
print(f"Median: {median_epoch}")
print(f"Variance: {variance_epoch:.2f}")
print(f"Standard Deviation: {std_deviation_epoch:.2f}")
