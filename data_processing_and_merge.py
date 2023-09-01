# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import csv


# ディレクトリの作成
output_directory = 'data_for_1dcnn'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# ディレクトリ内のCSVファイルを処理
for n in range(1, 101):
    file_name = f'{n}_1.csv'

    # CSVファイルからデータを読み込む
    df = pd.read_csv(file_name)

    # Sample IDの取得
    sample_id = n

    # Stage列の値を置換
    stage_mapping = {
        'W': 0,
        'N1': 1,
        'N2': 1,
        'N3': 1,
        'R': 2
    }
    df['Stage'] = df['Stage'].replace(stage_mapping)

    # クラスをStage_0, Stage_1, Stage_2でエンコーディング
    df_encoded = pd.get_dummies(df['Stage'], prefix='Stage')
    stage_columns = ['Stage_0', 'Stage_1', 'Stage_2']
    
    # もしエンコードされたカラムが不足している場合、足りないカラムを追加して0で埋める
    for col in stage_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df['Sample ID'] = sample_id

    # 新しい行のデータを作成
    df_one_hot_encoding = pd.concat([df['Sample ID'], df['Epoch'], df_encoded], axis=1)
    df_features = pd.concat([df['Sample ID'], df['Epoch'], df[['SpO2', 'HR','Light']]], axis=1)

    # 新しいDataFrameをCSVファイルとして保存
    one_hot_encoding_filename = os.path.join(output_directory, f'one_hot_encoding_{n}_1.csv')
    df_one_hot_encoding.to_csv(one_hot_encoding_filename, index=False)

    features_filename = os.path.join(output_directory, f'features_{n}_1.csv')
    df_features.to_csv(features_filename, index=False)


# パディングするターゲットの長さ
target_length = 1024

input_directory = "data_for_1dcnn"

# ディレクトリを指定
output_directory = f'padded_data_for_1dcnn_length_{target_length}'
os.makedirs(output_directory, exist_ok=True)





# Function to pad data
def pad_data(data, target_length):
    padded_data = []
    for sample in data:
        if sample.shape[0] < target_length:
            padded_sample = np.pad(sample, ((0, target_length - sample.shape[0]), (0, 0)), mode='constant')
        else:
            padded_sample = sample[:target_length]
        padded_data.append(padded_sample)
    return np.array(padded_data)

# Function to pad labels
def pad_labels(labels, target_length, num_classes):
    padded_labels = np.zeros((num_samples, target_length, num_classes))
    for i, label_seq in enumerate(labels):
        if label_seq.shape[0] >= target_length:
            padded_labels[i, :, :] = label_seq[:target_length]
        else:
            padding_length = target_length - label_seq.shape[0]
            padding = np.tile(label_seq[-1], (padding_length, 1))
            padded_label_seq = np.vstack((label_seq, padding))
            padded_labels[i, :, :] = padded_label_seq
    return padded_labels

for n in range(1, 101):
    # CSVファイルからデータを読み込む
    csv_filename = os.path.join(input_directory, f'one_hot_encoding_{n}_1.csv')
    labels = pd.read_csv(csv_filename)

    # Sample ID, Time Step, およびクラスの列を取得
    sample_ids = labels['Sample ID'].values
    time_steps = labels['Epoch'].values
    class_columns = [f'Stage_{i}' for i in range(3)]
    class_labels = labels[class_columns].values

    # オリジナルの形式に戻すためのデータを作成
    num_samples = len(np.unique(sample_ids))
    num_time_steps = len(np.unique(time_steps))
    num_classes = class_labels.shape[1]

    # 元の形状に戻す
    labels = class_labels.reshape(num_samples, num_time_steps, num_classes)

    data_filename = os.path.join(input_directory, f"features_{n}_1.csv")
    data_column_names = pd.read_csv(data_filename, nrows=0).columns[2:]
    num_features = len(data_column_names)

    data = pd.read_csv(data_filename, usecols=[2, 3, 4], header=None, skiprows=1, dtype=float)
    data = data.rename(columns={2: data_column_names[0], 3: data_column_names[1], 4: data_column_names[2]})
    data = data.values.reshape(num_samples, num_time_steps, num_features)
    
    # パディング
    padded_data = pad_data(data, target_length)
    
    print(f'padded_data shape for {n}:', padded_data.shape)

    

    # CSVファイルにデータを書き込む
    data_filename = os.path.join(output_directory, f"padded_features_{n}_1.csv")
    with open(data_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Sample ID", "Time Step"] + data_column_names.tolist()  # カラム名を追加
        writer.writerow(header)

        for sample_idx, sample_data in enumerate(padded_data):
            for time_idx, features in enumerate(sample_data):
                writer.writerow([n, time_idx + 1] + features.tolist())

    print(f'Padded data written to {data_filename}')

    # パディングされたラベルを作成
    padded_labels = pad_labels(labels, target_length, num_classes)
    
    print(f'padded_labels shape for {n}:', padded_labels.shape)

    
    # CSVファイルにラベルを書き込む
    labels_filename = os.path.join(output_directory, f"padded_one_hot_encoding_{n}_1.csv")
    with open(labels_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Sample ID", "Time Step"] + [f"Class_{i}" for i in range(num_classes)]
        writer.writerow(header)

        for sample_idx, sample_labels in enumerate(padded_labels):
            for time_idx, label_row in enumerate(sample_labels):
                row = [n, time_idx + 1] + label_row.tolist()
                writer.writerow(row)

    print(f'Padded labels written to {labels_filename}')

print("All files processed.")

# データを結合する関数
def combine_data(files_prefix, num_files, output_filename):
    combined_data = None
    combined_labels = None

    for n in range(1, num_files + 1):
        # 読み込むファイル名を作成
        data_filename = os.path.join(output_directory, f"{files_prefix}_{n}_1.csv")
        labels_filename = os.path.join(output_directory, f"{files_prefix}_{n}_1.csv")

        # データを読み込む
        data = pd.read_csv(data_filename)
        labels = pd.read_csv(labels_filename)

        # データを結合
        if combined_data is None:
            combined_data = data
        else:
            combined_data = pd.concat([combined_data, data], ignore_index=True)

        # ラベルを結合
        if combined_labels is None:
            combined_labels = labels
        else:
            combined_labels = pd.concat([combined_labels, labels], ignore_index=True)

    # 結合したデータをCSVファイルに書き込む
    combined_data_filename = os.path.join(output_directory, f"{output_filename}.csv")
    combined_labels_filename = os.path.join(output_directory, f"{output_filename}.csv")

    combined_data.to_csv(combined_data_filename, index=False)
    
    combined_labels.to_csv(combined_labels_filename, index=False)
    

    print(f'Combined data written to {combined_data_filename}')
    print(f'Combined labels written to {combined_labels_filename}')

# データを結合して保存
combine_data("padded_features", 100, "padded_features_all_1")
combine_data("padded_one_hot_encoding", 100, "padded_one_hot_encoding_all_1")

print("All files combined and processed.")

