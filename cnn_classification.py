# -*- coding: utf-8 -*-

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from keras.layers import Input, Conv1D, MaxPooling1D, Conv1DTranspose
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.utils.class_weight import compute_sample_weight
from keras.layers import Dropout
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import plot_model
# 他のimport文もそのまま

# CSVファイルからデータを読み込む部分もそのまま

# 以降の部分もそのまま
output_directory = "padded_data_for_1dcnn_length_1024"

# CSVファイルからデータを読み込む
csv_filename = os.path.join(output_directory, 'padded_one_hot_encoding_all_1.csv')
labels = pd.read_csv(csv_filename)

# Sample ID, Time Step, およびクラスの列を取得
sample_ids = labels['Sample ID'].values
time_steps = labels['Time Step'].values
class_columns = [f'Class_{i}' for i in range(3)]
class_labels = labels[class_columns].values

# オリジナルの形式に戻すためのデータを作成
num_samples = len(np.unique(sample_ids))
num_time_steps = len(np.unique(time_steps))
num_classes = class_labels.shape[1]

# 元の形状に戻す
labels = class_labels.reshape(num_samples, num_time_steps, num_classes)

print('labels shape:', labels.shape)


data_filename = os.path.join(output_directory, "padded_features_all_1.csv")
data_column_names = pd.read_csv(data_filename, nrows=0).columns[2:]
num_features = len(data_column_names)
data = pd.read_csv(data_filename, usecols=[2, 3, 4], header=None, skiprows=1, dtype=float)
data = data.rename(columns={2: data_column_names[0], 3: data_column_names[1], 4: data_column_names[2]})
data = data.values.reshape(num_samples, num_time_steps, num_features)

print('data shape:', data.shape)
print("num_samples:", num_samples)
print("num_time_steps:", num_time_steps)
print("num_features:", num_features)
print("num_classes:", num_classes)
# データの前処理
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=1)




'''
# クラスごとのサンプル数を計算
class_sample_counts = np.sum(y_train, axis=(0, 1))  # 各クラスの総サンプル数

print("Class Sample Counts:", class_sample_counts)

# サンプルごとの重みを計算
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train.reshape(-1, num_classes))

print("Sample Weights length:", len(sample_weights))

# サンプルごとの重みを元の形状に戻す
sample_weights = sample_weights.reshape(y_train.shape[:2])

print("Sample Weights:", sample_weights)

# クラスごとの重みを計算
class_weights = np.sum(sample_weights, axis=0)  # axis 0 で合計します

print("Class Weights:", class_weights)

# クラスごとの重みを辞書にまとめる
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

print("Class Weight Dictionary:", class_weight_dict)
'''




# モデルの構築
input_shape = (num_time_steps, num_features)
input_layer = Input(shape=input_shape)

# Convolutional layers
conv1 = Conv1D(64, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(input_layer)
conv1 = Dropout(0.2)(conv1)  # Add Dropout here
conv2 = Conv1D(128, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv1)
conv2 = Dropout(0.2)(conv2)  # Add Dropout here
pool1 = MaxPooling1D(pool_size=2)(conv2)
conv3 = Conv1D(256, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(pool1)
conv3 = Dropout(0.2)(conv3)  # Add Dropout here
conv4 = Conv1D(256, 5, padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv3)
conv4 = Dropout(0.2)(conv4)  # Add Dropout here
pool2 = MaxPooling1D(pool_size=2)(conv4)

# Transposed convolutional layers (Upsampling)
up1 = Conv1DTranspose(256, 3, strides=2, padding='same', activation='relu')(pool2)
up2 = Conv1DTranspose(128, 3, strides=2, padding='same', activation='relu')(up1)

# Adjust the output Conv1D layer to produce (None, 1062, 3) shape
output_layer = Conv1D(3, 1, padding='same', activation='softmax')(up2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# Train the model
batch_size = 16
epochs = 30

# モデルのサマリーを表示
model.summary()




# Compile the model with class weights
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))

# ... (rest of the code)





# 訓練履歴のプロット
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# テストセットでモデルを評価
evaluation = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

plot_model(model, to_file='temp.png', show_shapes=True, show_layer_names=True, rankdir='TB')

# テストセットの予測を行い、クラスごとの予測確率を取得
y_pred = model.predict(X_test)



# Calculate confusion matrix
y_pred_classes = np.argmax(y_pred, axis=-1)
y_true_classes = np.argmax(y_test, axis=-1)

print(y_pred_classes)
print(y_true_classes)


conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Accuracy を計算
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

# Precision, Recall, Specificity, F1-score を計算
precision = np.zeros(num_classes, dtype=float)
recall = np.zeros(num_classes, dtype=float)
specificity = np.zeros(num_classes, dtype=float)
f1_score = np.zeros(num_classes, dtype=float)

for i in range(num_classes):
    true_positives = conf_matrix[i, i]
    false_positives = np.sum(conf_matrix[:, i]) - true_positives
    false_negatives = np.sum(conf_matrix[i, :]) - true_positives
    true_negatives = np.sum(conf_matrix) - (true_positives + false_positives + false_negatives)
    
    precision[i] = true_positives / (true_positives + false_positives)
    recall[i] = true_positives / (true_positives + false_negatives)
    specificity[i] = true_negatives / (true_negatives + false_positives)
    f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

# 結果を出力
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)
print("F1-score:", f1_score)

