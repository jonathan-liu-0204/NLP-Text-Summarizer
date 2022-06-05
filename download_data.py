import os
import datasets
import pandas as pd

dataset = datasets.load_dataset("cnn_dailymail", '3.0.0')

dataset_path = os.path.join(".", "cnn_daily_ds")
if os.path.exists(dataset_path):
    raise Exception(f"{dataset_path} exists")
else:
    os.mkdir(dataset_path)

print("Convert train dataset to csv")
train_data = dataset['train']
train_data.to_csv(os.path.join(dataset_path, "train.csv"))
train_data = pd.read_csv(os.path.join(dataset_path, "train.csv"))
print(f"Only save 50000/{len(train_data)} training samples")
train_data = train_data.iloc[:50000]
train_data.to_csv(os.path.join(dataset_path, "train.csv"))

print("Convert validation dataset to csv")
valid_data = dataset['validation']
valid_data.to_csv(os.path.join(dataset_path, "valid.csv"))

print("Convert test dataset to csv")
test_data = dataset['test']
test_data.to_csv(os.path.join(dataset_path, "test.csv"))