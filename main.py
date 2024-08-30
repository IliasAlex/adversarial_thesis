from datasets.datasets import UrbanSound8KDataset, get_data_loaders

data_csv='/data/urbansound8k/UrbanSound8K.csv'
root_dir='/data/urbansound8k'
train_fold=[1,2,3,4,5,6,7,8]
val_fold=[9]
test_fold=[10]

train_loader, val_loader, test_loader = get_data_loaders(data_csv, root_dir, train_fold, val_fold, test_fold, batch_size=32)