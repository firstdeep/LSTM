from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import torch.optim as optim
import yaml, torch, os, csv
import pandas as pd

def load_yaml(path):

	with open(path, 'r') as f:
		return yaml.load(f, Loader=yaml.FullLoader)


def get_optimizer(optimizer_str: str) -> 'optimizer':
    if optimizer_str == 'SGD':

        optimizer = optim.SGD

    elif optimizer_str == 'Adam':

        optimizer = optim.Adam

    elif optimizer_str == 'AdamW':

        optimizer = optim.AdamW

    return optimizer


def get_scheduler(scheduler_str) -> object:
    scheduler = None

    if scheduler_str == 'CosineAnnealingLR':

        scheduler = lr_scheduler.CosineAnnealingLR

    elif scheduler_str == 'CosineAnnealingWarmRestarts':

        scheduler = lr_scheduler.CosineAnnealingWarmRestarts

    elif scheduler_str == 'Exponential':

        scheduler = lr_scheduler.ExponentialLR

    elif scheduler_str == 'MultiStepLR':

        scheduler = lr_scheduler.MultiStepLR

    return scheduler

def get_loss_function(loss_function_str: str):

    if loss_function_str == 'BCE':
        return torch.nn.BCEWithLogitsLoss()

def save_checkPoint(PATH, model, optimizer, scheduler, epoch, loss):
    torch.save({
        'epoch':epoch+1,
        'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),
        'scheduler':scheduler.state_dict(),
        'loss':loss,
    },PATH)

def save_plot(save_path, fold, csv_path,  plots: list):

    record_df = pd.read_csv(csv_path)
    current_epoch = record_df['epoch_id'].max()
    epoch_range = list(range(0, current_epoch + 1))
    color_list = ['red', 'blue']

    for plot_name in plots:
        # columns = [f'train_{plot_name}', f'val_{plot_name}']

        fig = plt.figure(figsize=(20, 8))

        values = record_df['train_loss'].tolist()
        plt.plot(epoch_range, values, marker='.', c=color_list[1], label='loss')

        plt.title(plot_name, fontsize=15)
        plt.legend(loc='upper right')
        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel(plot_name)
        plt.xticks(epoch_range, [str(i) for i in epoch_range])
        plt.close(fig)
        fig.savefig(os.path.join(save_path, plot_name + '_%d.png'%fold))

def add_row(save_path,fold, row_dict: dict):

    save_path = os.path.join(save_path, 'record_%d.csv'%fold)
    fieldnames = list(row_dict.keys())

    with open(save_path, newline='', mode='a') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if f.tell() == 0:
            writer.writeheader()

        writer.writerow(row_dict)
