import torch.nn

from models.senosr3D import Senosr3D
from models.spiderUnet import SpiderUnet
from models.unet3d_model import UNet3D
from IMvolDataset import *
from IMm2oDataset import *
from vol_eval import *
from tqdm import tqdm
from datetime import datetime
import cv2, os, random
import volumentations as volaug
import shutil
from trainer_3D import *
from utils_3D import *
from eval_only import *

if __name__ == '__main__':
    print("Start")
    # Load config
    config_path = os.path.join('./train.yaml')
    config = load_yaml(config_path)
    print("Start")
    # Set random seed, deterministic
    torch.cuda.manual_seed(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set device(GPU/CPU)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu_num'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create train result directory and set logger
    if config['mode'] == 'train' and not config['checkpoint']:
        train_serial = datetime.now().strftime("%Y%m%d_%H%M%S")
        train_result_dir = os.path.join(config['save_folder'], 'train', train_serial)

        os.makedirs(train_result_dir, exist_ok=True)

        # config file save
        shutil.copy(config_path, os.path.join(train_result_dir, 'train.yaml'))

    # Data loader
    train_transform = volaug.Compose([
        volaug.ElasticTransformPseudo2D(alpha=1, sigma=10, alpha_affine=10),
        # volaug.RotatePseudo2D(axes=(2,3), limit=(-10,10), interpolation=1),
        # volaug.Normalize(),
    ])

    test_transform = volaug.Compose([
        volaug.Normalize(),
    ])

    test_index_list = [(0, 14), (15, 29), (30, 44), (45, 59)]

    for fold, test_index in enumerate(test_index_list):
        print("\n*** Fold: %d ***"%fold)
        subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])
        test_subjects = subjects[test_index[0]:test_index[1] + 1]
        train_subjects = [index for index in subjects if index not in test_subjects]

        if config['prefix'] == 'spiderUnet' or config['prefix'] == '3DUnet':
            data_path = './dataset/FC_pos'
            dataset = IMDataset(data_path, max_slice=config["slice"], pm_path=None)

        elif config['prefix']  == 'i_sensor3D':
            data_path = './dataset/RAW_PNG'
            dataset = IMm2oDataset(data_path, m_length=config["slice"], pm_path=None)

        train_index = []
        test_index = []

        for subject in train_subjects:
            train_index.extend(dataset.img_subject_vol_index[subject])

        for subject in test_subjects:
            test_index.extend(dataset.img_subject_vol_index[subject])

        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)

        train_dataset.dataset.transform = train_transform
        test_dataset.dataset.transform = test_transform

        # define dataloader
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True,
            num_workers=config['num_workers'], collate_fn=collate_fn_pad_z, pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False,
            num_workers=config['num_workers'], collate_fn=collate_fn_pad_z, pin_memory=True)

        # Load model
        if config['prefix'] == 'spiderUnet':
            bhw = (config['batch_size'], 512, 512)
            model = SpiderUnet(n_channels=1, n_classes=1, b_h_w=bhw)
        elif config['prefix'] == 'sensor3D':
            bhw = (config['batch_size'], 512, 512)
            model = Senosr3D(n_channels=1, n_classes=1, b_h_w=bhw, m_length=slice)
        elif config['prefix'] == '3DUnet':
            model = UNet3D(n_channels=1, n_classes=1)

        model.to(device)
        torch.cuda.empty_cache()

        if config['mode'] == 'train':
            # Set optimizer
            optimizer = get_optimizer(optimizer_str=config['optimizer']['name'])
            optimizer = optimizer(model.parameters(), **config['optimizer']['args'])

            # Set Scheduler
            scheduler = get_scheduler(scheduler_str=config['scheduler']['name'])
            scheduler = scheduler(optimizer=optimizer, **config['scheduler']['args'])

            # Set loss function
            loss_func = get_loss_function(loss_function_str=config['loss']['name'])

            start_epoch = 0
            # if fold != 2:
            #     continue
            if config['checkpoint']:
                train_result_dir = os.path.join(config['save_folder'], 'train', config['checkpoint_path'])
                check_point_path = os.path.join(config['save_folder'], 'train', config['checkpoint_path'], '%d_model_%d.pt'%(fold,config['checkpoint_epoch']))
                check_point = torch.load(check_point_path)
                model.load_state_dict(check_point['model'])
                optimizer = check_point['optimizer']
                scheduler = check_point['scheduler']
                start_epoch = check_point['epoch']
                print("*** Load model Success ***")

            for epoch_id in range(start_epoch, config['n_epochs']):
                loss_total = 0
                for i, (volume, target) in enumerate(tqdm(train_data_loader)):
                    model.train()

                    volume = volume.to(device)
                    target = target.to(device)

                    if config['prefix'] == '3DUnet':
                        volume = volume.permute(0, 2, 1, 3, 4) # b t c h w => b c t h w
                        target = target.permute(0, 2, 1, 3, 4)
                        print(volume.shape)
                        pred = model(volume)

                    elif config['prefix'] == 'sensor3D' or config['prefix'] == 'spiderUnet':
                        pred = model(volume, device=device)

                    pred = pred.squeeze()
                    target = target.squeeze()

                    loss = loss_func(pred, target)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    loss_total += loss.item()

                    del volume, target

                print("\t=== Epoch: %d / Loss: %f ==="%(epoch_id,loss_total/len(train_data_loader)))

                row = dict()
                row['train_loss'] = loss_total/len(train_data_loader)
                row['epoch_id'] = epoch_id
                row['lr'] = scheduler.get_last_lr()

                add_row(train_result_dir, fold, row)
                save_plot(save_path=train_result_dir,fold=fold,
                          csv_path=os.path.join(train_result_dir, 'record_%d.csv'%(fold)), plots=config['plot'])

                if epoch_id in config['save_epoch']:
                    save_checkPoint(
                        PATH=os.path.join(train_result_dir, '%d_model_%d.pt'%(fold,epoch_id)),
                        model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch_id, loss=loss_total)

                scheduler.step()
                torch.cuda.empty_cache()

        if config['mode'] == 'test':
            check_point_path = os.path.join(config['save_folder'], 'train', config['checkpoint_path'],
                                            '%d_model_%d.pt' % (fold, config['checkpoint_epoch']))
            check_point = torch.load(check_point_path)
            model.load_state_dict(check_point['model'])
            print(check_point_path)
            sigmoid = torch.nn.Sigmoid()
            for idx in test_index:
                model.eval()
                with torch.no_grad():
                    volume, target = dataset[idx]
                    volume = volume.unsqueeze(0)
                    volume = volume.to(device)

                    # sensor3D, spider
                    if config['prefix'] == 'sensor3D' or config['prefix'] == 'spiderUnet':
                        prediction = model(volume, device=device)

                    elif config['prefix'] == '3DUnet':
                        volume = volume.permute(0, 2, 1, 3, 4)
                        prediction = model(volume)
                        prediction = prediction.permute(0, 2, 1, 3, 4)

                    prediction = prediction[-1]

                prediction = prediction.squeeze(0)
                prediction = sigmoid(prediction)

                for i in range(len(prediction)):
                    path = dataset.mask_volumes[idx]

                    path = dataset.img_volumes[idx][i]

                    write_path = os.path.join(config['save_folder'], 'train',
                                              config['checkpoint_path'],
                                              'predict_%d'%config['checkpoint_epoch'])
                    write_path_folder =os.path.join(write_path, path.split('/')[3])

                    if not os.path.isdir(write_path):
                        os.mkdir(write_path)
                    if not os.path.isdir(write_path_folder):
                        os.mkdir(write_path_folder)

                    write_name = os.path.join(write_path_folder, 'pred_' + path.split('/')[-1])

                    mask = prediction[i].cpu().numpy()
                    mask = mask.squeeze(0)  # c

                    mask = np.where(mask > 0.5, 1.0, 0.0)
                    mask *= 255
                    mask = mask.astype(np.uint8)
                    img_mask = np.array(mask)
                    cv2.imwrite(write_name, img_mask)

    if config['mode'] == 'test':
        eval_folder(os.path.join(config['save_folder'], 'train', config['checkpoint_path'],
                                 'predict_%d'%config['checkpoint_epoch']))
