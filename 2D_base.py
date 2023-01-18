import os
import cv2
import logging

import torch.optim.lr_scheduler
import volumentations as volaug

from IMvolDataset import *
from IMm2oDataset import *

from pm_generator.maskrcnn_pm_generator import maskrcnn_pm_generator
from pm_generator.mia_pm_generator import mia_pm_generator
from pm_generator.unet_pm_generator import unet_pm_generator
from pm_generator.uPP_pm_generator import uPP_pm_generator
from models.m2oLSTM import *
from result_compare import *

from torchvision.transforms import functional as F

from vol_eval import *

import focal_loss
import BinaryFocalLoss

def load_checkPoint(PATH, model, optimizer=None, scheduler=None, epoch=None, loss=None, only_model=True):
    checkpoint= torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    if not only_model:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return {'model':model, 'optimizer':optimizer, 'scheduler':scheduler, 'epoch':epoch, 'loss':loss}
    else:
        return model

def main(mode, gpu, exp_prefix='test', resume_exp=None, batch_size=1, fastEXP=False):


    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("***************************************")
    print("********* LSTM Training Start *********")
    print("DEVICE :",device, gpu , "& PREFIX :",exp_prefix)

    bhw = (batch_size,512,512)

    data_path = './dataset/RAW_PNG'

    pm_path = './pred_map_' + str(exp_prefix)


    threshold = 0.5

    train_transform = volaug.Compose([
        # volaug.RotatePseudo2D(axes=(2,3), limit=(-10,10), interpolation=1),
        # volaug.ElasticTransformPseudo2D(alpha=1, sigma=10, alpha_affine=10),
        volaug.Normalize(),
    ])

    test_transform = volaug.Compose([
        volaug.Normalize(),
    ])

    # [0, 14] [15, 29] [30, 44] [45, 59] (15, 15, 15, 15)\
    if fastEXP:
        test_index_list = [(45,59)]
    else:
        test_index_list = [(0, 14), (15, 29), (30, 44), (45, 59)]

    logger = logging.getLogger("tracker")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    # log_file = logging.FileHandler(save_path+'/train.log')
    # log_file.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    # logger.addHandler(log_file)

    ### test index for resume experiment
    # resume_exp = ([test_index],[start_epoch])
    if resume_exp is not None and mode == 'train':
        test_index_list = [test_index_list[i] for i in resume_exp[0]]

    if 'train' in mode:
        for i, test_index in enumerate(test_index_list):

            subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])
            # select test subject range
            test_subjects = subjects[test_index[0]:test_index[1]+1]
            subjects = [index for index in subjects if index not in test_subjects]

            # print(subjects)
            # print('train subject Num ', len(subjects))
            # print('test subject NUm ', len(test_subjects))
            epoch_2d = 40
            model_path = '%d%d_e%d_weight.pth' %(test_index[0],test_index[1],epoch_2d)
            save_path = './result/mask_%d'%(epoch_2d)
            print(model_path)
            print("="*20)

            ### maskrcnn
            if exp_prefix =='mask': # 40 fix
                maskrcnn_pm_generator(model_path,
                                      model_path='./SW_SEG_weights/maskrcnn',
                                      dir_path=pm_path, data_path=data_path, subjects=test_index,
                                      save_image=False, test_mode=True, folder_name=save_path)

            ### mia
            elif exp_prefix =='mia':
                mia_pm_generator(model_path,
                                      model_path='./SW_SEG_weights/mia',
                                      dir_path=pm_path, data_path=data_path, subjects=test_index,
                                      save_image=True, test_mode=True, folder_name=save_path)

            ### unet
            elif exp_prefix =='unet':
                unet_pm_generator(model_path,
                                      model_path='./SW_SEG_weights/unetAdam',
                                      dir_path=pm_path, data_path=data_path, subjects=test_index,
                                      save_image=True, test_mode=True, folder_name=save_path)

            ### unet++
            elif exp_prefix =='unetpp':
                uPP_pm_generator(model_path,
                                      model_path='./SW_SEG_weights/unetPP',
                                      dir_path=pm_path, data_path=data_path, subjects=test_index,
                                      save_image=True, test_mode=True, folder_name=save_path)

        subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])
        total_ol = []
        total_ja = []
        total_di = []
        total_fp = []
        total_fn = []

        for s in subjects:
            overlap, jaccard, dice, fn, fp = eval_volume_from_mask(s, pred_path=os.path.join(save_path))
            print(s + ' overlap: %.4f dice: %.4f jaccard: %.4f  fn: %.4f fp: %.4f' % (
                overlap, dice, jaccard, fn, fp))
            total_ol.append(overlap)
            total_ja.append(jaccard)
            total_di.append(dice)
            total_fn.append(fn)
            total_fp.append(fp)

        print('Average overlap: %.4f dice: %.4f jaccard: %.4f fn: %.4f fp: %.4f' % (
            np.mean(total_ol), np.mean(total_di), np.mean(total_ja), np.mean(total_fn), np.mean(total_fp)))


if __name__ == '__main__':

    resume_exp=None

    main('train', exp_prefix='mask', gpu=1, batch_size=2, resume_exp=resume_exp, fastEXP=False)

