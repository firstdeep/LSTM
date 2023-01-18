from vol_eval import *

def eval_folder(path):
    save_path = path
    subjects = sorted([name for name in os.listdir('./dataset/ROI_pos') if name.endswith('_1')])
    total_ol = []
    total_ja = []
    total_di = []
    total_fp = []
    total_fn = []

    file = open(os.path.join(path,'result.txt'), 'w')  # hello.txt 파일을 쓰기 모드(w)로 열기. 파일 객체 반환

    for s in subjects:
        overlap, jaccard, dice, fn, fp = eval_volume_from_mask(s, pred_path=os.path.join(save_path))
        file.write(s + ' overlap: %.4f dice: %.4f jaccard: %.4f  fn: %.4f fp: %.4f\n' % (
            overlap, dice, jaccard, fn, fp))

        total_ol.append(overlap)
        total_ja.append(jaccard)
        total_di.append(dice)
        total_fn.append(fn)
        total_fp.append(fp)

    file.write('\nAverage overlap: %.4f dice: %.4f jaccard: %.4f fn: %.4f fp: %.4f' % (
        np.mean(total_ol), np.mean(total_di), np.mean(total_ja), np.mean(total_fn), np.mean(total_fp)))

    file.close()

eval_folder('/home/gpuadmin/aaa_3d/pred_result/pred_unet_linear')