import os, cv2
import numpy as np
from tqdm import tqdm

start_path = 'pred_map_unetpp'
dst_path = 'pred_result/pred_unetpp'
list_folder = os.listdir(start_path)
print(list_folder)

for i in list_folder:
    if not os.path.isdir(os.path.join(dst_path, i)):
        os.mkdir(os.path.join(dst_path, i))
    for j in tqdm(os.listdir(os.path.join(start_path, i))):
        data = np.load(os.path.join(start_path, i,j))
        data *= 255
        data = data.astype(np.uint8)
        data[data > 127] = 255
        data[data <= 127] = 0

        file_name =  j[:-4]
        cv2.imwrite(os.path.join(dst_path, i, file_name), data)

