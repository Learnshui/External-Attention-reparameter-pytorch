# 获取肝脏前后扩充5个切片的数据 
import glob
import os
import numpy as np
import nibabel as nib
pre_data_result='/home/datacenter/ssd2/data_s/Lits_cut'
data_path='/home/datacenter/ssd1/datasets/LiTS/'
# savenpy(0, data_path, pre_data_result, 'train')

def read_image(img_path):
    data = nib.load(img_path)
    hdr = data.header
    # convert to numpy
    data = data.get_data()#(512, 512, 244)
    return data, hdr
def savenpy(data_path, pre_data_result, train_val_test):
    filelist = glob.glob(os.path.join(data_path, 'volume-*.nii'))
    print('start processing %d' % (len(filelist)))
    for id in range(4):
        name = os.path.splitext(os.path.split(filelist[id])[1])[0]
        # volume_id = int(name.split('-')[-1])

        pre_data_result_pro = os.path.join(pre_data_result, name)
        if not os.path.exists(pre_data_result_pro):
            os.makedirs(pre_data_result_pro)

        img, hdr = read_image(filelist[id]) #(512, 512, 549)
        #防止超出
        img = np.pad(img, [[0, 0], [0, 0], [5, 5]], 'constant')
        
        if train_val_test == 'train' or train_val_test == 'val':
            label, _ = read_image(filelist[id].replace('volume', 'segmentation'))#(512, 512, 244)
            label = np.pad(label, [[0, 0], [0, 0], [5, 5]], 'constant')
            if np.sum(label > 0):
                _, _, liver_zz = np.where(label > 0) #(2301071,)
                min_liver_z = np.min(liver_zz) #343
                max_liver_z = np.max(liver_zz) #515
                # liver_range = [min_liver_z-5, max_liver_z+5]
            if np.sum(label == 2):
                _, _, tumor_zz = np.where(label == 2)#(22179,)
                min_tumor_z = np.min(tumor_zz) #371
                max_tumor_z = np.max(tumor_zz)#510
                # tumor_range = [min_tumor_z, max_tumor_z]
            else:
                tumor_range = [None, None]
        if train_val_test == 'train' or train_val_test == 'val':

            for i in range(min_liver_z-5, max_liver_z+6):
                save_path = os.path.join(pre_data_result_pro, name + '_%d_clean.npy' % i)
                img_slice = img[:, :, i]
                #i：i+5
                np.save(save_path, img_slice.astype(np.float16))
                label_slice = label[:, :, i]
                np.save(save_path.replace('clean', 'label'), label_slice.astype(np.uint8))

        print('end processing %s \t %d/%d' % (filelist[id], id + 1, len(filelist)))
if __name__ == '__main__':
    savenpy(data_path, pre_data_result, 'train')  
    
