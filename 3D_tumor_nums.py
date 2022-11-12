from skimage import measure
import SimpleITK as sitk
def connected_component(image):
    # 标记输入的3D图像
    image = sitk.ReadImage(image, sitk.sitkUInt8)
    _input = sitk.GetArrayFromImage(image)
    #以医院给的一个数据为例（caolongfang）两种肿瘤类型标签为4,5，肝脏标签为8
    #1、肿瘤有两个
    # _input[_input==4]=1
    # _input[_input==5]=1
    _input[_input==8]=0
    #2、肿瘤有一个
    # _input[_input==4]=1
    # _input[_input==5]=0
    # _input[_input==8]=0
    #标签不做处理时，输出值为3=肿瘤两个+肝脏一个
    label, num = measure.label(_input, connectivity=1, return_num=True)
    print('tumor numbers:%d '% num)
    # print('共有%d个肿瘤'%m) 
    # if num < 1:
    #     return image
        
	# # 获取对应的region对象
    # region = measure.regionprops(label)
    # # 获取每一块区域面积并排序
    # num_list = [i for i in range(1, num+1)]
    # area_list = [region[i-1].area for i in num_list]
    # num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]
	# # 去除面积较小的连通域
    # if len(num_list_sorted) > 3:
    #     # for i in range(3, len(num_list_sorted)):
    #     for i in num_list_sorted[3:]:
    #         # label[label==i] = 0
    #         label[region[i-1].slice][region[i-1].image] = 0
    #     num_list_sorted = num_list_sorted[:3]
    return label
if __name__ == "__main__":
     connected_component('/Users/shuiyuanyuan/Documents/CaoLongfang/Cao Longfang.nii')
     #('/Users/shuiyuanyuan/home/caolongfang.nii')#('/Users/shuiyuanyuan/Desktop/CaoLongfang/caolongfang.nii')
     #('/Users/shuiyuanyuan/home/MICCAI-LITS2017-master/train/label/segmentation-120.nii')#
