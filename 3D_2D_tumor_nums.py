#https://blog.csdn.net/a563562675/article/details/107066836
还可以获取3D连通域对象对每个连通域进行属性获取和操作，比如计算面积、外接矩形、凸包面积等。
#https://www.cnblogs.com/zzc-Andy/p/16731034.html 
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

#2D tumor_nums+length+areas

import cv2
import numpy as np
import argparse
from matplotlib import pyplot as plt
from imutils import perspective
from imutils import contours
import imutils
font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体样式
parser = argparse.ArgumentParser(description='Calculation of length and area of irregularly divided area')
parser.add_argument('--pixelsPerMetric', default=347, help='pixelsPerMetric')
parser.add_argument('--path', default='./ISIC_0010019-mask.png', help='learning rate')
args = parser.parse_args()

def get_contour(img):
    """获取连通域
    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    x, y = img_gray.shape
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours,x,y
def midpoint(ptA, ptB):
    return ((ptA[0]+ptB[0])*0.5, (ptA[1]+ptB[1])*0.5)
def main():
    
    # 1.导入图片
    img_src = cv2.imread(args.path)#ISIC_0010019-mask.png volume-9-165-label.png
    img_result = img_src.copy()
    pixelsPerMetric = args.pixelsPerMetric
    # 2.获取连通域
    cont,x,y = get_contour(img_src)
    m=len(cont)
    for i in range(m):
        if cv2.contourArea(cont[i])<5:
            m=m-1
            continue
    print('共有%d个肿瘤'%m) 
    single_masks = np.zeros((x, y))
    for i in range(len(cont)):
        if cv2.contourArea(cont[i])<5:
            continue
        cv2.drawContours(img_result, cont[i], -1, (0, 0, 255), 2)
        # 3.获取轮廓面积
         
        fill_image = cv2.fillConvexPoly(single_masks, cont[i], 255)
        pixels = cv2.countNonZero(fill_image)#149 91
        cnt_area = cv2.contourArea(cont[i]) #计算轮廓内面积函数使用的是格林公式计算轮廓内面积的
        cnt_Length=cv2.arcLength(cont[i],True)
        rect = cv2.minAreaRect(cont[i])
        width = rect[1][0]
        height = rect[1][1]

        # box = cv2.boxPoints(rect)
        # box = np.int0(box)  # 获得矩形角点
        # # area = cv2.contourArea(box)
        
        # cv2.polylines(img_result, [box], True, (0, 255, 0), 3)
        # text1 = 'Width: ' + str(int(width)) + ' Height: ' + str(int(height))
        # text2 = 'Rect Area: ' + str(area)
        # cv2.putText(img_result, text1, (10, 30+pix), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
        # cv2.putText(img_result, text2, (10, 60+pix), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
        # pix+=60
        box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
        box = np.array(box, dtype="int")   
        # 对轮廓点进行排序，顺序为左上，右上，右下和左下
        # 然后绘制旋转边界框的轮廓
        box = perspective.order_points(box)
        cv2.drawContours(img_result, [box.astype("int")], -1, (0,255,0), 2)
        # 遍历原始点并绘制出来
        for (x, y) in box:
            cv2.circle(img_result, (int(x), int(y)), 5, (0,0,255), -1)
    # 打开有序的边界框，然后计算左上和右上坐标之间的中点，
        # 再计算左下和右下坐标之间的中点
        (tl, tr, br, bl) = box
        (tltrX, tltrY) = midpoint(tl, tr)
        (blbrX, blbrY) = midpoint(bl, br)
        # 计算左上点和右上点之间的中点
        # 然后是右上角和右下角之间的中点
        (tlblX, tlblY) = midpoint(tl, bl)
        (trbrX, trbrY) = midpoint(tr, br)
        #每英寸所占的像素点
        # if pixelsPerMetric is None:
        #     pixelsPerMetric = width/0.972  #width
        # 计算物体的大小
        dimwid = width / pixelsPerMetric
        dimhei = height / pixelsPerMetric
        # 4.计算等效直径
        equi_diameter = np.sqrt(4 * cnt_area / np.pi)
        real_area = cnt_area / (pixelsPerMetric*pixelsPerMetric)

        cv2.putText(img_result, "{:.1f}in".format(dimwid),
        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
        cv2.putText(img_result, "{:.1f}in".format(dimhei),
        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 255, 255), 2)
        cv2.putText(img_result, "{:.1f}sq.in".format(real_area),
        (int(rect[0][0] + 10), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (255, 0, 0), 2)
        print("非零像素数=", pixels)
        print("轮廓周长=", cnt_Length)
        print("等效直径=%.4f" % equi_diameter)
        print("轮廓面积=", cnt_area)
        print("实际面积=%.4fsq.in"%real_area)
        print("轮廓长度=%.4f,轮廓宽度=%.4f"% (height,width))
        print("实际长度=%.4fin,实际宽度=%.4fin"% (dimhei,dimwid))
    plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)), plt.title('Rectangle')
    plt.show()
if __name__ == '__main__':
    main()
#448*448
# 共有2个肿瘤
# 第1个肿瘤
# 非零像素数= 149
# 轮廓面积= 129.0
# 等效直径=12.8159
# 第2个肿瘤
# 非零像素数= 91
# 轮廓面积= 75.0
# 等效直径=9.7721
