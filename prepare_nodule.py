#-*-coding:	utf-8 -*-
'''
@DateTime:	2019.07.16	09.24.47
@Author:	dalei
@Email:	wenlei.wang@wowjoy.cn
'''

import os
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage
from skimage import filters
from skimage import measure
from skimage import morphology
from skimage import segmentation
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.morphology import binary_erosion
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.morphology import generate_binary_structure

from multiprocessing import Pool
import matplotlib.pyplot as plt

import pdb

#动脉硬化或者淋巴结钙化利用纵膈窗观察，结节或者索条利用肺窗观察
#肺窗：WW[1300,1700]HU、WL[-600,-800]HU


def Resample_ITK_Image(itk_image, old_spacing, new_spacing, is_label=False):
    original_size = itk_image.GetSize()
    out_size = [int(np.round(original_size[0]*(old_spacing[0]/new_spacing[0]))),
                int(np.round(original_size[1]*(old_spacing[1]/new_spacing[1]))),
                int(np.round(original_size[2]*(old_spacing[2]/new_spacing[2])))]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())

    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(itk_image)

def Create_Non_Mask(image):
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size/2+0.5, image_size/2-0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2+y**2)**0.5
    nan_mask = (d<image_size/2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    return nan_mask

def Binarize_Scan(image, spacing, sigma=1, HU_th=-600, Area_th=60, Eccen_th=0.99, BG_th=10):

    # step1: prepare a mask, with all corner values set to nan
    nan_mask = Create_Non_Mask(image)
    bw = np.zeros(image.shape, dtype=bool)

    #step2: iters binarize all slice
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice before Gaussian filtering
        if len(np.unique(image[i, 0:BG_th, 0:BG_th])) == 1:
            current_bw = gaussian_filter(np.multiply(image[i], nan_mask), sigma, truncate=2.0) < HU_th
        else:
            current_bw = gaussian_filter(image[i], sigma, truncate=2.0) < HU_th
        
        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if prop.area * spacing[1] * spacing[2] > Area_th and prop.eccentricity < Eccen_th:
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw
    return bw

def Analysis_Scan(bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e2, dist_th=62):
    
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)

    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1-cut_num, 0, 0], label[-1-cut_num, 0, -1], label[-1-cut_num, -1, 0], label[-1-cut_num, -1, -1], \
                    label[0, 0, mid], label[0, -1, mid], label[-1-cut_num, 0, mid], label[-1-cut_num, -1, mid]])
    for l in bg_label:
        label[label == l] = 0
        
    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if prop.area * spacing.prod() < vol_limit[0] * 1e6 or prop.area * spacing.prod() > vol_limit[1] * 1e6:
            label[label == prop.label] = 0
            
    # prepare a distance map for further analysis
    x_axis = np.linspace(-label.shape[1]/2+0.5, label.shape[1]/2-0.5, label.shape[1]) * spacing[1]
    y_axis = np.linspace(-label.shape[2]/2+0.5, label.shape[2]/2-0.5, label.shape[2]) * spacing[2]
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2+y**2)**0.5
    vols = measure.regionprops(label)
    valid_label = set()
    
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(single_vol[i] * d + (1 - single_vol[i]) * np.max(d))
        
        if np.average([min_distance[i] for i in range(label.shape[0]) if slice_area[i] > area_th]) < dist_th:
            valid_label.add(vol.label)    
    bw = np.in1d(label, list(valid_label)).reshape(label.shape)
    
    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for l in l_list:
            indices = np.nonzero(label==l)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)
    
    return bw, len(valid_label)

def Fill_Hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set([label[0, 0, 0], label[0, 0, -1], label[0, -1, 0], label[0, -1, -1], \
                    label[-1, 0, 0], label[-1, 0, -1], label[-1, -1, 0], label[-1, -1, -1]])
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)
    
    return bw

def Extract_Main(bw, cover=0.95):
    for i in range(bw.shape[0]):
        current_slice = bw[i]
        label = measure.label(current_slice)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        area = [prop.area for prop in properties]
        count = 0
        sum = 0
        while sum < np.sum(area)*cover:
            sum = sum+area[count]
            count = count+1
        filter = np.zeros(current_slice.shape, dtype=bool)
        for j in range(count):
            bb = properties[j].bbox
            filter[bb[0]:bb[2], bb[1]:bb[3]] = filter[bb[0]:bb[2], bb[1]:bb[3]] | properties[j].convex_image
        bw[i] = bw[i] & filter
        
    label = measure.label(bw)
    properties = measure.regionprops(label)
    properties.sort(key=lambda x: x.area, reverse=True)
    bw = label==properties[0].label

    return bw

def Fill_2d_Hole(bw):
    for i in range(bw.shape[0]):
        current_slice = bw[i]
        label = measure.label(current_slice)
        properties = measure.regionprops(label)
        for prop in properties:
            bb = prop.bbox
            current_slice[bb[0]:bb[2], bb[1]:bb[3]] = current_slice[bb[0]:bb[2], bb[1]:bb[3]] | prop.filled_image
        bw[i] = current_slice
    return bw

def Keep_Two_lung(bw, spacing, max_iter=22, max_ratio=4.8):    

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area/properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = binary_erosion(bw)
            iter_count = iter_count + 1
    
    if found_flag:
        d1 = distance_transform_edt(bw1 == False, sampling=spacing)
        d2 = distance_transform_edt(bw2 == False, sampling=spacing)
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)
                
        bw1 = Extract_Main(bw1)
        bw2 = Extract_Main(bw2)
        
    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype('bool')
        
    bw1 = Fill_2d_Hole(bw1)
    bw2 = Fill_2d_Hole(bw2)

    return bw1, bw2

def Dilated_Mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = morphology.convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3,1)  
    dilatedMask = binary_dilation(convex_mask,structure=struct,iterations=10) 
    return dilatedMask

#def HuToGray(img, ww=1500, wl=-600,water_HU=0,bone_HU=400):
def HuToGray(img, ww=1800, wl=-300,water_HU=0,bone_HU=300):
    min_val = (2*wl-ww)/2
    max_val = (2*wl+ww)/2
    img_out = (img-min_val)/(max_val-min_val)
    img_out[img_out>1]=1
    img_out[img_out<0]=0
    img_out = (255*img_out)
    water_value = 255*(water_HU-min_val)/(max_val-min_val)
    bone_value = 255*(bone_HU-min_val)/(max_val-min_val)
    return img_out, water_value, bone_value

def get_scan_info(mhd_dir,spacing=(1.0,1.0,1.0)):
    '''读取某文件夹内的所有dicom文件,并提取像素值(-4000 ~ 4000)'''
    info=dict()
    old_itk_img = sitk.ReadImage(mhd_dir) #x,y,z
    old_spacing = old_itk_img.GetSpacing() #x,y,z
    pdb.set_trace()
    #new_spacing = (spacing[0], spacing[1], old_spacing[2])
    new_spacing = (1.0,1.0,1.0)
    # new_spacing=old_spacing
    new_itk_img = Resample_ITK_Image(old_itk_img, old_spacing, new_spacing)
    scan = sitk.GetArrayFromImage(new_itk_img)                             #z,y,x
    info['seriesuid']=mhd_dir.split('/')[-1].split('.mhd')[0]
    info['old_spacing']=np.array(list(reversed(old_spacing)))              #z,y,x
    info['new_spacing']= np.array((1.0,1.0,1.0))
    info['origin']=np.array(list(reversed(new_itk_img.GetOrigin())))       #z,y,x
    info['direction']=np.diagonal(np.array(list(reversed(new_itk_img.GetDirection()))).reshape(3,3))#z,y,x

    return scan.astype('float32'), info

def get_lung_mask(scan_data,spacing):
    
    bw = Binarize_Scan(scan_data, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = Analysis_Scan(bw, spacing, cut_num=cut_num, vol_limit=[0.1,7.5])
        cut_num = cut_num + cut_step
    bw = Fill_Hole(bw)
    m1, m2= Keep_Two_lung(bw, spacing)
    Mask = m1+m2
    # return Mask
    
    dm1 = Dilated_Mask(m1)
    dm2 = Dilated_Mask(m2)    
    dilatedMask = dm1+dm2
    extramask = dilatedMask ^ Mask
    return dilatedMask, extramask

def get_lung_rect(lung,margin=1):
    ''' 剔除非肺腔部分，即获取肺腔掩模的最大外界长方体'''
    depth, height, width = lung.shape    
    if not lung.max():
        return np.array([[0,depth],[0,height],[0,width]])
    zz, yy, xx = np.where(lung)
    min_z, max_z = max(0, np.min(zz)-margin), min(depth, np.max(zz)+margin)
    min_y, max_y = max(0, np.min(yy)-margin), min(height, np.max(yy)+margin)
    min_x, max_x = max(0, np.min(xx)-margin), min(width, np.max(xx)+margin)
    bbox = np.array([[min_z,max_z],[min_y,max_y],[min_x,max_x]])
    bbox = np.floor(bbox).astype('int')
    return bbox

def preparing(mhd_dir,coords):
    print('SAVE RESULTS:',mhd_dir)
    # print(coords.shape)
    # print(coords)

    global data_ip
    global data_op

    # if os.path.exists(os.path.join(data_op, mhd_dir.split('/')[-1].replace('.mhd','_clean.npy'))):        
    #     print( '%s is exists'%os.path.join(data_op, mhd_dir.split('/')[-1].replace('.mhd','_clean.npy')))
    #     return

    #获得scan 的array,info
    scan, info = get_scan_info(mhd_dir)   #图片像素提取，尺度是new spacing，单位是像素

    #获得lung
    dilatedMask, extraMask = get_lung_mask(scan,info['new_spacing'])#尺度是new spacing，单位是像素
    
    #获得外接框
    rect = get_lung_rect(dilatedMask)#z,y,x 尺度是new spacing，单位是像素
     
    #预处理
    scan, pad_value, bone_value = HuToGray(scan,ww=1800, wl=-300)
    scan = scan*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
    bones = scan * extraMask > bone_value
    scan[bones] = pad_value

    scan = scan[rect[0,0]:rect[0,1],
                rect[1,0]:rect[1,1],
                rect[2,0]:rect[2,1]]
    #import pdb;pdb.set_trace()
    voxels = (coords[:,:3][:,::-1]-info['origin'])/info['direction']#z,y,x,物理距离,单位mm
    voxels = voxels / info['new_spacing']
    voxels = voxels - rect[:,0]

    # diameters = coords[:, 3] / info['spacing']
    #diameters = coords[:, 3:] * info['spacing'][1] / info['new_spacing'][1]
    diameters = coords[:,3:] / info['new_spacing'][1]
    nodules = np.hstack([voxels, diameters]) #z,y,x
    # nodules = np.concatenate((voxels, diameters), axis=1)  # z,y,x
    np.save(os.path.join(data_op, info['seriesuid']+'_clean.npy'), scan)
    np.save(os.path.join(data_op, info['seriesuid']+'_label.npy'), nodules.astype('float32'))
    


if __name__ == '__main__':

    # pool = Pool(8)
    data_ip = '/media/wow_cv/LUNA16/image/'#mhd以及raw文件存放的地址
    data_op = '/media/wow_cv/zly/LungnoduleDetection/DSB/prep_result2/'#预处理结果输出地址

    annotations_file = '/media/wow_cv/LUNA16/csv/annotations.csv'#标签文件
    annotations = pd.read_csv(annotations_file,dtype={'seriesuid':str})

    seriesuids_file = '/media/wow_cv/LUNA16/csv/seriesuids.csv'#mhd文件列表
    seriesuids = pd.read_csv(seriesuids_file,header=None)
    #import pdb;pdb.set_trace()

    seriesuids=[name[0] for name in seriesuids.values]
    if not os.path.exists(data_op):
        os.makedirs(data_op)  
    print(len(seriesuids))
    for seriesuid in seriesuids:
        #seriesuid = '1.3.6.1.4.1.14519.5.2.1.6279.6001.228511122591230092662900221600'
        scan_dir = os.path.join(data_ip, seriesuid+'.mhd')
        coords = np.array(annotations[annotations['seriesuid']==seriesuid])[:,1:].astype(np.float32)
        # print(coords.shape)
        # print(coords)
        preparing(scan_dir, coords)
        # pool.apply_async(preparing, (scan_dir,coords,))
        # pool.apply(preparing, (scan_dir, coords,))
    # pool.close()
    # pool.join()
