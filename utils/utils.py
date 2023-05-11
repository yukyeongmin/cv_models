import numpy as np
import tensorflow as tf

def normalization_to_pixel(normal):
    # (batch, 2) 1로 normalizaiton되어 있는 형태 
    # type - numpy

    normal_y = normal[:,0].astype(float)
    normal_x = normal[:,1].astype(float)
    y = 2*normal_y*256                  # elevation
    x = normal_x*1024                              # azimuth
    y = np.expand_dims(y, axis=1)
    x = np.expand_dims(x, axis=1)
    pixel = np.concatenate((x,y),axis=1)
    return pixel # (batch,2)

def pixel_to_position(image_shape, patch_size, y_real):
    '''
    image_shape = (height, width, channel)
    y_real = (가로, 세로)
    '''
    width = int(image_shape[1]//patch_size) # scalar
    height = int(image_shape[0]//patch_size) # scalar

    x = y_real[0]//patch_size # vector (batch,1)
    y = y_real[1]//patch_size # vector (batch,1)

    return (x,y) # 0부터 시작

def position_to_label(image_shape, patch_size, y_position):
    '''
    y_position = (2,) 높이, 폭
    '''

    width = int(image_shape[1]//patch_size) # scalar
    height = int(image_shape[0]//patch_size) # scalar

    label = y_position[0] + y_position[1]*width

    return int(label)

def pixel_to_startpoint(image_shape, patch_size, y):
    '''
    이미지에서 위치가 들어오면, 
    해당하는 패치의 좌상단의 픽셀 위치를 리턴

    y_label = (2,)
    '''
    width = int(image_shape[1]//patch_size) # scalar
    height = int(image_shape[0]//patch_size) # scalar
    position = pixel_to_position(image_shape, patch_size, y) # 0부터 시작

    y = position[0]*patch_size
    x = position[1]*patch_size
    startpoint = (y,x)
    return startpoint

def class_to_startpoint(image_shape, patch_size, patch_class):
    '''
    모델의 예상 class (patch번호)가 들어오면 
    해당하는 patch의 좌상단의 픽셀 위치를 리턴

    label = (patch_num)
    '''
    width = int(image_shape[1]//patch_size) # scalar
    height = int(image_shape[0]//patch_size) # scalar

    label = np.where(patch_class ==1) 
    x,y = np.divmod(label[0],width)

    startpoint = (y[0]*patch_size,x[0]*patch_size)
    return startpoint, label[0]