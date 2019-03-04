import os,  glob
os.environ['KERAS_BACKEND'] = 'tensorflow'

import cv2
import numpy as np
import keras
import tensorflow as tf
import time

import sys

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin
    __package__ = "keras_retinanet.bin"

from .. import models


CROP_SIZE=900;
STRIDE = 800;


#MODEL='/media/usb500gb/models/retina600dynamic/7/resnet50_csv_13.h5'
MODEL='/disk1/deep/cosmos900/mobilenet224/mobilenet224_1.0_csv_02.h5'

def read_image_bgr(path):
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()

def preprocess_image(x):
    # mostly identical to "https://github.com/fchollet/keras/blob/master/keras/applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already
    x = x.astype(keras.backend.floatx())
    if keras.backend.image_data_format() == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] -= 103.939
            x[1, :, :] -= 116.779
            x[2, :, :] -= 123.68
        else:
            x[:, 0, :, :] -= 103.939
            x[:, 1, :, :] -= 116.779
            x[:, 2, :, :] -= 123.68
    else:
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x

def resize_image(img, min_side=CROP_SIZE, max_side=CROP_SIZE*2):
    (rows, cols, _) = img.shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, wich can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def IoU(a,b):
# Area of Intersection/area of Union for two rects
    w=min(a[2],b[2])-max(a[0],b[0])
    h=min(a[3],b[3])-max(a[1],b[1])
    I=0
    if w>0 and h>0:
        I=w*h
    w1=a[2]-a[0]
    w2=b[2]-b[0]
    h1=a[3]-a[1]
    h2=b[3]-b[1]
    U=w1*h1+w2*h2-I
    return I/U

def NonMaxima(detections, IoU_threshold=0.5):
#Suppression of non-max
    N = len(detections)
    is_max=np.ones(N, dtype=np.bool)
    for i in range(0,N-1):
        for j in range(i+1,N):
            if IoU(detections[i], detections[j]) < IoU_threshold:
                continue;
            if detections[i][4] < detections[j][4]:
                is_max[i] = 0
            if detections[i][4] > detections[j][4]:
                is_max[j] = 0
    res=np.empty(shape=(0,6))         #TODO:change to list as it is faster
    for i in range(0,N):
        if is_max[i]:
            res=np.append(res,detections[i].reshape(1,6),axis=0)
    return res


def get_detections(img, model, score_threshold=0.01, max_detections=100, save_path=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    #all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    image = preprocess_image(img.copy())
#    image = cv2.copyMakeBorder(image,0,CROP_SIZE-(img.shape[0]-100) % STRIDE,0,img.shape[1],cv2.BORDER_CONSTANT,value=BLACK)
    H=image.shape[0]
    W=image.shape[1]
    numCrops=0
    alldets=np.empty(shape=(0,6))     # TODO: Change to list as it is faster
    # run network
    for top in range(0,H,STRIDE):
        for left in range(0,W,STRIDE):
            if (left + CROP_SIZE-STRIDE >= W) or (top + CROP_SIZE-STRIDE >= H):
                 continue; # no sense to make such small crops
            numCrops+=1
            right = left + CROP_SIZE
            bottom = top + CROP_SIZE
            crop=image[top:min(bottom,H),left:min(right,W)]
            crop=cv2.copyMakeBorder(crop, 0, CROP_SIZE-crop.shape[0],0,CROP_SIZE-crop.shape[1],cv2.BORDER_CONSTANT,value=(0,0,0))
            #cv2.imwrite("cr"+str(left)+"-"+str(top)+".jpg",crop,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
            #crop=read_image_bgr("cr"+str(left)+"-"+str(top)+".jpg")
            #crop = preprocess_image(crop)
            #crop, scale = resize_image(crop) # not needed for crops 600x600

            detections, scores, labels = model.predict_on_batch(np.expand_dims(crop, axis=0))
            # shift according to crop position
            detections[:, :, 0]=detections[:, :, 0]+left
            detections[:, :, 2]=detections[:, :, 2]+left
            detections[:, :, 1]=detections[:, :, 1]+top
            detections[:, :, 3]=detections[:, :, 3]+top
            # clip to image shape
            detections[:, :, 0] = np.maximum(0, detections[:, :, 0])
            detections[:, :, 1] = np.maximum(0, detections[:, :, 1])
            detections[:, :, 2] = np.minimum(image.shape[1], detections[:, :, 2])
            detections[:, :, 3] = np.minimum(image.shape[0], detections[:, :, 3])
            # correct boxes for image scale
            #detections[0, :, :4] /= scale # not needed for crops 600x600

            indices = np.where(scores[0, :] > score_threshold)[0]

            result=np.concatenate([detections,  np.expand_dims(scores,axis=2),np.expand_dims(labels,axis=2)], axis=2)[0,indices]
            alldets=np.append(alldets,result,axis=0)
    NMdets=NonMaxima(alldets,0.5)

    # copy detections to all_detections
#    for label in range(generator.num_classes()):
 #       all_detections[i][label] = image_detections[image_predicted_labels == label, :]

#        print('{}/{}'.format(i, generator.size()), end='\r')

    return NMdets, numCrops


def scale_img(img_in, Height, kren_deg, tangazh_deg,f_distance,sensorWidth,sensorHeight):
    if Height==0:
        return img_in, [1,1]
    kren_rad = kren_deg * pi / 180.0
    tangazh_rad = tangazh_deg  * pi / 180.0
    PPMref = 10.0 #desired pixels per meter
    x = 1.0*f_distance / Height; # 1m on the matrix
    len_v = (img_in.shape[0] * x) / sensorHeight
    len_h = (img_in.shape[1] * x) / sensorWidth
    vscale = PPMref / len_v
    hscale = PPMref / len_h
    scale = [ vscale / cos(tangazh_rad), hscale / cos(kren_rad) ]
    img_out=cv2.resize(img_in, (0,0),fx=scale[0],fy=scale[1]);
    return img_out,scale



def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# Load NN model
st0 = time.time()
keras.backend.tensorflow_backend.set_session(get_session())
model = models.load_model(MODEL, backbone_name="mobilenet224_1.0",convert=True)
st3 = time.time()

#img=cv2.imread("in.jpg");
#img,scale=scale_img(img,dr.metadata.height, dr.metadata.pitch, dr.metadata.roll,dr.metadata.focalLength, dr.metadata.sensorWidth,dr.metadata.sensorHeight);
#cv2.imwrite("in_scaled.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY), 100]);
fl=glob.glob("/home/deep/COSMOStest/*.png")
totalCrops=0
totalTime=0
for fn in fl:
    print(fn)
    img=cv2.imread(fn)
    st = time.time()
    dets,crops=get_detections(img, model)
    st1 = time.time()
    with open(fn+".det","w") as f:
        for det in dets:
            f.write(str(int(det[5]))+' '+str(int(det[0]))+' '+str(int(det[1]))+' '+str(int(det[2]-det[0]))+' '+str(int(det[3]-det[1]))+' 1 '+str(det[4])+'\n')
            cv2.rectangle(img,(int(det[0]),int(det[1])),(int(det[2]),int(det[3])),(0,255,0),3)
    cv2.imwrite(fn+'.det.jpg',img)
    print('Number of crops = {}'.format(crops))
    print('Inference time = {}'.format(st1 - st))
    print('Inference per crop = {}'.format((st1 - st)/crops))
    print('Elapsed time = {}'.format(st1 - st0))
    totalCrops+=crops
    totalTime+=st1-st
st4 = time.time()
print('Initialization time = {}'.format(st3 - st0))
print('Number of images = {}'.format(len(fl)))
print('Total number of crops = {}'.format(totalCrops))
print('Total inference time = {}'.format(totalTime))
print('Average inference per crop = {}'.format(totalTime/totalCrops))
print('Total time elapsed = {}'.format(st4 - st0))
