import cv2
import numpy as np


def getImagePartAsNumpyArray(image,box, new_shape):
    #if new_shape is 0 then we don't need to resize the image
    if (new_shape == 0):
        return image[box[1]:box[3], box[0]:box[2]]
    else:
        try:
            res = cv2.resize(image[box[1]:box[3], box[0]:box[2]],new_shape)
        except:
            res = None
        finally:
            return res
def getImage(path):
    return cv2.imread(path)
def pyramidGenerator(image,minsize = (20,20),scaledown = 1.5):
    w,h,c = image.shape
    if (w >= minsize[0] or h >= minsize[1]):
        yield image
        while True:
            w = int(w / scaledown)
            h = int(h / scaledown)
            if (w >= minsize[0] or h >= minsize[1]):
                yield cv2.resize(image,(w, h))
            else:
                break


def printImages():
    for image in pyramidGenerator(getImage('test_pic.jpg'),(200,200),1.5):
        cv2.imshow('image',np.asarray(image))
        cv2.waitKey()
def doIntersect(box1,box2):
    b1x1 = box1[1] - box1[3] / 2
    b1x2 = box1[1] + box1[3] / 2
    b1y1 = box1[2] - box1[4] / 2
    b1y2 = box1[2] + box1[4] / 2

    b2x1 = box2[1] - box2[3] / 2
    b2x2 = box2[1] + box2[3] / 2
    b2y1 = box2[2] - box2[4] / 2
    b2y2 = box2[2] + box2[4] / 2

    if((b1x2<b2x1) or (b1x1>b2x2)
       or (b1y1<b2y1) or (b1y1*b2y2)):
        return False
    else:
        return True
def doOverlap(box1,box2,thresh=0.5):
    b1x1 = box1[1] - box1[3] / 2
    b1x2 = box1[1] + box1[3] / 2
    b1y1 = box1[2] - box1[4] / 2
    b1y2 = box1[2] + box1[4] / 2

    b2x1 = box2[1] - box2[3] / 2
    b2x2 = box2[1] + box2[3] / 2
    b2y1 = box2[2] - box2[4] / 2
    b2y2 = box2[2] + box2[4] / 2

    #corners of interection:
    qf = max(b1x1,b2x1)
    qa = min(b1x2,b2x2)
    rf = max(b1y1,b2y1)
    ra = min(b1y2,b2y2)

    a1 = (b1x2-b1x1)*(b1y2-b1y1)
    a2 = (b2x2-b2x1)*(b2y2-b2y1)
    intersection = (qa-qf)*(ra-rf)
    try:
        iou = intersection/(a1+a2-intersection)
    except:
        return True
    print(intersection)
    print(a1)
    print(a2)
    if(iou>thresh):
        return True
    else:
        return False