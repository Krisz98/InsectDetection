import utils
import cv2
import numpy as np
import tensorflow as tf

tf.keras.backend.clear_session()
kernel_1 = np.array([[1,2,1],
                   [2, 4,2],
                   [1,2,1]])/16
kernel_2 = np.array([[1,4,6,4,1],
                   [4, 16,24,16,4],
                   [6,24,36,24,6],
                   [4,16,24,16,4],
                   [1,4,6,4,1]])/256
boxes =[]
def slide(image,new_size, wsize = (50,50),steps = (5,5),x0=0,y0=0):
    i = 0
    for simage in utils.pyramidGenerator(image,wsize,scaledown=1.5):
        i=i+1
        print(i)
        width = simage.shape[0]
        height = simage.shape[1]
        for y in range(y0,height,steps[1]):
            if(y+wsize[1]>height):
                break
            for x in range(x0,width,steps[0]):
                if(x+wsize[0]>width):
                    continue
                #img = simage
                #cv2.imshow('asd',cv2.rectangle(img,(x,y),(x+wsize[0],y+wsize[1]),(255,0,0),3))
                #cv2.waitKey()
                cx = float(x+wsize[0]/2)/width
                cy = float(y+wsize[1]/2)/height
                w = float(wsize[0])/width
                h = float(wsize[1])/height
                yield utils.getImagePartAsNumpyArray(simage,(x,y,x+wsize[0],y+wsize[1]),new_size), cx,cy,w,h

def printImages():
    for i in slide(utils.getImage('insects.jpg'),0,(100,100),(10,10)):
        continue

def runPrediction(threshold = 0.5):
    print('Loading model...')
    model    = tf.keras.models.load_model('epoch_15.hdf5')
    placeholder = np.empty((1, 150, 150, 3))
    j = 0
    preds = []
    img = utils.getImage('insects.jpg')
    print('scanning...')
    for image,x,y,w,h in slide(img,(150,150),wsize=(40,40),steps =(20,20),x0= 0,y0=0):
        if (image is not None):
            image = image/255#cv2.filter2D(image/255, -1, kernel_2)
            cv2.imshow('cropped,resized', image)
            cv2.waitKey()
            placeholder[0] = image
            res = model.predict(placeholder)[0]
            print(res)
            if (res[1] > threshold):
                preds.append((res[1], x, y, w, h))
                # print(res)

    print('Image has been scanned!')
    img = utils.getImage('insects.jpg')
    width = img.shape[0]
    height = img.shape[1]
    print(preds)
    finalboxes = []
    preds = sorted(preds,key=(lambda x: x[0]),reverse=True)
    copyp = []
    print(preds)
    i =0
    while(i<len(preds)):
        stop = True #optimistic
        base = preds[i]
        comp = preds[i+1:]
        copyp.extend(preds[:i+1])
        for k in range(len(comp)):
            if(not utils.doOverlap(base,comp[k])):
                copyp.append(comp[k])
        preds = copyp[:]
        copyp = []
        i = i+1



    for box in preds:
        p,x,y,w,h = box
        p1 = [int((x-w/2)*width),int((y-h/2)*height)]
        p2 = [int((x+w/2)*width),int((y+h/2)*height)]
        print(p1)
        print(p2)
        img = cv2.rectangle(img,(p1[0],p1[1]),(p2[0],p2[1]),(255,0,0),3)
    cv2.imwrite('result.jpg',img)
    cv2.imshow('result',img)
    cv2.waitKey()
runPrediction()

