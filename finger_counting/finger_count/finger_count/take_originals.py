import cv2
import copy
import os
import numpy as np
import random

def generate_augmented_images():

    orig_path = './originals/'
    dest_path = './augmented/'
    path = os.listdir(orig_path)

    file = path[5]
    for file in path:
        fname,ext = os.path.splitext(file)

        orig_img = cv2.imread(os.path.join(orig_path,file),cv2.IMREAD_GRAYSCALE)
        h,w = orig_img.shape

        flip = [0,1]
        rotate = random.sample(range(-30, 30), 3)
        shiftx = random.sample(range(-15, 15), 2)
        shifty = random.sample(range(-15, 15), 2)
        zoom = random.sample(range(0, 30), 3)

        count = 1
        for f in flip:
            for r in rotate:
                for sx in shiftx:
                    for sy in shifty:
                        for z in zoom:
                            img = copy.deepcopy(orig_img)
                            img = ~img
                            # flip
                            if f == 1:
                                img = cv2.flip(img,1)
                            # rotate
                            rmatr = cv2.getRotationMatrix2D(center=(w//2,h//2),angle=r,scale=1)
                            img = cv2.warpAffine(img,rmatr,(w,h),borderMode = cv2.BORDER_REPLICATE)
                            ## shift
                            smatr = np.float32([[1,0,sx],[0,1,sy]])
                            img = cv2.warpAffine(img,smatr,(w,h),borderMode = cv2.BORDER_REPLICATE)
                            ## zoom
                            inc_w = (w * z)//100
                            inc_h = (h * z)//100
                            im = cv2.resize(img, (w+inc_w,h+inc_h), interpolation = cv2.INTER_AREA)
                            img = im[inc_h//2:inc_h//2+h,inc_w//2:inc_w//2+w]
                            # save new image
                            img = ~img
                            new_file_name = "from_%s_%d%s" % (fname,count,ext)
                            cv2.imwrite(os.path.join(dest_path,new_file_name),img)
                            count += 1

def binaryMask(img):
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, ksize=(7,7), sigmaX=2)
    img = cv2.adaptiveThreshold(img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, blockSize=13, C=2)
    ret, img = cv2.threshold(img, thresh=25, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img

def binaryMask2(img):
    img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
    img = cv2.bilateralFilter(img, 7, 50, 50)
    img = cv2.Canny(img, 30, 60)
    kernel = np.ones((3,3), np.uint8) 
    img = cv2.dilate(img,kernel)
    img = ~img
    return img

def main():

    from keras.models import load_model
    model = load_model('model4.h5')

    # Initialize variables
    mask_visible2 = False
    mask_visible = False
    mode_predict = False
    x = 400
    y = 150
    w = 200
    h = 200
    nb = 0
    orig_path = './originals/'

    # Define last count value for original pictures
    count = len(os.listdir(orig_path))+1

    # Initialize Webcam
    cam = cv2.VideoCapture(0)
    cv2.namedWindow('WebCam', cv2.WINDOW_NORMAL)

    while True:
        # Get image from Camera
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        # copy frame to window to work on the copy
        window = copy.deepcopy(frame)

        # Define a ROI, show bounds as a rectangle
        roi = frame[y:y+h,x:x+w]
        cv2.rectangle(img=window, pt1=(x,y), pt2=(x+w,y+h), color=(0,255,0), thickness=2)

        masked = binaryMask(roi)
        masked2 = binaryMask2(roi)

        # Shows masks in ROI
        if mask_visible:    
            window[y:y+h,x:x+w] = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)

        if mask_visible2:
            window[y:y+h,x:x+w] = cv2.cvtColor(masked2, cv2.COLOR_GRAY2BGR)
    
        # Use keyboard 
        key = cv2.waitKey(5) & 0xff
        
        # select nb of finger class for saving
        keys={
                ord('0'):0,
                ord('1'):1,
                ord('2'):2,
                ord('3'):3,
                ord('4'):4,
                ord('5'):5,
             }
        if key in keys:
            nb = keys.get(key)

        # save image to disk as original
        if key == ord('s') and mode_predict == False:
            
            #saveimg = masked
            saveimg = cv2.cvtColor(roi, code=cv2.COLOR_BGR2GRAY)
            
            filename = os.path.join(orig_path , 'original_%d_%d.png' % (nb,count))
            cv2.imwrite(filename,saveimg)
            count += 1

        # activate/deactivate filters
        if key == ord('x'):
            mask_visible = not mask_visible
        if key == ord('c'):
            mask_visible2 = not mask_visible2
        if key == ord('p'):
            mode_predict = not mode_predict
        if key == ord('q'):
            break

        # Show which number for saving
        if mode_predict == False:
            cv2.putText(window, "Mode - Record pictures", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(window, "(s) to save", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(window, "nb of fingers = %d" % nb, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.putText(window, "(p) for predict mode", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
        else:
            predimg = masked / 255.
            predimg = predimg.reshape(1,200,200,1)
            classes = model.predict(predimg)
            mypredict = np.argmax(classes)    
            cv2.putText(window, "Mode - Predict", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            cv2.putText(window, "prediction = %d" % mypredict, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            cv2.putText(window, "(p) for record mode", (10, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
        
        cv2.putText(window, "(q) to quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

        # show some guilines to help getting good pictures
        cv2.circle(img=window,center=(x+w//2,y+h//2), radius=w//2, color=(255,0,0), thickness=1)
        cv2.circle(img=window,center=(x+w//2,y+h//2), radius=(7*w//20), color=(255,0,0), thickness=1)

        # Display image
        cv2.imshow('WebCam', window)

    cam.release()


if __name__ == '__main__':
    #main()
    generate_augmented_images()