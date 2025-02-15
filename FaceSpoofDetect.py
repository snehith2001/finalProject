#{'Fake': 0, 'Real': 1}
from tkinter import *
import tkinter
from tkinter import filedialog
import numpy as np
from tkinter.filedialog import askopenfilename
import pandas as pd 
from keras.optimizers import Adam
from keras.models import model_from_json
from tkinter import simpledialog

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Activation,BatchNormalization
import os
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from tkinter import messagebox
import cv2
from imutils import paths
import imutils
import cv2
import numpy as np
import pickle
import matplotlib.pyplot as plt

main = tkinter.Tk()
main.title("Image Classification Using CNN") #designing main screen
main.geometry("600x500")

global filename
global lbp_accuracy
global loaded_model

def get_pixel(img, center, x, y):
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    center = img[x][y]
    val_ar = []
    val_ar.append(get_pixel(img, center, x-1, y+1))     # top_right
    val_ar.append(get_pixel(img, center, x, y+1))       # right
    val_ar.append(get_pixel(img, center, x+1, y+1))     # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y))       # bottom
    val_ar.append(get_pixel(img, center, x+1, y-1))     # bottom_left
    val_ar.append(get_pixel(img, center, x, y-1))       # left
    val_ar.append(get_pixel(img, center, x-1, y-1))     # top_left
    val_ar.append(get_pixel(img, center, x-1, y))       # top
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val    


def upload(): #function to upload tweeter profile
    global filename
    filename = filedialog.askopenfilename(initialdir="testimages")
    messagebox.showinfo("File Information", "image file loaded")
    

def generateModel():
    global loaded_model
    global lbp_accuracy
    if os.path.exists('model/lbpmodel.json'):
        with open('model/lbpmodel.json', "r") as json_file:
           loaded_model_json = json_file.read()
           loaded_model = model_from_json(loaded_model_json)
        json_file.close()
        loaded_model.load_weights("model/lbpmodel_weights.h5")
        loaded_model._make_predict_function()   
        print(loaded_model.summary)
        f = open('model/lbphistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[0] * 100
        lbp_accuracy = accuracy
        messagebox.showinfo("LBP CNN Model Generated", "LBP CNN Training Model Accuracy on Fake & Real Faces = "+str(accuracy))
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (48, 48, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 128, activation = 'relu'))
        classifier.add(Dense(output_dim = 2, activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        train_datagen = ImageDataGenerator()
        test_datagen = ImageDataGenerator()
        training_set = train_datagen.flow_from_directory('LBP/train',
                                                 target_size = (48, 48),
                                                 batch_size = 32,
                                                 class_mode = 'categorical',
						 shuffle=True)
        test_set = test_datagen.flow_from_directory('LBP/validation',
                                            target_size = (48, 48),
                                            batch_size = 32,
                                            class_mode = 'categorical',
					    shuffle=False)
        hist = classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 1,
                         validation_data = test_set,
                         nb_val_samples = 2000)
        classifier.save_weights('model/lbpmodel_weights.h5')
        model_json = classifier.to_json()
        with open("model/lbpmodel.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        print(training_set.class_indices)
        print(classifier.summary)
        f = open('model/lbphistory.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        f = open('model/lbphistory.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        accuracy = acc[0] * 100
        lbp_accuracy = accuracy
        messagebox.showinfo("LBP CNN Model Generated", "LBP CNN Training Model Accuracy on Fake & Real Faces = "+str(accuracy))
        loaded_model = classifier

def classify():
    global loaded_model
    name = os.path.basename(filename)
    image_file = filename;
    img_bgr = cv2.imread(image_file)
    height, width, channel = img_bgr.shape
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_lbp = np.zeros((height, width,3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)
    cv2.imwrite('testimages/lbp_'+name, img_lbp)
    imagetest = image.load_img('testimages/lbp_'+name, target_size = (48,48))
    imagetest = image.img_to_array(imagetest)
    imagetest = np.expand_dims(imagetest, axis = 0)
    preds = loaded_model.predict(imagetest)
    print(str(preds)+" "+str(np.argmax(preds)))
    predict = np.argmax(preds)
    msg = ""
    if predict == 0:
        msg = "Image Contains Spoof face"
    if predict == 1:
        msg = "Image Contains non Spoofing face"
    imagedisplay = cv2.imread(filename)
    orig = imagedisplay.copy()
    output = imutils.resize(orig, width=400)
    cv2.putText(output, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0,255,255), 2)
    cv2.imshow("Predicted Image Result ", output)
    imagedisplay = cv2.imread('testimages/lbp_'+name)
    orig = imagedisplay.copy()
    output = imutils.resize(orig, width=400)
    os.remove('testimages/lbp_'+name)
    cv2.imshow("LBP Image", output)
    cv2.waitKey(0)
    


def extensionGabor():
    global lbp_accuracy
    gabor_accuracy = 0

    X_train = np.load('model/gaborX.txt.npy')
    test = X_train[3]
    test = cv2.resize(test,(500,300))
    cv2.imshow("Sample image after applying Gabor Filters",test)
    cv2.waitKey(0)

    f = open('model/gaborhistory.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    gabor_accuracy = acc[9] * 100

    msg = "LBP Accuracy : "+str(lbp_accuracy)+"\nGabor Accuracy : "+str(gabor_accuracy)
    messagebox.showinfo("LBP & Gabor CNN Accuracy", msg)

    height = [lbp_accuracy,gabor_accuracy]
    bars = ('LBP CNN Accuracy','Gabor CNN Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.title("Propose LBP & Extension Gabor Accuracy Comparison Graph")
    plt.show()

    
font = ('times', 16, 'bold')
title = Label(main, text='Deep Texture Features for Robust Face Spoofing Detection', justify=LEFT)
title.config(bg='lavender blush', fg='DarkOrchid1')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=100,y=5)
title.pack()

font1 = ('times', 14, 'bold')
model = Button(main, text="Generate NLBPNet Train & Test Model", command=generateModel)
model.place(x=200,y=100)
model.config(font=font1)  

uploadimage = Button(main, text="Upload Test Image", command=upload)
uploadimage.place(x=200,y=150)
uploadimage.config(font=font1) 

classifyimage = Button(main, text="Classify Picture In Image", command=classify)
classifyimage.place(x=200,y=200)
classifyimage.config(font=font1) 

exitapp = Button(main, text="Extension Gabor Filter vs LBP Accuracy", command=extensionGabor)
exitapp.place(x=200,y=250)
exitapp.config(font=font1) 

main.config(bg='light coral')
main.mainloop()
