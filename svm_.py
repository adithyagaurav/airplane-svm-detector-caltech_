Python 3.7.0 (v3.7.0:1bf9cc5093, Jun 27 2018, 04:59:51) [MSC v.1914 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> import cv2
>>> import os
>>> 
g
Traceback (most recent call last):
  File "<pyshell#3>", line 2, in <module>
    g
NameError: name 'g' is not defined
>>> ppath=r'C:\Program Files\Python37\airplanes'
>>> npath=r'‪C:\Program Files\Python37\background'
>>> for filename in os.listdir(ppath):
	img=cv2.imread(os.path.join(ppath, filename))

	
>>> img.shape
(524, 800, 3)
>>> posImages=[]
>>> for filename in os.listdir(ppath):
	img=cv2.imread(os.path.join(ppath,filename))
	gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray=cv2.resize(gray,(20,20))
	posImages.append(gray)

	
>>> posImags.shape
Traceback (most recent call last):
  File "<pyshell#18>", line 1, in <module>
    posImags.shape
NameError: name 'posImags' is not defined
>>> posImages.shape
Traceback (most recent call last):
  File "<pyshell#19>", line 1, in <module>
    posImages.shape
AttributeError: 'list' object has no attribute 'shape'
>>> posImages.size
Traceback (most recent call last):
  File "<pyshell#20>", line 1, in <module>
    posImages.size
AttributeError: 'list' object has no attribute 'size'
>>> im=posImage[1070]
Traceback (most recent call last):
  File "<pyshell#21>", line 1, in <module>
    im=posImage[1070]
NameError: name 'posImage' is not defined
>>> im=posImages[1070]
>>> im.shape
(20, 20)
>>> negImages=[]
>>> for filename in os.listdir(npath):
	imgn=cv2.imread(os.path.join(npath,filename))
	grayn=cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
	grayn=cv2.resize(grayn,(20,20))
	negImages.append(grayn)

	
Traceback (most recent call last):
  File "<pyshell#30>", line 1, in <module>
    for filename in os.listdir(npath):
OSError: [WinError 123] The filename, directory name, or volume label syntax is incorrect: '\u202aC:\\Program Files\\Python37\\background'
>>> npath=r'"C:\Program Files\Python37\background"
SyntaxError: EOL while scanning string literal
>>> npath=r'"C:\Program Files\Python37\background"
SyntaxError: EOL while scanning string literal
>>> npath=r"C:\Program Files\Python37\background"
>>> for filename in os.listdir(npath):
	imgn=cv2.imread(os.path.join(npath,filename))
	grayn=cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
	grayn=cv2.resize(grayn,(20,20))
	negImages.append(grayn)

	
>>> negImages.shape
Traceback (most recent call last):
  File "<pyshell#36>", line 1, in <module>
    negImages.shape
AttributeError: 'list' object has no attribute 'shape'
>>> im=negImages[450]
>>> im.shape
(20, 20)
>>> len(posImages)
1074
>>> plabels=np.ones(len(posImages))
>>> nlabels=np.zeros(len(negImages))
>>> winSize=(20,20)
>>> blockSize=(8,8)
>>> blockStride=(4,4)
>>> cellSize=(8,8)
>>> nbins=9
>>> derivAperture=1
>>> winSigma=-1
>>> histogramNormType=0
>>> L2HysThreshold=0.2
>>> gammaCorrection=1
>>> nlevels=64
>>> signedGradient=True
>>> hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
>>> hog_descriptors=[]
>>> hog_descriptors_pos=[]
>>> for img in posImages:
	hog_descriptors_pos.append(hog.compute(img))

	
>>> hog_descriptors_pos.shape
Traceback (most recent call last):
  File "<pyshell#61>", line 1, in <module>
    hog_descriptors_pos.shape
AttributeError: 'list' object has no attribute 'shape'
>>> a=np.squeeze(hog_descriptors_pos)
>>> a.shape
(1074, 144)
>>> a[1].shape
(144,)
>>> b=np.reshape(a[1],(20,20))
Traceback (most recent call last):
  File "<pyshell#65>", line 1, in <module>
    b=np.reshape(a[1],(20,20))
  File "C:\Program Files\Python37\lib\site-packages\numpy\core\fromnumeric.py", line 257, in reshape
    return _wrapfunc(a, 'reshape', newshape, order=order)
  File "C:\Program Files\Python37\lib\site-packages\numpy\core\fromnumeric.py", line 52, in _wrapfunc
    return getattr(obj, method)(*args, **kwds)
ValueError: cannot reshape array of size 144 into shape (20,20)
>>> 
>>> hog_descriptors_pos=np.squeze(hog_descriptors_pos)
Traceback (most recent call last):
  File "<pyshell#67>", line 1, in <module>
    hog_descriptors_pos=np.squeze(hog_descriptors_pos)
AttributeError: module 'numpy' has no attribute 'squeze'
>>> hog_descriptors_pos=np.squeeze(hog_descriptors_pos)
>>> hog_descriptors_neg=[]
>>> for img in negImages:
	hog_descriptors_neg.append(hog.compute(img))

	
>>> hog_descriptors_neg=np.squeeze(hog_descriptors_neg)
>>> hog_descriptors_neg.shape
(451, 144)
>>> len(hog_descriptors_pos)
1074
>>> train_p=int(0.9*len(hog_descriptors_pos))
>>> train_n=int(0.9*len(hog_descriptors_neg))
>>> hog_descriptors_pos_train, hog_descriptors_pos_test=np.split(hog_descriptors_pos, [train_p])
>>> hog_descriptors_neg_train, hog_descriptors_neg_test=np.split(hog_descriptors_neg, [train_n])
>>> plabels_train,plabels_test=np.split(plabels, [train_p])
>>> nlabels_train, nlabels_test=np.split(nlabels, [train_n])
>>> hog_descriptors_train=np.float32(np.concatenate([hog_descriptors_pos_train, hog_descriptors_neg_train]))
>>> hog_descriptors_train.shape
(1371, 144)
>>> hog_descriptors_test=np.float32(np.concatenate([hog_descriptors_pos_test, hog_descriptors_neg_test]))
>>> labels_train=np.concatenate([plabels_train,nlabels_train])
>>> labels_test=np.concatenate([plabels_test,nlabels_test])
>>> model=cv2.ml.SVM_create()
>>> model.setGamma(0.50625)
>>> model.setC(12.5)
>>> model.setKernel(cv2.ml.SVM_RBF)
>>> model.setType(cv2.ml.SVM_C_SVC)
>>> labels_train.shape
(1371,)
>>> model.train(hog_descritors_train, cv2.ml.ROW_SAMPLE, labels_train)
Traceback (most recent call last):
  File "<pyshell#95>", line 1, in <module>
    model.train(hog_descritors_train, cv2.ml.ROW_SAMPLE, labels_train)
NameError: name 'hog_descritors_train' is not defined
>>> model.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
Traceback (most recent call last):
  File "<pyshell#96>", line 1, in <module>
    model.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
cv2.error: OpenCV(3.4.2) d:\build\opencv\opencv-3.4.2\modules\ml\src\data.cpp:292: error: (-215:Assertion failed) responses.type() == 5 || responses.type() == 4 in function 'cv::ml::TrainDataImpl::setData'

>>> labels_train.type
Traceback (most recent call last):
  File "<pyshell#97>", line 1, in <module>
    labels_train.type
AttributeError: 'numpy.ndarray' object has no attribute 'type'
>>> labels_train.dtype
dtype('float64')
>>> labels_train=np.int32(labels_train)
>>> labels_test=np.int32(labels_test)
>>> labels_train.dtype
dtype('int32')
>>> model.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
True
>>> model.predict(hog_descriptors_test)[1].ravel()
array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
       1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
       1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0.], dtype=float32)
>>> predictions=model.predict(hog_descriptors_test)[1].ravel()
>>> accuracy=(labels_test==predictions).mean()
>>> print('percentage accuracy: %.2f'% (accuracy*100))
percentage accuracy: 93.51
>>> 
