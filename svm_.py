import numpy as np
import cv2
import os
ppath=r'C:\Program Files\Python37\airplanes'
npath=r'C:\Program Files\Python37\background'
posImages=[]
for filename in os.listdir(ppath):
	img=cv2.imread(os.path.join(ppath,filename))
	gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray=cv2.resize(gray,(20,20))
	posImages.append(gray)

negImages=[]
for filename in os.listdir(npath):
	imgn=cv2.imread(os.path.join(npath,filename))
	grayn=cv2.cvtColor(imgn,cv2.COLOR_BGR2GRAY)
	grayn=cv2.resize(grayn,(20,20))
	negImages.append(grayn)

plabels=np.ones(len(posImages))
nlabels=np.zeros(len(negImages))
winSize=(20,20)
blockSize=(8,8)
blockStride=(4,4)
cellSize=(8,8)
nbins=9
derivAperture=1
winSigma=-1
histogramNormType=0
L2HysThreshold=0.2
gammaCorrection=1
nlevels=64
signedGradient=True
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels, signedGradient)
hog_descriptors_pos=[]
for img in posImages:
	hog_descriptors_pos.append(hog.compute(img))

hog_descriptors_pos=np.squeeze(hog_descriptors_pos)
hog_descriptors_neg=[]
for img in negImages:
	hog_descriptors_neg.append(hog.compute(img))

hog_descriptors_neg=np.squeeze(hog_descriptors_neg)
train_p=int(0.9*len(hog_descriptors_pos))
train_n=int(0.9*len(hog_descriptors_neg))
hog_descriptors_pos_train, hog_descriptors_pos_test=np.split(hog_descriptors_pos, [train_p])
hog_descriptors_neg_train, hog_descriptors_neg_test=np.split(hog_descriptors_neg, [train_n])
plabels_train,plabels_test=np.split(plabels, [train_p])
nlabels_train, nlabels_test=np.split(nlabels, [train_n])
hog_descriptors_train=np.float32(np.concatenate([hog_descriptors_pos_train, hog_descriptors_neg_train]))
hog_descriptors_test=np.float32(np.concatenate([hog_descriptors_pos_test, hog_descriptors_neg_test]))
labels_train=np.int32(np.concatenate([plabels_train,nlabels_train]))
labels_test=np.int32(np.concatenate([plabels_test,nlabels_test]))
model=cv2.ml.SVM_create()
model.setGamma(0.50625)
model.setC(12.5)
model.setKernel(cv2.ml.SVM_RBF)
model.setType(cv2.ml.SVM_C_SVC)
model.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)
predictions=model.predict(hog_descriptors_test)[1].ravel()
accuracy=(labels_test==predictions).mean()
print('percentage accuracy: %.2f'% (accuracy*100))
