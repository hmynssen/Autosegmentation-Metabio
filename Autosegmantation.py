import matplotlib.pyplot as plt
import Segmentation
import pydicom
import os
import numpy as np
import time
import skimage.feature as ft
from skimage.measure import find_contours
from scipy import signal

##pseudo-settings###

save_tmp = False
save_initialguess = False
plot_lenHisto = False
plot_figure = True
subtract = False
pixeltreshold = 1460

#############
filenumber = 0
dir_name = "DCM/Heitor"#"DCM/HU/T2_FLAIR_TE_38"
for file in os.listdir(dir_name):
    filenumber += 1
    #pwd = os.getcwd()
    #os.chdir("DCM/HU")
    dataset = pydicom.dcmread(dir_name + '/'+ file)
    #os.chdir(pwd)


    if subtract:
        contrast = 1*np.load("tmHU.npy")
        image = dataset.pixel_array
        kernel = np.outer(signal.gaussian(image.shape[0],0.3), signal.gaussian(image.shape[1],0.3))
        contrast = signal.fftconvolve(contrast, kernel, mode='same')
        image[contrast > 0] = pixeltreshold
    else:
        image = dataset.pixel_array
    start = time.time()
    seg = Segmentation.ActiveContours(
                                image, multilevel = 3, mu = [0.5,0.5,0.5],
                                lambdas = [1.2,0.6,0.8,1,1,1,1,1], sigma = 2,
                                maxiter = 10, dt=0.1, initguess="cv"
                                )##level = 1, mu = 0, lambdas = [1.13, 0.95]; [1.,0.65,0.6,0.7,0.8,0.9,1.,1.]
    print("\nTotal run time: "+str(time.time() - start))
    if plot_figure:
        if len(seg[0].shape)>2:
            totalsubplots = seg[0].shape[2]
        else:
            totalsubplots = 1
        lines = totalsubplots//3 + 1


        if len(seg[0].shape)>2:
            fig, axs = plt.subplots(lines, 3)
            axs[0,0].imshow(dataset.pixel_array, cmap=plt.cm.gray)
        else:
            fig, axs = plt.subplots(lines, 2)
            axs[0].imshow(dataset.pixel_array, cmap=plt.cm.gray)
        col=1
        line=0
        for i in range(0,totalsubplots):
            if totalsubplots == 1:
                axs[col].imshow(seg[0], cmap=plt.cm.gray)
            else:
                axs[line, col].imshow(seg[0][:,:,i], cmap=plt.cm.gray)
            col += 1
            if col == 3:
                col = 0
                line += 1

        #plt.savefig('HUiter1500.png', bbox_inches = 'tight', pad_inches = 0)
        plt.show()
        #plt.clf()


    if plot_lenHisto:
        plt.hist(lengths, bins = 1000)
        plt.xlabel("Number of point on contour segments")
        plt.title("Original segmentation")
        plt.show()
        plt.clf()
        plt.hist(lengths2, bins = 1000)
        plt.xlabel("Number of point on contour segments")
        plt.title("Blurred segmentation")
        plt.show()

    if save_tmp:
        np.save(str(filenumber)+"ggg.npy", seg[0])
    if save_initialguess:
        np.save("initialguess", seg[2])
