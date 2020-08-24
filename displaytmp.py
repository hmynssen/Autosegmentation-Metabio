import matplotlib.pyplot as plt
import numpy as np
from scipy import signal, ndimage
import os
import pydicom
from skimage.measure import find_contours
import json
#conv = 1*np.load("tmpHU.npy")
#plt.imshow(conv)
#plt.show()

#kernel = np.outer(signal.gaussian(512,1), signal.gaussian(512,1))
#conv = ndimage.laplace(image)#signal.fftconvolve(image, kernel, mode='same')

#pwd = os.getcwd()
#os.chdir("DCM/HU")#
#dataset = pydicom.dcmread("HU.dcm")#"IM-0001-0028.dcm")#
#os.chdir(pwd)
#image = np.array(dataset.pixel_array)


filenumber = 0
ordered = []
if 0:
    for tfile in os.listdir("DCM/HU/T2_FLAIR_TE_38"):
        filenumber += 1
        #print(filenumber, tfile)
        number = int(tfile[len(tfile) - 1 - 5]) * 10 + int(tfile[len(tfile) - 1 - 4])
        ordered.append([filenumber, number])
    full = True
    counter = 1
    loop = 0
    while full:
        if ordered[loop][1] == counter:
            print(str(ordered[loop][0])+',',end="")
            counter += 1
            loop = 0
        else:
            loop += 1
        if counter == len(ordered) - 1:
            print()
            full = False

rightorder = [40,42,57,58,67,21,18,28,4,74,12,26,20,34,60,64,13,24,49,43,59,15,50,
            70,38,8,7,23,47,54,65,19,73,25,35,39,17,46,2,71,52,72,51,22,30,36,5,3,
            16,55,45,69,14,6,48,62,11,1,9,33,27,56,44,37,61,68,29,32,41,10,53,66]
filelist = []
segment = [8,7,23,47,54]
for file in os.listdir("DCM/HU/T2_FLAIR_TE_38"):
    filelist.append(file)

if 1:
    def savejson(x,y,z,number,islandnumber):
        pwd = os.getcwd()
        os.chdir("json")
        name = "f" + str(number)+ "i" + str(islandnumber)
        file = open(name+".json","w")

        coord = []
        for i in range(len(x1)):
            coord.append("["+str(x[i])+","+str(y[i])+","+str(z[i])+"],")
        text = {"ROI3DPoints" : []}
        text["ROI3DPoints"] = coord
        json.dump(text,file)
        os.chdir(pwd)
        return
    for index in rightorder:
        filenumber += 1
        if index in segment:
            file = filelist[index - 1]
            dataset = pydicom.dcmread("DCM/HU/T2_FLAIR_TE_38" + '/'+ file)
            conv = 1*np.load(str(index)+".npy")
            a=0.5
            kernel = np.outer(signal.gaussian(512,a), signal.gaussian(512,a))
            #conv = signal.fftconvolve(conv, kernel, mode='same')
            contours = find_contours(conv,0.5, 'high')
            lengths = np.array([0]*len(contours))
            for n, contour in enumerate(contours):
                if contour.shape[0] > 50:
                    x1,y1 = contour[:,1],contour[:,0]
                    z1 = [float(dataset.SliceThickness)]*(len(x1))
                    savejson(x1,y1,z1,filenumber,n)

if 0:

    for index in rightorder:
        filenumber += 1
        if index in segment:
            if index == rightorder[len(rightorder) - 1]:
                break
            file = filelist[index - 1]

            print(filenumber, file)
            dataset = pydicom.dcmread("DCM/HU/T2_FLAIR_TE_38" + '/'+ file)
            image = np.array(dataset.pixel_array)


            #plt.clf()

            fig, ax = plt.subplots(2, 2)
            ax[0,0].imshow(image, cmap=plt.cm.gray)
            #ax2.imshow(dataset.pixel_array, cmap=plt.cm.gray)
            ax[0,1].invert_yaxis()
            conv = 1*np.load(str(index)+".npy")
            a=0.5
            kernel = np.outer(signal.gaussian(512,a), signal.gaussian(512,a))
            #conv = signal.fftconvolve(conv, kernel, mode='same')
            contours = find_contours(conv,0.5, 'high')
            lengths = np.array([0]*len(contours))
            for n, contour in enumerate(contours):
                lengths[n] = contour.shape[0]
                if contour.shape[0] > 50:
                    #print(n,contour.shape[0])
                    ax[0,1].plot(contour[:,1],contour[:,0])
                    x1,y1 = contour[0,1],contour[0,0]

                    annot = ax[0,1].annotate(str(n), xy=(x1,y1), xytext=(x1,y1))
                    annot.set_visible(True)
            ax[0,0].axis('off')
            ax[0,0].set_aspect('equal')
            ax[0,1].axis('off')
            ax[0,1].set_aspect('equal')

            file = filelist[rightorder[filenumber] - 1]
            print(filenumber + 1, file)
            dataset = pydicom.dcmread("DCM/HU/T2_FLAIR_TE_38" + '/'+ file)
            image = np.array(dataset.pixel_array)


            #plt.clf()
            ax[1,0].imshow(image, cmap=plt.cm.gray)
            #ax2.imshow(dataset.pixel_array, cmap=plt.cm.gray)
            ax[1,1].invert_yaxis()
            conv = 1*np.load(str(rightorder[filenumber])+".npy")
            a=0.5
            kernel = np.outer(signal.gaussian(512,a), signal.gaussian(512,a))
            #conv = signal.fftconvolve(conv, kernel, mode='same')
            contours = find_contours(conv,0.5, 'high')
            lengths = np.array([0]*len(contours))
            for n, contour in enumerate(contours):
                lengths[n] = contour.shape[0]
                if contour.shape[0] > 50:
                    #print(n,contour.shape[0])
                    ax[1,1].plot(contour[:,1],contour[:,0])
                    x1,y1 = contour[0,1],contour[0,0]

                    annot = ax[1,1].annotate(str(n), xy=(x1,y1), xytext=(x1,y1))
                    annot.set_visible(True)
            ax[1,0].axis('off')
            ax[1,0].set_aspect('equal')
            ax[1,1].axis('off')
            ax[1,1].set_aspect('equal')



            #plt.box(on=None)
            #plt.subplots_adjust(wspace=0, hspace=0)
            #plt.clf()
            plt.show()

if 0:
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.box(on=None)
    plt.savefig("BestLBFsp2mu01dt001sigma10l101l1.png", bbox_inches= 'tight', pad_inches=0)
    plt.clf()
    plt.imshow(dataset.pixel_array, cmap=plt.cm.gray)
    plt.axis('off')
    plt.box(on=None)
    plt.savefig("Originalsp2.png", bbox_inches= 'tight', pad_inches=0)


#plt.savefig("BestLBFsp2.png", bbox_inches= 'tight', pad_inches=0)
