import numpy as np
from scipy import signal, ndimage, misc
from scipy.fftpack import fftn, ifftn
import nvidiaGPU
import time
import time
import matplotlib.pyplot as plt

def average(image, surface):
    Hsum = np.sum(surface)
    avarage = np.sum(image * surface)
    if Hsum != 0:
        avarage /= Hsum
    return avarage

def div(x,y):
    Px = np.pad(x, 1, mode='edge')
    Py = np.pad(y, 1, mode='edge')
    dx = (Px[1:-1, 2:] - Px[1:-1, :-2]) / 2.0
    dy = (Py[2:, 1:-1] - Py[:-2, 1:-1]) / 2.0

    return dx+dy

def grad(phi):
    ##creating a border to move all the pixels in dx and dy directions
    Extended = np.pad(phi, 1, mode='edge')
    ##P[1:-1,2:] is a shift to the left on the x axis
    ##P[1:-1, 1:-1] is the original phi
    ##ephiex and ephiey are simple discrete varations
    ephiex = (Extended[1:-1, 2:] - Extended[1:-1, :-2]) / 2.0
    ephiey = (Extended[2:, 1:-1] - Extended[:-2, 1:-1]) / 2.0
    return ephiex, ephiey

def delta(x, eps=1.):
    return eps / ((np.pi)*(eps**2 + x**2))

def xadrez(image, width):
    y = np.arange(image.shape[0]).reshape(image.shape[0], 1)
    x = np.arange(image.shape[1])
    return (np.sin(np.pi/width*x) * np.sin(np.pi/width*y))

def init_guess(Flag, image, mu, lambdas, dt):
    ##create a chess like board across all image
    ##as very simple and generic segmentation
    phi = xadrez(image, 10)

    phi2 = np.array([phi,-phi,phi])
    ## A good way to speed up the LBF segmentation is by giving a
    ##good enough first guess. This could be done in a number of
    ##different ways, but global Chan-Vese algorithm converges
    ##really quicklly and its 3 parameters are basically the same
    ##used in LBF.
    if Flag:
        for i in range(0,100):
            ##The discrete diferentiation and the divergent are better
            ##explained in the walk() function
            ephiex, ephiey = grad(phi)
            phixnorm = np.sign(ephiex)
            phiynorm = np.sign(ephiey)
            K =div(phixnorm,phiynorm)
            inside = 1 * (phi > 0)
            outside = 1 - inside
            c1 = average(image, outside)
            c2 = average(image, inside)
            phi = phi + 20* (-lambdas[0] * (image - c1) ** 2 + lambdas[1] * (image - c2) **2 + mu[0]*K) * delta(phi)
        ##normalization to avoid memory leak (specially when using GPU optimization)
        positive = phi * (phi>=0)
        negative = phi * (phi<0)
        phi = (positive/np.max(phi))-(negative/np.min(phi))

    phi2[0] = phi
    phi2[1] = -phi
    #phi2[2] = -phi
    return phi2

def LBF(fullkernel, image, upper, lower):
    f = np.divide(upper, lower, out=np.zeros(image.shape), where=lower!=0)
    return nvidiaGPU.E(fullkernel, image, f)

def LBF_procedural(fullkernel, image, M):
    upper = signal.fftconvolve(fullkernel,image*M,mode='same')
    lower = signal.fftconvolve(fullkernel,M,mode='same')
    f = np.divide(upper, lower, out=np.zeros(image.shape), where=lower!=0)
    ## K * I2 - 2I (f*K) + f2 * K
    a1 = signal.fftconvolve(fullkernel,image**2, mode="same")
    a2 = 2 * image * (signal.fftconvolve(fullkernel,f,mode="same"))
    a3 = signal.fftconvolve(fullkernel,f**2,mode="same")
    return a1 - a2 + a3

def Chanvese_subregion(image, region):
    value = average(image, region)
    return np.abs(image - value)

def Regularize(phi):
    ephiex, ephiey = grad(phi)
    ephiexnorm = np.sign(ephiex)
    ephieynorm = np.sign(ephiey)
    K = div(ephiexnorm,ephieynorm)
    return div(*grad(phi)) - K

def Length_penalty(phi):
    ephiex, ephiey = grad(phi)
    ephiexnorm = np.sign(ephiex)
    ephieynorm = np.sign(ephiey)
    return div(ephiexnorm,ephieynorm)

def getCombination(phi):
    def subregion(levels, i, j, phi):
        ##this will define if it is the inside or the outside of
        ##the phi level-set function
        curvetype = "{0:b}".format(i)
        for aux in range(levels - len(curvetype)):
            curvetype = "0" + curvetype
        if curvetype[j] == "0":
            return 1 * (phi > 0)
        else:
            return 1 * (phi < 0)

    levels = phi.shape[0]
    combinations = np.ones((phi[0].shape + (2**levels,)))
    for i in range(2**levels):
        for j in range(levels):
            combinations[:,:,i] *= subregion(levels, i, j, phi[j])
    return combinations

def ChooseCombination(levels,orientation,skip,phi):

    Region = np.ones(phi[0].shape)
    curvetype = "{0:b}".format(orientation)
    for i in range(levels - len(curvetype)):
        curvetype = "0" + curvetype

    if curvetype[skip] == "1":
        sign = -1
    else:
        sign = 1
    for i in range(levels):
        if i != skip:
            if curvetype[i] == "0":
                Region *= (1*phi[i]>0)
            else:
                Region *= (1*phi[i]<0)

    return sign*Region

def walk(theconv, fullkernel, image, phi, mu, lambdas, dt, ni, levels = 1):


    if False:
        #SPF model doi:10.1016/j.imavis.2009.10.009
        alfaSPF = 20
        inside = 1 * (phi > 0)
        outside = 1 - inside
        c1 = average(image, inside)
        c2 = average(image, ouside)
        spfIx = (image - (c1+c2)/2)/np.max(np.abs(image - (c1+c2)/2))
        phi = phi + dt * (spfIx * alfaSPF)
        phi = signal.fftconvolve(fullkernel,new_phi,mode='same')


    if False:
        #Local Binary Fit

        ## Note that div( grad(phi)/abs(grad(phi))) is simply
        ##the divergent of the signs of grad(phi). Taking the signs
        ##explicitly is safer than trying to calculated the absolute
        ##and performing a division, because computers make
        ##approximations on every iteration and one has to explicitly
        ##define what division by zero means in this context.
        ephiex, ephiey = grad(phi)
        ephiexnorm = np.sign(ephiex)
        ephieynorm = np.sign(ephiey)
        K = div(ephiexnorm,ephieynorm)
        Regu_term = div(ephiex,ephiey) - K
        inside = 1 * (phi > 0)
        outside = 1 - inside
        firstterm = signal.fftconvolve(fullkernel,image*inside,mode='same')
        secondterm = signal.fftconvolve(fullkernel,inside,mode='same')
        thirdterm = signal.fftconvolve(fullkernel,outside,mode='same')
        f1 = np.divide(firstterm, secondterm, out=np.zeros(firstterm.shape), where=secondterm!=0)
        f2 = np.divide((theconv-firstterm), thirdterm, out=np.zeros(theconv.shape), where=thirdterm!=0)
        time1 = time.time()
        E1 = nvidiaGPU.E(fullkernel, image, f1)
        E2 = nvidiaGPU.E(fullkernel, image, f2)
        time2 = time.time()
        print("Loop time: "+str(time2-time1))
        delta_phi = delta(phi)
        phi = phi + dt *(delta_phi * (-(lambdas[0]*E1)+(lambdas[1]*E2)) + ni * delta_phi * K + mu * Regu_term)

    if levels == 1:

        ## Note that div( grad(phi)/abs(grad(phi))) is simply
        ##the divergent of the signs of grad(phi). Taking the signs
        ##explicitly is safer than trying to calculated the absolute
        ##and performing a division, because computers make
        ##approximations on every iteration and one has to explicitly
        ##define what division by zero means in this context
        ephiex, ephiey = grad(phi)
        ephiexnorm = np.sign(ephiex)
        ephieynorm = np.sign(ephiey)
        K = div(ephiexnorm,ephieynorm)
        Regu_term = div(ephiex,ephiey) - K
        inside = 1 * (phi > 0)
        outside = 1 - inside
        w = 0.

        #LGIF
        time1 = time.time()
        ##LBF terms
        firstterm = signal.fftconvolve(fullkernel,image*inside,mode='same')
        secondterm = signal.fftconvolve(fullkernel,inside,mode='same')
        thirdterm = signal.fftconvolve(fullkernel,outside,mode='same')
        f1 = np.divide(firstterm, secondterm, out=np.zeros(firstterm.shape), where=secondterm!=0)
        f2 = np.divide((theconv-firstterm), thirdterm, out=np.zeros(theconv.shape), where=thirdterm!=0)
        E1 = nvidiaGPU.E(fullkernel, image, f1)
        E2 = nvidiaGPU.E(fullkernel, image, f2)

        ##Global Chan-Vese terms
        ##Note that the other Chan-Vese terms have alredy been calculated
        c1 = average(image, outside)
        c2 = average(image, inside)

        ##Organizing terms as in the main article
        F1 = -lambdas[0] * E1 + lambdas[1] * E2
        F2 = -lambdas[0] * np.abs(image - c1) + lambdas[1] * np.abs(image - c2)

        time2 = time.time()
        print("Loop time: "+str(time2-time1))
        delta_phi = delta(phi)
        phi = phi + dt *(delta_phi * ((1-w)*F1 + w*F2) + ni * delta_phi * Length_penalty(phi) + mu * Regularize(phi))


    if levels == 20:
        w = 0.
        #LGIF multi-levelset formulation with n=2 (4 subdivisions)
        time1 = time.time()
        ##LBF terms
        M1 = 1 * (phi[0] > 0) * (phi[1] > 0)
        M2 = 1 * (phi[0] > 0) * (phi[1] < 0)
        M3 = 1 * (phi[0] < 0) * (phi[1] > 0)
        M4 = 1 * (phi[0] < 0) * (phi[1] < 0)

        M = np.array([M1,M2,M3,M4])
        height = image.shape[0]
        width = image.shape[1]
        E = np.zeros((height,width,2**levels))
        cv = np.zeros((height,width,2**levels))
        smalle = np.zeros((height,width,2**levels))

        for i in range(2**levels):
            #firstterm = signal.fftconvolve(fullkernel,image*M[i],mode='same')
            #secondterm = signal.fftconvolve(fullkernel,M[i],mode='same')
            #E[:,:,i] = LBF(fullkernel, image, firstterm, secondterm)
            cv[:,:,i] = Chanvese_subregion(image, M[i])
            #smalle[:,:,i] = (1-w) * E[:,:,i] + w * cv[:,:,i]
            smalle[:,:,i] = cv[:,:,i]


        ##Organizing terms as in the main article
        time2 = time.time()
        print("Loop time for phi1: "+str(time2-time1))

        phi[0] += dt * (delta(phi[0]) * ( (phi[1]>0)*(lambdas[2]*smalle[:,:,2]-lambdas[0]*smalle[:,:,0] )+(phi[1]<0)*(lambdas[3]*smalle[:,:,3]-lambdas[1]*smalle[:,:,1]) )
                + ni * delta(phi[0]) * Length_penalty(phi[0])
                + mu[0] * Regularize(phi[0]))

        phi[1] += dt * (delta(phi[1]) * ( (phi[0]>0)*(lambdas[1]*smalle[:,:,1]-lambdas[0]*smalle[:,:,0] )+(phi[0]<0)*(lambdas[3]*smalle[:,:,3]-lambdas[2]*smalle[:,:,2]) )
                + ni * delta(phi[1]) * Length_penalty(phi[1])
                + mu[1] * Regularize(phi[1]))

    if levels >= 2:
        w = 0.0
        Regions = getCombination(phi)
        height = image.shape[0]
        width = image.shape[1]
        cv = np.zeros((height,width,2**levels))
        storedphi = np.zeros(phi.shape)
        for i in range(2**levels):
            cv[:,:,i] = (1-w) * Chanvese_subregion(image, Regions[:,:,i])# + w * LBF_procedural(fullkernel, image, Regions[:,:,i])

        for i in range(2**levels):
            for j in range(levels):
                storedphi[j] += lambdas[i]*cv[:,:,i]*ChooseCombination(levels,i,j,phi)
        for j in range(levels):
            storedphi[j] *= -delta(phi[j])
            storedphi[j] += (ni * delta(phi[j]) * Length_penalty(phi[j]) + mu[j] * Regularize(phi[j]))
            phi[j] += dt*storedphi[j]



    return phi

def ActiveContours(image, multilevel : int = 1, mu = 1., lambdas = [1.05,1.], sigma = 5, maxiter = 50, dt = 0.1, initguess = 'chanvese', gpuacc = 'y'):

    chanveselist = ["CHAN","VESE","CV","CHANVESE", "C_V", "CHAN_VESE", "C-V", "CHAN-VESE"]
    try:
        if len(image.shape) != 2:
            raise ValueError("Image should be 2D (and Black-and-White).")
    except ValueError:
        raise ValueError("Image should be converted into a 2D numpy array.")
    image = image - np.min(image)

    try:
        multilevel = int(multilevel)
    except:
        raise ValueError("The 'multilevel' parameter must be int or float type.")
    if not isinstance(mu, list):
        mu = np.array([mu], dtype = np.float64)
    else:
        mu = np.array(mu, dtype = np.float64)
    if not isinstance(lambdas, list):
        lambdas = np.array([lambdas], dtype = np.float64)
    else:
        lambdas = np.array(lambdas, dtype = np.float64)

    if 2**mu.shape[0] != lambdas.shape[0]:
        raise ValueError("Can't input different number of parameters for each curve.")

    if mu.shape[0] != multilevel or lambdas.shape[0] != 2**multilevel:
        raise ValueError("Can't input different number of parameters for each curve.")



    ##initial segmentation guess
    CVFlag = False
    if isinstance(initguess, str):
        for elem in chanveselist:
            if initguess.upper() in elem:
                CVFlag = True
                phi = init_guess(CVFlag, image, mu, lambdas, dt)
                break
            elif ".npy" in initguess:
                print("Loanding")
                try:
                    phi = 1*np.load(initguess)
                    break
                except:
                    print("Failed to find tmp.npy file.")
                    try:
                        print("Loading tmp.npy...")
                        phi = 1*np.load("tmp.npy")
                        break
                    except ValueError:
                        raise ValueError("Failed to find tmp.npy file.\nExiting now.")
        if CVFlag:
            phi = init_guess(CVFlag, image, mu, lambdas, dt)

    else:
        phi = initguess


    if not isinstance(phi, (np.ndarray, np.generic)):
        raise ValueError("Initial Numpy Array should be the same size of the image.")
    elif len(phi.shape) >= 3:
         if phi[0].shape != image.shape:
             raise ValueError("Initial Numpy Array should be the same size of the image.")
    elif phi.shape != image.shape:
        raise ValueError("Initial Numpy Array should be the same size of the image.")

    initialguess = phi > 0

    ##NOW THE CODE STARTS

    ## Kernel and the blurred initial image wont change so there
    ## is no need to calculate them on every iteration
    kernel = np.outer(signal.gaussian(image.shape[0], sigma), signal.gaussian(image.shape[1], sigma))
    conv = signal.fftconvolve(image, kernel, mode='same')
    for i in range(maxiter):
        print("Loop: "+str(i+1))
        phi = walk(conv, kernel, image, phi, mu, lambdas, dt, ni = 0., levels = multilevel )#*image.shape[0]*image.shape[1]
    if multilevel > 1:
        return getCombination(phi), phi, initialguess
    else:
        return phi>0, phi, initialguess
