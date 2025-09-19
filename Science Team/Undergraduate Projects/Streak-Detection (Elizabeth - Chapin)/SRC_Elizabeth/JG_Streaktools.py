#### HERE ARE ALL OF THE STREAKTOOLS I'VE COMPLETED UP TO AUG10 ## SLAY

import numpy as np
#from pyradon.streak import model
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy.signal import convolve2d,fftconvolve
from astropy import stats
from multiprocessing import Pool

from trippy import psf, pill, psfStarChooser,scamp,MCMCfit
from trippy_utils import *

#import psf, psfStarChooser   #### THESE ARE THE OPTIONAL TRIPPY PIECES
#import scamp,MCMCfit

import pylab as pyl
import imp
import os
import time
import sys

import emcee

import scipy as sci
from scipy import optimize as opti, interpolate as interp
from scipy import signal

def gen_TSF(section,xc,yc,psf,L,angle,OS = 3): ## gen_TSF is not an interface function - this works behind the scenes to generate a 
                                        ## trailed spread function (TSF) with a given centre, length, angle and PSF. The TSF
                                        ## returned is normalized to 1 (integrated sum). For accuracy the TSF is normally created
                                        ## with 3x oversampling (OS parameter), downsampled before use.
                
    '''
    PARAMETERS:
    
    'section': 2D array of pixel values. Provided so that the streak, which itself will be a 2D array, will have the same dimensions
                and can be added to a section
                
    'xc' & 'yc': centre offset of the streak. With both of these set to zero, the streak is centred in the centre of the output. Larger
                xc values move the streak down, larger yc values move the streak right
                
    'psf': either square 2D array of pixel values (odd sidelength!) or float. If an array, this is passed to adapted_model as the psf.
            if a float, then this is taken to be the sigma parameter in a gaussian psf
            
    'L': desired (unconvolved) streak length in pixels. Nonzero pixel values will be added farther out than this due to convolution
    
    'angle': clockwise angle of the streak, from horizontal, in radians.
    
    OPTIONAL:
    
    'OS': streak generation oversampling. Higher values will give a more accurate simulation of pixel values from a real streak, but will
        take longer to generate. This function is called many hundreds of times during emcee runs and therefore the default value of 3 was
        chosen, blending accuracy and computational time
        
    RETURNS:
    
    'TSF': normalized trailed spread function (TSF) with the same dimensions as the input section. Normalized to one.
    '''
    
    angle = np.pi/2 - angle ## corrects the angle so that it can be input with my convention
    
    if section.shape[0] == section.shape[1]:  ## Setup lengths Lsx and Lsy for a symmetric section
        
        Lsect = int(len(section))
        Lsx = Lsect
        Lsy = Lsect
        
    else: ## or for if the section is not symmetric
        
        Lsect = section.shape
        Lsy = Lsect[0]
        Lsx = Lsect[1]
    
    x1 = Lsx/2 - L/2 * np.sin(angle) + xc  ## set up x1 points in reference to the centre of the section and offsets xc,yc
    y1 = Lsy/2 - L/2 * np.cos(angle) + yc
    
    x2 = x1 + L * np.sin(angle) + xc ## As above but the x2 points
    y2 = y1 + L * np.cos(angle) + yc
        
    TSF = adapted_model(Lsect, x1, x2, y1, y2, psf_sigma = psf,
                                oversample = OS) ### generate the TSF using adapted model, with 3x oversampling to prevent aliasing

    TSF /= np.sum(TSF) ### Normalize the TSF to sum to one for easy amplitude fitting
    
    return TSF

def chisq(p,image,psf,limiter,basenoise): ### THIS IS THE VERSION OF CHISQ THAT ALLOWS FITTING THE BACKGROUND
    
    '''
    chisq is purely supposed to be an under-the-hood function. For a given parameter set (packed in 'p'), an image, a psf,
    a fitbound mask, and a 1sigma background noise in the image, returns the chi-squared value computed by comparing the model
    to the image being fit to. Computes the variance map under the assumption of poisson noise, with mean values (and therefore
    variance contribution) being the model value. Total variance is the base 1sigma noise squared, plus the model variance contribution
    '''
    
    x0,y0,f0,L0,angle,b = p # Unpack the state vector p
        
    L0 = int(L0) # fix the given length to be an integer
    
    tsf = gen_TSF(image,x0,y0,psf,L0,angle)
    
    #comparim = image[anchor[0]:anchor[0]+Lx,anchor[1]:anchor[1]+Ly]
    
    model = f0*tsf + b # 
    
    var = f0*tsf+basenoise**2 ## poisson!
    
    resid = limiter*(model-image)**2/var # this is the chisquared contribution from each cell (limiter is an array of ones and zeros specified by fitbound in emcee_fit()
                                        # that prevents features far beyond the streak body from being considered in chi-squared calculations
    C = np.sum(resid) # chi - squared calculated by summing the contributions from each cell (above)
    
    return C

def chisq_b(p,b,image,psf,limiter,basenoise): ## ALL AS IN CHISQ ABOVE BUT WITH A FIXED BACKGROUND (b) PASSED ALONG
    
    '''
    Exactly as in chisq, but with a fixed background ('b')
    '''
    
    x0,y0,f0,L0,angle = p
        
    L0 = int(L0)
    
    tsf = gen_TSF(image,x0,y0,psf,L0,angle)
    
    model = f0*tsf + b 
    
    var = f0*tsf+basenoise**2 ## poisson!
    
    resid = limiter*(model-image)**2/var
    
    C = np.sum(resid)
    
    return C

def lnprob(p,bounds,image,psf,limiter,basenoise,skysubbed,approx = False):  ## lnprob is the logarithm of the likelihood function (which is, up to a constant, exp(-chisquared/2)) 
    
    '''
    the log likelihood function for MCMC. Usually passes -chisquared/2, passes -infinity if a parameter is outside of bounds
    so as to confine the accessible parameter space, and returns a very large negative number (-1e20) if a NAN somehow shows up
    in order to prevent walkers from entering any weird spaces
    '''
    
                                                    ## this is passed to emcee for walker evolution 
    for i in range(len(p)):
        
        if (p[i] < bounds[i][0]) or (p[i] > bounds[i][1]):
            
            return -np.inf ### if a parameter violates a boundary, make sure it isn't accepted as a state (effectively restricts parameter space available)
    
    x0,y0,f0,L0,angle,b = p # Unpack the state vector p
        
    L0 = int(L0) # fix the given length to be an integer
    
    tsf = gen_TSF(image,x0,y0,psf,L0,angle)
    
    #comparim = image[anchor[0]:anchor[0]+Lx,anchor[1]:anchor[1]+Ly]
    
    model = f0*tsf + b
    
    expected = model
    observed = image
    
    if skysubbed == True:
    
        expected += basenoise**2
        observed += basenoise**2
        
    observed = observed.astype(int)
    
    if approx == False: ### compute the factorial
        
        resid = limiter*np.nan_to_num((observed*np.log(expected)-expected-np.log(sci.special.factorial(observed))),nan = -1e20)
        
        return np.sum(resid)
    
    elif approx == True: ### use Stirling's approximation
        
        resid = -limiter*np.nan_to_num((expected-observed)+observed*np.log(observed/expected),nan = 1e20)
        
        return np.sum(resid)

def lnprob_b(p,bounds,b,image,psf,limiter,basenoise,skysubbed,approx = False): ## AS WITH lnprob ABOVE BUT FOR FIXED BACKGROUND
    
    for i in range(len(p)):
        
        if (p[i] < bounds[i][0]) or (p[i] > bounds[i][1]):
            
            return -np.inf ### if a parameter violates a boundary, make sure it isn't accepted as a state (effectively restricts parameter space available)
    
    x0,y0,f0,L0,angle = p # Unpack the state vector p
        
    L0 = int(L0) # fix the given length to be an integer
    
    tsf = gen_TSF(image,x0,y0,psf,L0,angle)
    
    #comparim = image[anchor[0]:anchor[0]+Lx,anchor[1]:anchor[1]+Ly]
    
    model = f0*tsf + b
    
    expected = model
    observed = image
    
    if skysubbed == True:
    
        expected += basenoise**2
        observed += basenoise**2
        
    observed = observed.astype(int)
    
    if approx == False: ### compute the factorial
        
        resid = limiter*np.nan_to_num((observed*np.log(expected)-expected-np.log(sci.special.factorial(observed))),nan = -1e20)
        
        return np.sum(resid)
    
    elif approx == True: ### use Stirling's approximation
        
        resid = -limiter*np.nan_to_num((expected-observed)+observed*np.log(observed/expected),nan = 1e20)
        
        return np.sum(resid)

def chisq2(x0,y0,f0,L0,angle,b,image,psf,basenoise): ## chisq but without packaged parameter vector
    
    '''
    exactly as in chisq but without the packaged parameter vector
    '''
    
    #if np.isscalar(L0):
        
        #L0 = int(L0)
        
    #else:
        
        #for i in range(len(L0)):
            
            #L0[i] = int(L0[i])
            
    print(int(len(image)))
    
    tsf = gen_TSF(image,x0,y0,psf,L0,angle)
    
    #comparim = image[anchor[0]:anchor[0]+Lx,anchor[1]:anchor[1]+Ly]
    
    model = f0*tsf + b 
    
    var = model+basenoise**2 ## poisson!
    
    resid = (model-image)**2/var
    
    C = np.sum(resid)
    
    return C

def just_model(image,p,psf): ## Returns the model corresponding to a given set of parameters p as an array
    
    '''
    takes in a parameter vector p, image and psf and generates & returns a TSF corresponding to them using gen_TSF
    '''
    
    x0,y0,f0,L,angle,b = p
    
    tsf = gen_TSF(image,x0,y0,psf,L,angle)
    
    model = f0*tsf+b
    
    return model
    
def full_optimize(image,psf,p0,b,basenoise): 
    
    ## chi-squared minimization using scipy.optimize. This doesn't work well given the high dimensionality of the parameter
    ## space, but I got this sort of thing to work well enough for my senior thesis with a lot of babying. Fully recommend
    ## using emcee_fit over this
    
    '''
    Fits a model to an image of a streak to determine the best-fit parameter set by chi-squared minimization. Uses the Nelder-
    Mead algorithm. Almost always inferior to the MCMC approach since the high dimensional parameter space is potentially full of local
    minima.
    
    PARAMETERS:
    
    image: the 2D image section with a streak to be fit to
    
    psf: either the sigma value of a gaussian psf, or the 2D (odd sidelength!) array containing the psf to be used for fitting
    
    p0: best initial guess for the parameters, in the order (x,y,flux,length,angle,background)
    
    b: vector of how far from the guess each parameter is allowed to venture. For example, if p0 is (23,...)
        and b is (5,...) then the x parameter is initialized at 23 and is allowed between values of 18 and 28 (23 +/- 5)
        
    basenoise: The 1 sigma noise in the image. Contributes to the variance in chi-squared calculations. Important that this not be set to zero or                 near zero values! If it is, the algorithm will find everything unlikely since background pixels will be many, many sigma from                     likely values and therefore the algorithm be unlikely to converge to any solution
    
    RETURNS:
    
    u: vector of best-fit parameters
    
    best_model: output of the function just_model given the solution parameter set
    '''
    
    result = opti.minimize(chisq, x0 = p0, args = (image,psf,basenoise),method = 'Nelder-Mead',
                           bounds = ((p0[0]-b[0],p0[0]+b[0]),
                                     (p0[1]-b[1],p0[1]+b[1]),
                                     (p0[2]-b[2],p0[2]+b[2]),
                                     (p0[3]-b[3],p0[3]+b[3]),
                                     (p0[4]-b[4],p0[4]+b[4]),
                                     (p0[5]-b[5],p0[5]+b[5])))
    u = result.x
    #pcov = result.hess_inv
    
    chisq_1 = chisq(p0,image,psf,basenoise)
    chisq_2 = chisq(u,image,psf,basenoise)
    
    if chisq_2 < chisq_1:
        print(r'Result Chi-Squared is less than for the initial guess. Initial chi^2 = %.2f, chi^2 after = %.2f' %(chisq_1, chisq_2))
    else:
        print('Optimization failed')
        
    best_model = just_model(image,u,psf)
    
    return u,best_model

def trimCatalog(cat): ### function imported from Trippy to cut down returned star catalogs for psf fitting
    good=[]
    for i in range(len(cat['XWIN_IMAGE'])):
        try:
            a = int(cat['XWIN_IMAGE'][i])
            b = int(cat['YWIN_IMAGE'][i])
            m = int(cat['MAG_AUTO'][i])
        except: pass
        dist = np.sort(((cat['XWIN_IMAGE']-cat['XWIN_IMAGE'][i])**2+(cat['YWIN_IMAGE']-cat['YWIN_IMAGE'][i])**2)**0.5)
        d = dist[1]
        if cat['FLAGS'][i]==0 and d>30 and m<70000:
            good.append(i)
    good=np.array(good)
    outcat = {}
    for i in cat:
        outcat[i] = cat[i][good]
    return outcat

class streak_interface: ### THE PRIMARY CLASS OF THIS TOOLSET. PROVIDES A FRAMEWORK FOR STREAK ADDITION AND FITTING, ZEROPOINT COMPUTATION, ETC.
    
    '''
    Primary framework class in this toolbox. See documentation and tutorial for information and demonstration of use
    '''
    
    def __init__(self, image, extension = 0, comparim = None):
        
        if isinstance(image,str): ### If image is a string, then it is interpreted as an address for a fits image with a certain extension
        
            self.image = image
            self.ext = extension

            f = fits.open(self.image)

            if comparim != None:
                self.comparison = True
                g = fits.open(comparim)
                self.comparim = g[self.ext].data

            else:
                self.comparison = False

            self.imdata = f[self.ext].data
            self.current_image = self.imdata
            self.ext = extension

            f.close()
            
        elif isinstance(image,np.ndarray): ## if image is an array, it is interpreted as the image data to operate on directly
            
            self.imdata = image
            self.current_image = image
            self.ext = extension
            
            if comparim != None:
                self.comparison = True
                self.comparim = g[self.ext].data

            else:
                self.comparison = False
                
        ############### DEFAULT PARAMETERS #################
        ####################################################
        
        self.streak_list = []
        self.R = 6371e3 #m #RADIUS OF EARTH
        self.pixscale = 0.27 # asec/pixel
        self.Fzero = 3.727e-9 #W/m2 ## Computed using integrated r band response
        self.im_noise = 6.4 ## 1 sigma image noise
        self.exp_t = 30 # image exposure time, in seconds
        self.psf = 1.6 # sigma parameter for gaussian psf. Can be set to 2D array for different PSF shapes
        self.fwhm = 3.72 # fwhm of the psf profile. Needs to be changed with psf appropriately
        self.magzero = 28.88802 # True magnitude zeropoint of the associated image, to be used when simulating streaks to calculate counts
        self.VM = 30 # Limit of image display linearity. Set to 30, pixel values with a magnitude greater than 30 are set to +/- 30. Lower values
                        # aid dim contrast, higher values allow one to see differences between brighter sources
        
        #####################################################
        ####################################################
        
    def print_properties(self):
        
        '''
        Prints out all of the parameters and properties of the simulation. Shown in brackets next to the parameter is the 
        name of the property which needs to be referenced to change this parameter. For example, the flux zeropoint is read out
        as 'Flux zeropoint (Fzero)'. To change the flux zeropoint, I need to set (name of this interface instance).Fzero
        ''' 
            
        print('### INTERFACE PROPERTIES ###\n')
            
        print('Earth radius (R): %.1fm' %self.R)
        print('Pixel scale (pixscale): %.3fasec/pixel' %self.pixscale)
        print('Flux zeropoint (Fzero): %.3fnW/m^2' %(self.Fzero*1e9))
        print('Background 1sigma noise (im_noise): %.2f counts' %self.im_noise)
        print('Image exposure time (exp_t): %.1fs' %self.exp_t)
        try:
            print('PSF 1sigma width (psf): %.2f pixels' %self.psf)
        except:
            pass
        print('PSF FWHM (fwhm): %.2f pixels' %self.fwhm)
        print('Image magnitude zeropoint (magzero): %.5f' %self.magzero)
        print('display vmax (VM): %.1f\n' %self.VM)
        
    def section(self,streak,mode = 'smallbox',visout = True):
        
        '''
        Cuts out a section of the current image with minimal area beyond the streak. This allows MCMC fitting to be done faster.
        The simulated or real streak instance must be passed. There are also two optional parameters:
        
        mode: 'smallbox' by default. Makes a small box around the streak which will have a different shape depending on the orientation
        of the streak. A mostly vertical streak will have a tall skinny section, etc. Can also be set to 'twomatch', which will create a square
        section with a sidelength which is the smallest possible power of 2 (can help computation)
        
        visout: True by default. If false, hides the visible, visual output that shows the section created
        '''
        
        if streak.in_image == False:
            
            print('Streak is not currently in image!')
            
        else:
            
            if mode == 'twomatch':
            
                L = 2

                while L < streak.L:

                    L *= 2

                print(L)
                
                self.sect = self.current_image[int(streak.cheatx-L/2):int(streak.cheatx+L/2),
                                               int(streak.cheaty-L/2):int(streak.cheaty+L/2)]
                
            elif mode == 'smallbox':
                
                Lx = int(streak.L * np.abs(np.sin(streak.theta))) + 0.1*streak.L + 30
                Ly = int(streak.L * np.cos(streak.theta)) + 0.1*streak.L + 30
                
                print(Lx,Ly)
        
                self.sect = self.current_image[int(streak.cheatx-Lx/2):int(streak.cheatx+Lx/2),
                                               int(streak.cheaty-Ly/2):int(streak.cheaty+Ly/2)]
            
            if visout == True:
            
                print('Section created (called .sect())')

                plt.imshow(self.sect,cmap = 'Greys',vmax = self.VM,vmin = -self.VM)
    
    def add_streak(self,streak,visout = True):
        
        '''
        Behind-the-scenes function which is used when the .add() method of a streak is called
        '''
        
        d = streak.dist
        R = self.R
        
        if streak.theta_override == None:
            
            r_star = np.sqrt((R/d)**2+2*(R/d)*np.sin(streak.alpha)+1)
            nadir = np.pi/2 - streak.alpha - np.arcsin(np.cos(streak.alpha)/r_star)
            
        else: 
            
            nadir = streak.theta_override
            
        if visout == True:
            
            print('NADIR angle: %.2f degrees' %np.rad2deg(nadir))
        
        streak.nadirangle = nadir
        I = streak.IS_rad*streak.A_IS*np.cos(nadir)/d**2
        m = -2.5*np.log10(I*streak.delta_t*streak.S_of_lambda/self.exp_t/self.Fzero)
        streak.totalmag = m
        
        if visout == True:
            
            print('Streak total magnitude: %.3f' %m)
            
        counts = 10**(2/5*(self.magzero - m))
        
        cpul = 10**(2/5*(self.magzero - m))/streak.L
        
        if visout == True:
            
            print('CPUL: %.3f' %cpul)
            #print('CALCULATED MAGZERO WITH PERFECT FITTING: %.5f' %(2.5*np.log10(counts) + m))
        
        streak.cpul = cpul
        
        IS = 2

        while (IS < (streak.L + 20)):

            IS *= 2

        if visout == True:
            
            print('Required streakmodel sizescale: %d' %IS)
        
        self.im_size = IS
        streak.x2 = streak.x1 + streak.L * np.sin(streak.theta)
        streak.y2 = streak.y1 + streak.L* np.cos(streak.theta)
        streak.cheatx = (streak.x1 + streak.x2)/2
        streak.cheaty = (streak.y1 + streak.y2)/2
        
        self.newdata = adapted_model(self.im_size, 10, streak.L * np.cos(streak.theta) + 10, 10, streak.L * np.abs(np.sin(streak.theta)) + 10, psf_sigma = self.psf,
                                    oversample = 3)
        
        self.newdata /= np.sum(self.newdata)
        
        self.ndata = self.newdata*counts/np.sum(self.newdata)
        
        if visout == True:
            
            print('raw counts = %d' %np.sum(self.ndata))
        
        streak.gen_counts = np.sum(self.ndata)
        var = np.abs(self.ndata)
        
        self.newdata = np.random.poisson(var)
        
        if visout == True:
            
            print('counts with noise = %d' %np.sum(self.newdata))
        
        streak.poiss_counts = np.sum(self.newdata)
        
        if streak.theta >= 0 and streak.theta <= np.pi/2:
        
            self.current_image,streak.map = padmatch2(self.current_image,self.newdata,streak.x1-10,streak.y1-10)
            
        elif streak.theta < 0 and streak.theta >= -np.pi/2:
            
            self.current_image,streak.map = padmatch2(self.current_image[::-1],self.newdata,len(self.current_image) - streak.x1-10,streak.y1-10)
            
            self.current_image = self.current_image[::-1]
            
            streak.map = streak.map[::-1]
        
        if visout == True:
            
            print('Streak added successfully')
        
        
    def clear_streaks(self):
        
        '''
        This method clears all streaks from the image. It does this by restoring the original image data. And setting all streaks' in_image
        parameters to False. 
        '''
        
        self.current_image = np.copy(self.imdata)
        
        for i in self.streak_list:
            
            if i.in_image == True:
                
                i.in_image = False
        
        print('Streaks removed')
        
    def display(self):
        
        '''
        This method displays the current image data, linearly, limited at +/- VM. 
        '''
        
        plt.figure(figsize = (12,12))
        plt.imshow(self.current_image, vmax = self.VM, vmin = -self.VM,cmap = 'Greys')
        plt.title('Current Interface State')
        plt.xlabel('---Increasing y--->')
        plt.ylabel('<---Increasing x---')
        
    def gen_trippy_psf(self,baseim,config,catname,param_file,scale):
        
        '''
        This method generates a PSF for the image using source extractor and TRIPPy. If sextractor/trippy are unavailable to you,
        you may want to comment this function and its dependencies, MCMCfit, scamp, psf and psfStarChooser, out. Arguements are, in order,
        the directory of the image from which to generate the psf (uses the same extension! If you're working with a source-subtracted image
        then the extension of the non-source subtracted image should be the same!), the address/name of the sextractor config file to use,
        the name of the output catalog to be generated, the address/name of the sextractor parameter file to use, and finally the 'scale' of the psf
        to be generated, which will be the psf array sidelength. This should be large enough to encompass the psf structure. SCALE MUST BE AN
        ODD NUMBER!
        '''
        
        v = fits.open(baseim)
        self.source_data = v[self.ext].data
        newhdul = fits.PrimaryHDU(self.source_data)
        hdul = fits.HDUList([newhdul])
        hdul.writeto('TEMP.fits',overwrite = True)

        scamp.runSex(config, 'TEMP.fits',options={'CATALOG_NAME':catname},verbose=False)
        catalog = trimCatalog(scamp.getCatalog(catname,paramFile=param_file))
        
        starChooser=psfStarChooser.starChooser(self.source_data,
                                    catalog['XWIN_IMAGE'],catalog['YWIN_IMAGE'],
                                    catalog['FLUX_AUTO'],catalog['FLUXERR_AUTO'])
        
        (goodFits,goodMeds,goodSTDs) = starChooser(30,200,noVisualSelection=True,autoTrim=True, 
                                                   bgRadius=15, quickFit = False,
                                                   repFact = 10, ftol=1.49012e-08)
        
        self.goodPSF = psf.modelPSF(np.arange(scale),np.arange(scale), alpha=goodMeds[2],beta=goodMeds[3],repFact=10)
        self.goodPSF.genLookupTable(self.source_data,goodFits[:,4],goodFits[:,5],verbose=False)
        self.moff_fwhm = self.goodPSF.FWHM(fromMoffatProfile=True) ###this is the pure moffat FWHM.
        self.goodPSF.genPSF()
        self.psf = downSample2d(self.goodPSF.moffProf,self.goodPSF.repFact)
        
        #self.psf = self.goodPSF.dsPSF

        print("Full width at half maximum {:5.3f} (in pix).".format(self.fwhm))
        
        #self.goodPSF.computeLineAperCorrFromTSF(np.array([1,2,3,4,5])*self.fwhm,
                                                         #l=streak.L,a=-90+np.rad2deg(streak.theta),display=False,displayAperture=False)
        #lineAperCorr = self.goodPSF.lineAperCorr(2*self.fwhm)
        #print(lineAperCorr,roundAperCorr)
        
    def chisquarepsffit(self,section,p0,b0,psf,basenoise = 6.4):
        
        '''
        Attempt to fit a model to a streak which has been carved into a section using streaksim.section(), using chi-squared minimization
        (specifically the Nelder-Mead algorithm). 
        
        Arguments are
        
        ~the section, the tuple of best parameter guesses in order (x,y,flux,length,angle,background) where angle is measured
        CW from horizontal and x and y are in reference to the centre of the section 
        
        ~the parameter bounds as another tuple of the same format with bound magnitudes (ex an x bound of 30 will prohibit 
        exploring the parameter space beyond +/- 30 of the initial guess)
        
        ~the streak psf
        
        ~basenoise, the skynoise value. Defaults to 6.4, optional
        
        Note that this approach is inefficient and highly susceptible to finding local chi-squared minima as opposed to global minima. 
        Highly suggest the use of emcee_fit instead
        '''
        
        bestfit,bestmodel = full_optimize(section,psf,p0,b0,basenoise)
        
        fig,axs = plt.subplots(1,2,figsize = (16,8))
        
        axs[0].imshow(section,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        axs[1].imshow(bestmodel,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        
        print('Success! (maybe?)')
        print(bestfit)
        
    def chitest(self,section,p0,psf):
        
        '''
        This is a utility method to compare a streak generated from an initial guess with the real streak, and observe their differences.
        This is hopefully useful when trying to establish best-guess parameters. Arguements are:
        
        ~The streak section (generated using streaksim.section())
        
        ~The 6-tuple of parameters to use to generate the simulated streak (x,y,flux,length,angle,background)
        
        ~The PSF to use
        '''
        
        bestmodel = just_model(section,p0,psf)
        
        chisquared = chisq(p0,section,psf,basenoise = 6.4)
    
        fig,axs = plt.subplots(1,3,figsize = (24,8))
        
        axs[0].imshow(section,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        axs[0].set_title('Section')
        axs[1].imshow(bestmodel,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        axs[1].set_title('Model')
        axs[2].imshow(section-bestmodel,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        axs[2].set_title('Section - Model')
        
        plt.figure(figsize = (12,12))
        plt.imshow(bestmodel[100:-100,100:-100],vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        plt.title('Model')
        
        plt.figure(figsize = (12,12))
        plt.imshow((section - bestmodel)[100:-100,100:-100],vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        plt.title('Section - Model')
        
        print('Chi Squared = %d' %chisquared)
    
        print('Success! (maybe?)')
        
    def rawline(self,section,p0):
        
        '''
        Compare a streak section to an unconvolved line. May be easier to align angle and length than with the convolved streak (see streaksim.chitest())
        
        Arguements are:
        
        ~The streak section (generated using streaksim.section())
        
        ~The 6-tuple of parameters to use to generate the simulated streak (x,y,flux,length,angle,background)
        '''
        
        bestmodel = adapted_model(section,p0)
    
        fig,axs = plt.subplots(1,2,figsize = (24,8))
        
        axs[0].imshow(section,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        axs[0].set_title('Section')
        axs[1].imshow(bestmodel,vmax = self.VM,vmin = -self.VM,cmap = 'Greys')
        axs[1].set_title('Model')
        
        
        print('Success! (maybe?)')
        
    def simpill(self,xc,yc,r,L,angle,streak,visout = True):
        
        '''
        This method performs pill-aperture photometry on a streak. Parameters are as described in TRIPPy pill photometry, but are implemented outside
        of TRIPPy here. The function will use interface and streak parameters to calculate the zero point of the image using flux measurements.
        
        Parameters are:
        
        xc: x location (downwards from top!!) of the pill centroid. Suggest using streak.cheatx
        
        yc: As with xc, but for y (right from left edge!!) Suggest using streak.cheaty
        
        r: radial size of the pill. This is the radius of the circular ends, and is half the width of the rectangular section. Suggest using several times
            streaksim.fwhm
            
        L: Length of the rectangular portion of the streak (total length of pill along longest axis is L + 2*r)
        
        angle: The angle, CW from horizontal, of the pill (Radians).
        
        streak: The streak object that is being measured. This is specified so that magnitude zeropoints can be calculated. 
        '''
        
        cosan = np.cos(np.pi/2 - angle) ## Evaluate these once so that cosines and sines aren't being called continuously.
        sinan = -np.sin(np.pi/2 - angle) ## Using pi/2 - angle given to match rest of angle convention in streaktools
        
        def markercheck_(x,y,xc,yc,r,L,angle): ## Markercheck checks to see whether the criteria to be in the 'pill' are satisfied for each
                                                ## image cell, and marks those that satisfy with the value 1, returning 0 for those that don't
            xp = (y-xc)*cosan-(x-yc)*sinan
            yp = (x-yc)*cosan+(y-xc)*sinan
            
            if (np.abs(yp) <= r):
                
                if (np.abs(xp) <= L/2):
                
                    return 1
            
                elif (xp < -L/2):
                    
                    if np.sqrt(yp**2+(xp+L/2)**2) <= r:
                        
                        return 1
                    
                    else: 
                        
                        return 0
                    
                elif (xp > L/2):
                    
                    if np.sqrt(yp**2+(xp-L/2)**2) <= r:
                        
                        return 1
                    
                    else:
                        
                        return 0
                    
                else: 
                    
                    return 0
                
            else:
                
                return 0
            
        markercheck = np.vectorize(markercheck_)
        
        def checkedge(marker): ### This inner function identifies edge pixels by whether or not they have neighboring zero entries in the markercheck array
            
            edge = marker-1+1 ## copy rather than reference
            
            for i in range(len(marker)):
                for j in range(len(marker[0])):
                    
                    if marker[i][j] == 1:
                        
                        if marker[i-1][j] == 1 and marker[i+1][j] == 1 and marker[i][j-1] == 1 and marker[i][j+1] == 1:
                            
                            edge[i][j] = 0
                            
            edgex = []
            edgey = []
                            
            for i in range(len(marker)):
                for j in range(len(marker[0])):
                    
                    if edge[i][j] == 1:
                        
                        edgex.append(j)
                        edgey.append(i)
                            
            return edgex,edgey
        
        x_s = np.linspace(0,len(self.current_image[0])-1,len(self.current_image[0]))
        y_s = np.linspace(0,len(self.current_image)-1,len(self.current_image))
        
        xx,yy = np.meshgrid(x_s,y_s)
        
        self.pill = markercheck(xx,yy,xc,yc,r,L,angle)
        N = np.sum(self.pill) ## Number of points
        print('%d points in pill aperture' %N)
        
        edgex,edgey = checkedge(self.pill)
        
        if visout == True:
        
            fig,axs = plt.subplots(1,2,figsize = (14,7))
            axs[0].imshow(self.current_image,vmax = self.VM,vmin = -self.VM, cmap = 'Greys')
            axs[0].scatter(edgex,edgey,c = 'r',s = 0.02,alpha = 0.8)
            axs[1].imshow(self.current_image,vmax = self.VM,vmin = -self.VM, cmap = 'Greys')
            axs[1].set_xlim(yc-1.1/2*L,yc+1.1/2*L)
            axs[1].set_ylim(xc+1.1/2*L,xc-1.1/2*L)
            axs[1].scatter(edgex,edgey,c = 'r',s = 50/L,alpha = 0.8,zorder = 5)
            
        flux = np.sum(self.pill*self.current_image)
        
        self.pill_flux = (np.sum(self.pill*self.current_image),np.sqrt(np.sum(self.pill*self.current_image)+N*self.im_noise**2))
        
        if self.comparison == True:
            
            if len(self.pill) > len(self.comparim):
            
                self.pill = self.pill[:len(self.comparim),:]
            
            elif len(self.comparim) > len(self.pill):
            
                self.comparim = self.comparim[:len(self.pill),:]
            
            compar_flux = np.sum(self.pill*self.comparim)
            
        streak.pill_flux = (flux,np.sqrt(flux + N*self.im_noise**2))
        
        if self.comparison == True:
            
            streak.pill_comparflux = (flux-compar_flux,np.sqrt(flux+compar_flux+2*N*self.im_noise**2))
            
            streak.pill_comparmagzero = (streak.totalmag+2.5*np.log10(flux-compar_flux),2.5/np.log(10)*np.sqrt(flux+compar_flux+2*N*self.im_noise**2)/(flux-compar_flux))
            
        streak.pill_magzero = (streak.totalmag+2.5*np.log10(flux),2.5/np.log(10)*np.sqrt(flux+N*self.im_noise**2)/flux)
            
        if visout == True:
        
            print('Total Flux: %d +- %d' %(self.pill_flux[0],self.pill_flux[1]))
            print('Calculated Magnitude Zeropoint: %.4f +- %.4f' %(streak.pill_magzero[0],streak.pill_magzero[1]))
            if self.comparison == True:
                print('Calculated Zeropoint with comparison subtracted: %.4f' %(streak.totalmag+2.5*np.log10(flux-compar_flux)))
            print('True Zeropoint: %.4f' %(self.magzero))
            
    def emcee_fit(self,streak,section,psf,bestguess,bounds,basenoise = 6.4,nwalk = 20,nburn = 200,niter = 500,walkerreport = False,fitbound = None,visout = True,fixed_b = None,SM = 2.5,skysubbed = True,approx = False):
        '''
        emcee_fit uses MCMC algorithms from the package 'emcee' to fit a streak model to data. This is a much better approach than the raw chi-           squared minimization, as the 6 dimensional parameter space and noisy images mean that there are potentially a great number of local chi-         squared minima. MCMC fitting is well suited to high-dimensional optimization and also lends itself to good estimates of parameter                 certainty. The many arguements are as follows:
        
        streak: the streak object being fit to
        
        section: the section surrounding the streak (generated using streaksim.section()) to use
        
        psf: the point spread function of the image. A scalar indicates the sigma value of a Gaussian kernel, a 2D array specifies the exact psf
        
        bestguess: the 6-tuple of parameters to use to generate the simulated streak (x,y,flux,length,angle,background)
        
        bounds: The spread of each parameter to use when initializing the walkers. Should be as tightly bound as possible within confidence. Ex if 
                bestguess is (72,...) and bounds is (10,...) then x values will be initialized in the range (62,82)
                
        basenoise (default 6.4): the 1 sigma sky noise (used to compute uncertainties)
        
        nwalk (default 20): The number of walkers. Should be at least twice the number of dimensions (12, for us) or emcee will complain! Higher
                            numbers are better since walkers influence eachother and help shepherd one another, but increase computation time
                            
        nburn (default 200): The number of initial steps to 'burn', or ignore for statistics, since the walker ensemble is still assumed to be
                                converging to the posterior distribution from initial setup
                                
        niter (default 500): The number of steps to take after the burn-in period. This should not start before the ensemble has converged. A
                            higher number gives better statistics, but will take longer to run.
                            
        walkerreport (default False): Boolean; specifies whether or not to print the initial walker setup. Mostly for diagnostic purposes
        
        visout (default True): Print and show output? You'll want this when using this in a notebook, not so much in a script with many streaks
        
        fixed_b: Optional fixed background parameter. If this is set, then bounds and bestguess should have 5 entries, not 6
        
        SM (default 2.5): This is the sigma clipping threshold to be used when computing best fit parameters. Be very cautious when setting this                            to 2 or lower.
        '''
        
        def markercheck_(x,y,xc,yc,r,L,angle): ## Markercheck checks to see whether the criteria to be in the 'pill' are satisfied for each
                                                ## image cell, and marks those that satisfy with the value 1, returning 0 for those that don't
                
            cosan = np.cos(np.pi/2 - angle)
            sinan = -np.sin(np.pi/2 - angle)
            
            xp = (y-xc)*cosan-(x-yc)*sinan ### xp and yp are rotated coordinates - xp is along the length of the pill, yp is perpendicular
            yp = (x-yc)*cosan+(y-xc)*sinan
            
            if (np.abs(yp) <= r): ## Check for a point whether it falls within the yp limits of the pill rectangle
                
                if (np.abs(xp) <= L/2): ## Check for a point if its xp coordinate is within the length of the pill
                
                    return 1
            
                elif (xp < -L/2):  ## If not, check if within end of the pill. This is only possible if the yp criteria are met
                    
                    if np.sqrt(yp**2+(xp+L/2)**2) <= r:   ## condition to be in the circular ends of the pill
                        
                        return 1   
                    
                    else: 
                        
                        return 0
                    
                elif (xp > L/2):
                    
                    if np.sqrt(yp**2+(xp-L/2)**2) <= r:  ## or the other end
                        
                        return 1
                    
                    else:
                        
                        return 0
                    
                else: 
                    
                    return 0       # <----- Return zeros everywhere the pill criteria are not met, and 1 where they are
                
            else:
                
                return 0
             
        markercheck = np.vectorize(markercheck_) # < --- vectorize the above function for efficiency
        
        def checkedge(marker): ### This inner function identifies edge pixels by whether or not they have neighboring zero entries in the markercheck array
            
            edge = marker-1+1 ## copy rather than reference this array
            
            for i in range(len(marker)):
                for j in range(len(marker[0])):
                    
                    if marker[i][j] == 1:
                        
                        if marker[i-1][j] == 1 and marker[i+1][j] == 1 and marker[i][j-1] == 1 and marker[i][j+1] == 1:   ## if the marker is one, and ALL neighbors are,
                                                                                                                            ## then this cannot be an edge pixel and return 0
                            edge[i][j] = 0
                            
            edgex = []
            edgey = []
                            
            for i in range(len(marker)):
                for j in range(len(marker[0])):
                    
                    if edge[i][j] == 1:
                        
                        edgex.append(j)  ## make arrays of the x and y points of the edge pixels to be plotted for visualization later
                        edgey.append(i)
                            
            return edgex,edgey
        
        x_s = np.linspace(0,len(section[0])-1,len(section[0]))   ### make arrays corresponding to image position indices
        y_s = np.linspace(0,len(section)-1,len(section))
        
        xx,yy = np.meshgrid(x_s,y_s) ## Create the meshgrid for the points to run markercheck and checkedge on
        
        if fitbound != None:  ## If fitbound is specified, then the array 'limiter', which will multiply the returned chi^2 values, is set equal to 1 inside this region
                                ## and equal to zero outside
        
            limiter = markercheck(xx,yy,bestguess[0]+len(section)/2,bestguess[1]+len(section[0])/2,fitbound,bestguess[3],bestguess[4])
            
            edgex,edgey = checkedge(limiter)
            
        else: 
            
            limiter = section*0+1 ## Return all ones for no fitbound, so that chi-squared contributions from all across the image are considered
    
        dim = len(bestguess) ## dimension of the parameter space
        
        global start_time
        
        start_time = time.time()  ## record the time the emcee fitting starts
        
        if visout == True:
        
            fig,axs = plt.subplots(1,2,figsize = (14,7))
            axs[0].imshow(section,vmax = self.VM,vmin = -self.VM, cmap = 'Greys')
            if fitbound != None:
                axs[0].scatter(edgex,edgey,c = 'r',s = 50/len(section[0]),alpha = 0.8,zorder = 5)
            axs[0].set_title('Section given')
            axs[0].grid()
            axs[1].imshow(bestguess[2]*gen_TSF(section,bestguess[0],bestguess[1],psf,bestguess[3],bestguess[4]),vmax = self.VM,vmin = -self.VM, cmap = 'Greys')
            axs[1].set_title('Generated from initial parameters')
            axs[1].grid()
            plt.show()
            
        boundary = []
        
        for i in range(len(bestguess)):
            
            boundary.append((bestguess[i] - bounds[i], bestguess[i] + bounds[i]))
            
        start_time = time.time()
            
        with Pool() as pool:
            
            if fixed_b == None:

                sampler = emcee.EnsembleSampler(nwalk,dim,lnprob,args = [boundary,section,psf,limiter,basenoise,skysubbed,approx],pool = pool)

            else:

                sampler = emcee.EnsembleSampler(nwalk,dim,lnprob_b,args = [boundary,fixed_b,section,psf,limiter,basenoise,skysubbed,approx],pool = pool)

            r0 = []

            for ii in range(nwalk):
                walker = []
                for j in range(len(bestguess)):
                    walker.append(bestguess[j] + (2*np.random.rand()-1)*bounds[j])
                r0.append(walker)

            r0 = np.array(r0)

            if walkerreport == True:

                for i in range(len(r0)):
                    if fixed_b == None:
                        print('Walker %d: x = %.2f, y = %.2f, f = %.2f, L = %.2f, angle = %.2f, b = %.2f'
                                  %(i,r0[i][0],r0[i][1],r0[i][2],r0[i][3],np.rad2deg(r0[i][4]),r0[i][5]))
                    elif fixed_b != None:
                        print('Walker %d: x = %.2f, y = %.2f, f = %.2f, L = %.2f, angle = %.2f, b = %.2f'
                                  %(i,r0[i][0],r0[i][1],r0[i][2],r0[i][3],np.rad2deg(r0[i][4]),fixed_b))


            print("Performing burn-in...")
            
            pos_burn, prob_burn, state_burn = sampler.run_mcmc(r0, nburn, progress = True)
            time.sleep(0.1)
            print("Burn in complete. Elapsed time: %.1fs                                                             " %(time.time()-start_time)) #THESE SPACES ARE INTENTIONAL!
            start_time = time.time()
            samps_burn = sampler.chain
            sampler.reset()
            print("Performing main run...")
            pos, prob, state = sampler.run_mcmc(pos_burn, niter, rstate0=state_burn, progress = True)
            time.sleep(0.1)
            print("Finished. Elapsed time: %.1fs                                                           " %(time.time()-start_time)) #SPACES ARE INTENTIONAL!

        samps = sampler.chain
        probs = sampler.lnprobability

        out = (pos,prob,state,samps,probs)

        streak.emcee_out = out

        print('Complete!')

        positions = []

        for j in range(len(bestguess)):
            u = []
            for i in range(nwalk):
                y = list(out[3][i][:,j])
                u = u + y
            positions.append(u)
            
        positions = np.array(positions)
        
        ############## THE FOLLOWING CODE DISPLAYS THE WALKER TRACES IF VISOUT IS TRUE ####################
        
        if visout == True:
        
            fig,axs = plt.subplots(6,1,figsize = (9,12))

            Titles = ['X','Y','FLUX','LENGTH','ANGLE','BACKGROUND']

            for i in range(len(bestguess)):

                for j in range(nwalk):

                    full_state = list(samps_burn[j][:,i]) + list(samps[j][:,i])

                    axs[i].plot(full_state,zorder = 0)
                    axs[i].plot(samps_burn[j][:,i],zorder = 1,color = 'grey')


                axs[i].grid()
                axs[i].axvline(nburn)
                axs[i].set_title(Titles[i])

            for i in range(5):
                axs[i].set_xticklabels([])

            axs[5].set_xlabel('Count number')

            plt.plot()
        
        ###################################################################################################
        
        ############### BELOW CODE ACTUALLY MAKES MEASUREMENTS FROM THE WALKER DISTRIBUTIONS ##############
        
        mes_x = stats.sigma_clipped_stats(positions[0],sigma = SM)
        mes_y = stats.sigma_clipped_stats(positions[1],sigma = SM) #### <------ Computing the measured parameter mean, median and stdev
        mes_f = stats.sigma_clipped_stats(positions[2],sigma = SM)
        mes_L = stats.sigma_clipped_stats(positions[3],sigma = SM)
        mes_theta = stats.sigma_clipped_stats(positions[4],sigma = SM)
                          
        if fixed_b == None:
            mes_b = stats.sigma_clipped_stats(positions[5],sigma = SM)
        else:
            mes_b = (fixed_b,fixed_b,0)
        
        ############ CODE BELOW SHOWS DISTRIBUTION OF WALKER POSITIONS FROM AFTER THE BURN IN PERIOD, MEAURED VALUES AND 1SIG ENVELOPES ####
        
        if visout == True:

            fig,axs = plt.subplots(2,3,figsize = (9,6))

            axs[0][0].hist(positions[0],bins = 50,lw = 1,edgecolor = 'black')
            axs[0][1].hist(positions[1],bins = 50,lw = 1,edgecolor = 'black')
            axs[0][2].hist(positions[2],bins = 50,lw = 1,edgecolor = 'black')
            axs[1][0].hist(positions[3],bins = 50,lw = 1,edgecolor = 'black')
            axs[1][1].hist(positions[4],bins = 50,lw = 1,edgecolor = 'black')

            axs[0][0].axvspan(mes_x[0]-mes_x[2],mes_x[0]+mes_x[2], alpha=0.5, color='red',hatch = '/')
            axs[0][1].axvspan(mes_y[0]-mes_y[2],mes_y[0]+mes_y[2], alpha=0.5, color='purple',hatch = '/')
            axs[0][2].axvspan(mes_f[0]-mes_f[2],mes_f[0]+mes_f[2], alpha=0.5, color='green',hatch = '/')
            axs[1][0].axvspan(mes_L[0]-mes_L[2],mes_L[0]+mes_L[2], alpha=0.5, color='orange',hatch = '/')
            axs[1][1].axvspan(mes_theta[0]-mes_theta[2],mes_theta[0]+mes_theta[2], alpha=0.5, color='lime',hatch = '/')

            axs[0][0].axvline(mes_x[0], alpha=1, color='red',lw = 2, ls = '-.')
            axs[0][1].axvline(mes_y[0], alpha=1, color='purple',lw = 2, ls = '-.')
            axs[0][2].axvline(mes_f[0], alpha=1, color='green',lw = 2, ls = '-.')
            axs[1][0].axvline(mes_L[0], alpha=1, color='orange',lw = 2, ls = '-.')
            axs[1][1].axvline(mes_theta[0], alpha=1, color='lime',lw = 2, ls = '-.')

            axs[0][0].set_title('x = %.2f $\pm$ %.2f' %(mes_x[0],mes_x[2]))
            axs[0][1].set_title('y = %.2f $\pm$ %.2f' %(mes_y[0],mes_y[2]))
            axs[0][2].set_title('flux = %.2f $\pm$ %.2f' %(mes_f[0],mes_f[2]))
            axs[1][0].set_title('length = %.2f $\pm$ %.2f' %(mes_L[0],mes_L[2]))
            axs[1][1].set_title('angle = %.2f $\pm$ %.2f' %(np.rad2deg(mes_theta[0]),np.rad2deg(mes_theta[2])))
                          
            if fixed_b == None:
                axs[1][2].hist(positions[5],bins = 50,lw = 1,edgecolor = 'black')
                axs[1][2].axvspan(mes_b[0]-mes_b[2],mes_b[0]+mes_b[2], alpha=0.5, color='cyan',hatch = '/')
                axs[1][2].axvline(mes_b[0], alpha=1, color='cyan',lw = 2, ls = '-.')
                axs[1][2].set_title('b = %.2f $\pm$ %.2f' %(mes_b[0],mes_b[2]))
            else:
                axs[1][2].set_title('b = FIXED')   
                          
            plt.show()
        
        #################################################################
        
            print('Calculated Magnitude Zeropoint: %.4f' %(streak.totalmag+2.5*np.log10(mes_f[0])))
            print('Lower bound Zeropoint Error: %.4f' %(2.5/np.log(10)*mes_f[2]/mes_f[0]))
            print('True Zeropoint: %.4f' %(self.magzero))
        
        streak.emcee_flux = (mes_f[0],mes_f[2])
        streak.emcee_magzero = (streak.totalmag+2.5*np.log10(mes_f[0]),2.5/np.log(10)*mes_f[2]/mes_f[0])
        
        streak.emcee_x = (mes_x[0],mes_x[2])
        streak.emcee_y = (mes_y[0],mes_y[2])  ## <------------ SETTING THESE TO MEASURED VALUES AND 1SIG UNCERTAINTIES
        streak.emcee_L = (mes_L[0],mes_L[2])
        streak.emcee_theta = (mes_theta[0],mes_theta[2])
        streak.emcee_b = (mes_b[0],mes_b[2])

class sim_streak:  ## sim_streak is the class corresponding to a simulated streak in the image. Instances are intialzed in reference to a                                streak_interface
                    ## instance, which is the '.sim' parameter of the sim_streak instance
        
        def __init__(self, sim, x1 = 100, y1 = 100, L = 500, theta = np.pi/4,S = 0.92335, dist = 6e5, alpha = np.pi/4, IS_rad = 16.97,A_IS =                          1.27e-4, delta_t = 0.15,theta_override = None):
        
            self.in_image = False
            self.hasbeenadded = False ## this keep track of whether or not the streak has ever been added
            self.x1 = x1
            self.y1 = y1
            self.L = L
            self.theta = theta
            self.S_of_lambda = S
            self.dist = dist
            self.alpha = alpha
            self.IS_rad = IS_rad
            self.A_IS = A_IS
            self.delta_t = delta_t
            self.theta_override = theta_override
            
            self.sim = sim
            
            self.sim.streak_list.append(self)
            
        def pilot(self): ## The pilot method allows one to quickly see clearly where the streak should appear in the image, if .add() is run
            
            x = [self.x1,self.x1+self.L*np.sin(self.theta)]
            y = [self.y1,self.y1+self.L*np.cos(self.theta)]
            
            self.sim.display()
            plt.plot(y,x, c = 'r',linestyle = '-.',lw = 1.5)
            
        def add(self): ## adds the streak, with all of its properties, to the image associated with the streak_interface instance. Does this
                        ## by adding the precomputed map if no parameters have been changed, or recalculates the map if parameters have been
                        ## changed

            if self.in_image == True:
                
                print('Streak is currently in the image! Remove it to change its properties')
                
            else:
                
                if not self.hasbeenadded: ### If the streak has never been added before, set all of the 'last' values to the current parameters
                                            ## so these can be checked against for parameter changes later

                    self.hasbeenadded = True
                    self.last_x1 = self.x1
                    self.last_y1 = self.y1
                    self.last_L = self.L
                    self.last_theta = self.theta
                    self.last_S_of_lambda = self.S_of_lambda
                    self.last_dist = self.dist
                    self.last_alpha = self.alpha
                    self.last_IS_rad = self.IS_rad
                    self.last_A_IS = self.A_IS
                    self.last_delta_t = self.delta_t
                    self.last_theta_override = self.theta_override

                    self.sim.add_streak(self)
                    self.in_image = True                                   # (below) checks to see if all parameters are the same as when last added, because then
                                                                            # can just add the old map rather than recomputing

                elif (self.last_x1 == self.x1) and (self.last_y1 == self.y1) and (self.last_L == self.L) and (self.last_theta == self.theta) and (self.last_S_of_lambda == self.S_of_lambda) and (self.last_dist == self.dist) and (self.last_alpha == self.alpha) and (self.last_IS_rad == self.IS_rad) and (self.last_A_IS == self.A_IS) and (self.last_delta_t == self.delta_t) and (self.last_theta_override == self.theta_override): ## if no parameters have been changed:

                    self.sim.current_image += self.map
                    print('Streak total magnitude: %.3f' %self.totalmag)
                    print('CPUL: %.3f' %self.cpul)
                    print('Counts before noise: %d' %self.gen_counts)
                    print('Counts with noise: %d' %self.poiss_counts)
                    print('Streak added successfully')
                    self.in_image = True

                else: ## if any parameters have been changed, update all 'last' parameters and recompute the streak map. 
                    
                    self.last_x1 = self.x1
                    self.last_y1 = self.y1
                    self.last_L = self.L
                    self.last_theta = self.theta
                    self.last_S_of_lambda = self.S_of_lambda
                    self.last_dist = self.dist
                    self.last_alpha = self.alpha
                    self.last_IS_rad = self.IS_rad
                    self.last_A_IS = self.A_IS
                    self.last_delta_t = self.delta_t
                    self.last_theta_override = self.theta_override

                    self.sim.add_streak(self)
                    self.in_image = True
            
        def remove(self):
            
            if self.in_image == True:
            
                self.sim.current_image -= self.map
                self.in_image = False
                print('Streak removed successfully')
                
            else: 
                
                print('Streak is not in the image, and can therefore not be removed!')
                
        def print_properties(self):
            
            '''
            Prints out all of the parameters and properties of the simulation and of the streak. Shown in brackets next to the parameter is the 
            name of the property which needs to be referenced to change this parameter. For example, the flux zeropoint is read out
            as 'Flux zeropoint (Fzero)'. To change the flux zeropoint, I need to set (name of this interface instance).Fzero
            ''' 
            
            print('\nAssociated with interface: ', self.sim,'\n')
            
            print('### INTERFACE PROPERTIES ###\n')
            
            print('Earth radius (R): %.1fm' %self.sim.R)
            print('Pixel scale (pixscale): %.3fasec/pixel' %self.sim.pixscale)
            print('Flux zeropoint (Fzero): %.3fnW/m^2' %(self.sim.Fzero*1e9))
            print('Background 1sigma noise (im_noise): %.2f counts' %self.sim.im_noise)
            print('Image exposure time (exp_t): %.1fs' %self.sim.exp_t)
            try:
                print('PSF 1sigma width (psf): %.2f pixels' %self.sim.psf)
            except:
                pass
            print('PSF FWHM (fwhm): %.2f pixels' %self.sim.fwhm)
            print('Image magnitude zeropoint (magzero): %.5f' %self.sim.magzero)
            print('display vmax (VM): %.1f\n' %self.sim.VM)
            
            print('### STREAK SPECIFIC PROPERTIES ###\n')
            
            print('Is currently in interface image: ', self.in_image)
            print('x1: (from top left, down) %d' %int(self.x1))
            print('y1: (from left to right) %d' %int(self.y1))
            print('Length (L): %d pixels' %int(self.L))
            print('Theta (CW from rightwards direction) (theta): %.3f radians = %.1f degrees' %(self.theta,np.rad2deg(self.theta)))
            print('Bandpass transmission at source wavelength (S_of_lambda): %.5f' %self.S_of_lambda)
            print('Distance to satellite at time of observation (dist): %.1fkm' %(self.dist/1000))
            print('Satellite altitude angle in the sky (alpha): %.3f radians = %.1f degrees' %(self.alpha,np.rad2deg(self.alpha)))
            print('Satellite integrating sphere radiance (IS_rad): %.3f W/m^2/Sr' %self.IS_rad)
            print('Satellite integrating sphere aperture area (A_IS): %.3f cm^2' %(self.A_IS*1e4))
            print('Light streak time (delta_t): (elapsed time of pulse) %.3fs' %self.delta_t)
            try:
                print('Streak total magnitude: %.3f' %self.totalmag)
            except:
                pass
            try:
                print('Counts per unit length (per pixel) (CPUL): %.3f' %self.cpul)
            except:
                pass
            try:
                print('Counts before noise: %d' %self.rawcounts)
            except:
                pass
            try:
                print('Counts with noise: %d' %self.counts)
            except:
                pass
            print('\n')
            
class real_streak:  ## sim_streak is the class corresponding to a simulated streak in the image. Instances are intialzed in reference to a                              streak_interface
                    ## instance, which is the '.sim' parameter of the sim_streak instance
        
        def __init__(self, sim, x1 = 100, y1 = 100, L = 500, theta = np.pi/4,S = 0.92335, dist = 6e5, alpha = np.pi/4, IS_rad = 16.97,A_IS =                          1.27e-4, delta_t = 0.15,theta_override = None):
        
            self.in_image = True
            self.x1 = x1
            self.y1 = y1
            self.L = L
            self.theta = theta
            self.S_of_lambda = S
            self.dist = dist
            self.alpha = alpha
            self.IS_rad = IS_rad
            self.A_IS = A_IS
            self.delta_t = delta_t
            self.theta_override = theta_override
            
            self.sim = sim
            
        def compute_magnitude(self,visout = True):
            
            d = self.dist
            R = self.sim.R
            
            if self.theta_override == None:
                
                r_star = np.sqrt((R/d)**2+2*(R/d)*np.sin(self.alpha)+1)
                nadir = np.pi/2 - self.alpha - np.arcsin(np.cos(self.alpha)/r_star)
            
            else: 
                
                nadir = self.theta_override
            
            if visout == True:
            
                print('NADIR angle: %.2f degrees' %np.rad2deg(nadir))
        
            self.nadirangle = nadir
            I = self.IS_rad*self.A_IS*np.cos(nadir)/d**2
            m = -2.5*np.log10(I*self.delta_t*self.S_of_lambda/self.sim.exp_t/self.sim.Fzero)
            self.totalmag = m
            
            if visout == True:
                
                print('Computed magnitude: %.4f' %self.totalmag)
            
        def pilot(self,mode = 'smallbox',width = 10): ## This allows one to fit a good best guess for streak parameters, anticipating pill or                                                              MCMC fitting
            
            x = [self.x1,self.x1+self.L*np.sin(self.theta)]
            y = [self.y1,self.y1+self.L*np.cos(self.theta)]
            
            self.cheatx = np.mean(x)
            self.cheaty = np.mean(y)
            
            if mode == 'twomatch':
            
                L = 2

                while L < self.L:

                    L *= 2

                print('twomatch on - using symmetric box of edgelength %d' %L)
                
                testsect = self.sim.current_image[int(self.cheatx-L/2):int(self.cheatx+L/2),
                                               int(self.cheaty-L/2):int(self.cheaty+L/2)]
                
            elif mode == 'smallbox':
                
                Lx = int(self.L * np.abs(np.sin(self.theta))) + 0.1*self.L + 30
                Ly = int(self.L * np.cos(self.theta)) + 0.1*self.L + 30
                
                print('dimensions: %d x %d' %(Lx,Ly))
        
                testsect = self.sim.current_image[int(self.cheatx-Lx/2):int(self.cheatx+Lx/2),
                                               int(self.cheaty-Ly/2):int(self.cheaty+Ly/2)]
            try:
                
                x = np.array([Lx/2 - self.L/2*np.sin(self.theta),Lx/2 + self.L/2*np.sin(self.theta)])
                y = np.array([Ly/2 - self.L/2*np.cos(self.theta),Ly/2 + self.L/2*np.cos(self.theta)])
                
            except:
                
                x = np.array([L/2 - self.L/2*np.sin(self.theta),L/2 + self.L/2*np.sin(self.theta)])
                y = np.array([L/2 - self.L/2*np.cos(self.theta),L/2 + self.L/2*np.cos(self.theta)])
            
            plt.figure(figsize = (8,8))
            plt.imshow(testsect,vmax = self.sim.VM,vmin = -self.sim.VM,cmap = 'Greys')
            
            plt.plot(y-width*np.sin(self.theta),x+width*np.cos(self.theta), c = 'r',linestyle = '-.',lw = 2,alpha = 0.35)
            plt.plot(y+width*np.sin(self.theta),x-width*np.cos(self.theta), c = 'r',linestyle = '-.',lw = 2,alpha = 0.35)
            
            try:
                plt.scatter(int(Ly/2),int(Lx/2),c = 'r',s = 50,marker = 'x',alpha = 0.5)
            except:
                plt.scatter(int(L/2),int(L/2),c = 'r',s = 50,marker = 'x',alpha = 0.5)
                
        def print_properties(self):
            
            '''
            Prints out all of the parameters and properties of the simulation and streak. Shown in brackets next to the parameter is the 
            name of the property which needs to be referenced to change this parameter. For example, the flux zeropoint is read out
            as 'Flux zeropoint (Fzero)'. To change the flux zeropoint, I need to set (name of this interface instance).Fzero
            ''' 
            
            print('\nAssociated with interface: ', self.sim,'\n')
            
            print('### INTERFACE PROPERTIES ###\n')
            
            print('Earth radius (R): %.1fm' %self.sim.R)
            print('Pixel scale (pixscale): %.3fasec/pixel' %self.sim.pixscale)
            print('Flux zeropoint (Fzero): %.3fnW/m^2' %(self.sim.Fzero*1e9))
            print('Background 1sigma noise (im_noise): %.2f counts' %self.sim.im_noise)
            print('Image exposure time (exp_t): %.1fs' %self.sim.exp_t)
            try:
                print('PSF 1sigma width (psf): %.2f pixels' %self.sim.psf)
            except:
                pass
            print('PSF FWHM (fwhm): %.2f pixels' %self.sim.fwhm)
            print('Image magnitude zeropoint (magzero): %.5f\n' %self.sim.magzero)
            
            print('### STREAK SPECIFIC PROPERTIES ###\n')
    
            print('x1: (from top left, down) %d' %int(self.x1))
            print('y1: (from left to right) %d' %int(self.y1))
            print('Length (L): %d pixels' %int(self.L))
            print('Theta (CW from rightwards direction) (theta): %.3f radians = %.1f degrees' %(self.theta,np.rad2deg(self.theta)))
            print('Bandpass transmission at source wavelength (S_of_lambda): %.5f' %self.S_of_lambda)
            print('Distance to satellite at time of observation (dist): %.1fkm' %(self.dist/1000))
            print('Satellite altitude angle in the sky (alpha): %.3f radians = %.1f degrees' %(self.alpha,np.rad2deg(self.alpha)))
            print('Satellite integrating sphere radiance (IS_rad): %.3f W/m^2/Sr' %self.IS_rad)
            print('Satellite integrating sphere aperture area (A_IS): %.3f cm^2' %(self.A_IS*1e4))
            print('Light streak time (delta_t): (elapsed time of pulse) %.3fs' %self.delta_t)
            try:
                print('Streak total magnitude: %.3f' %self.totalmag)
            except:
                pass
            try:
                print('Counts per unit length (per pixel) (CPUL): %.3f' %self.cpul)
            except:
                pass
            print('\n')
            
def padmatch3(im1, im2, x0 = 0, y0 = 0):   ### PADMATCH 3 IS AN OLD AND LESS EFFICIENT VERSION OF PADMATCH2. KEPT HERE IN CASE PADMATCH2 SOMETIMES FAILS
    
    ori_shape = im1.shape
    
    ## start with x:
    
    im2 = list(im2)
    
    horiz_len = len(im2[0])
    topadd = list(np.zeros(horiz_len))
    
    for i in range(x0):
        
        im2 = [topadd] + im2
    
    if (len(im1) > len(im2)):
        
        diffx = len(im1) - len(im2)
        
        im2 = list(im2)
        
        for i in range(diffx):
            
            im2.append(list(np.zeros(len(im2[0]))))
                       
    elif len(im2) > len(im1):
                       
        diffx = len(im2) - len(im1)
                       
        im1 = list(im1)
                       
        for i in range(diffx):
                       
            im1.append(list(np.zeros(len(im1[0]))))
                       
    elif len(im2) == len(im1):
                       
        pass
                       
    # now y:
    
    for i in range(len(im2)):
        
        im2[i] = list(im2[i])
        
        im2[i] = list(np.zeros(y0))+im2[i]
                       
    if len(im1[0]) > len(im2[0]):
                       
        im2 = list(im2)
                       
        while len(im1[0]) > len(im2[0]):
                       
            for i in range(len(im2)):
                    
                im2[i] = list(im2[i])
                       
                im2[i].append(0)
                       
    elif len(im2[0]) > len(im1[0]):
                       
        im1 = list(im1)
                       
        while len(im2[0]) > len(im1[0]):
                       
            for i in range(len(im1)):
                    
                im1[i] = list(im1[i])
                       
                im1[i].append(0)
                       
    elif len(im1[0]) == len(im2[0]):
                       
        pass
                       
    im1 = np.array(im1)
                       
    im2 = np.array(im2)
    
    #print(im1+im2)
    
    add = (im1+im2)[:ori_shape[0],:ori_shape[1]]
    
    removeable = im2[:ori_shape[0],:ori_shape[1]]
    
    return add,removeable
                
def padmatch2(im1, im2, x0 = 0, y0 = 0): ## Padmatch2 takes the model generated by adapted model (im2) and places it in the correct 
                                           # place in im1 by zero-padding the generated model and original image until their sizes match and the                                              # streak is in the specified position
    
    ori_shape = im1.shape
    
    xlim = int(ori_shape[0])
    ylim = int(ori_shape[1])
    
    ## start with x: First adds rows of zeros until the streak x starts in the right position
    
    im1 = [list(i) for i in im1]
    im2 = [list(i) for i in im2]
    
    horiz_len = len(im2[0])
    topadd = np.zeros((x0,horiz_len))
    topadd = [list(i) for i in topadd]
        
    im2 = topadd + im2
    
    if (len(im1) > len(im2)):
        
        diffx = len(im1) - len(im2)
            
        im2 = im2 + [list(i) for i in np.zeros((diffx,len(im2[0])))]
                       
    elif len(im2) > len(im1):
                       
        diffx = len(im2) - len(im1)
                       
        im1 = im1 + [list(i) for i in np.zeros((diffx,len(im1[0])))]
                       
    elif len(im2) == len(im1):
                       
        pass
                       
    # now y:
    
    im1 = np.array(im1).T ### TRANSPOSE THE IMAGE AND REPEAT THE X PROCEDURE TO DO Y
    im2 = np.array(im2).T
    
    im1 = [list(i) for i in im1]
    im2 = [list(i) for i in im2]
    
    horiz_len = len(im2[0])
    topadd = np.zeros((y0,horiz_len))
    topadd = [list(i) for i in topadd]
    
    #print('topadd shape:',topadd.shape)
        
    im2 = topadd + im2
    
    if (len(im1) > len(im2)):
        
        diffx = len(im1) - len(im2)
            
        im2 = im2 + [list(i) for i in np.zeros((diffx,len(im2[0])))]
                       
    elif len(im2) > len(im1):
                       
        diffx = len(im2) - len(im1)
                       
        im1 = im1 + [list(i) for i in np.zeros((diffx,len(im1[0])))]
                       
    elif len(im2) == len(im1):
                       
        pass
                       
    im1 = np.array(im1).T
                       
    im2 = np.array(im2).T
    
    #print(im1+im2)
    
    plus = im1 + im2
    
    add = plus[:xlim,:ylim]
    
    removeable = im2[:xlim,:ylim]
    
    return add,removeable

def adapted_model(im_size, x1, x2, y1, y2, psf_sigma=2, replace_value=0, threshold=1e-10, oversample=1, JUSTLINE = False, dm = 0.7):
    
    #This is an adaptation of the 'model' function from pyradon. 
    
    '''
    
    Produces a streak with a certain PSF, predefined endpoints, etc.
    
    PARAMETERS:
    
    im_size: The power of two sidelength of the generated array which will contain the streak
    
    x1,x2,y1,y2: The endpoints of the streak
    
    psf_sigma: Can be either the sigma parameter for a Gaussian psf, or a 2D array (odd sidelength!!) containing the desired PSF
    
    oversample: the oversampling factor for streak generation. Higher values gives more accurate pixel values for a simulated streak
    
    JUSTLINE: If true, does not convolve with the PSF, allows line profile to be seen
    
    dm: 'distance max'. Pixels within this distance from the mathematical line are turned 'on'. Values that are too small lead to holes
        and artifacts in the line, values that are too large lead to a distorted streak where the psf isn't entirely correct. Suggest leaving at          0.7
        
    '''
    
    dmax = dm
    
    if np.isscalar(im_size):
        im_size = (im_size, im_size)
    elif not isinstance(im_size, tuple) or len(im_size) != 2:
        raise TypeError('Input "im_size" must be a scalar or 2-tuple')

    if not all(isinstance(s, int) for s in im_size):
        raise TypeError('Input "im_size" must have only int type values. ')

    if oversample:
        im_size = tuple(s * oversample for s in im_size)
        x1 = (x1 - 0.5) * oversample + 0.5
        x2 = (x2 - 0.5) * oversample + 0.5
        y1 = (y1 - 0.5) * oversample + 0.5
        y2 = (y2 - 0.5) * oversample + 0.5
        psf_sigma = psf_sigma * oversample

    #    (x,y) = np.meshgrid(range(im_size[1]), range(im_size[0]), indexing='xy')
    (y, x) = np.indices(im_size, dtype="float32")

    if x1 == x2:
        a = float("Inf")  # do we need this?
        b = float("NaN")  # do we need this?
        d = np.abs(x - x1)  # distance from vertical line
    else:
        a = (y2 - y1) / (x2 - x1)  # slope parameter
        b = (y1 * x2 - y2 * x1) / (x2 - x1)  # impact parameter
        d = np.abs(a * x - y + b) / np.sqrt(1 + a**2)  # distance from line
        
    if JUSTLINE == False:
    
        if np.isscalar(psf_sigma):

            # an image of an infinite streak with gaussian width psf_sigma
            im0 = (1 / np.sqrt(2.0 * np.pi) / psf_sigma) * np.exp(
                -0.5 * d**2 / psf_sigma**2)

        else:
        #print('nonscalar PSF - using given kernel')
        
            im0 = np.zeros(d.shape)
            
            im0[d <= dmax] = 1
        
    else:
        
        im0 = np.zeros(d.shape)
        im0[d <= dmax] = 1
        
    # must clip this streak:
    if x1 == x2 and y1 == y2:
        im0 = np.zeros(im0.shape)
    elif x1 == x2:
        if y1 > y2:
            im0[y > y1] = 0
            im0[y < y2] = 0
        else:
            im0[y < y1] = 0
            im0[y > y2] = 0

    elif y1 == y2: 
        if x1 > x2:
            im0[x > x1] = 0
            im0[x < x2] = 0
        else:
            im0[x < x1] = 0
            im0[x > x2] = 0

    elif y1 < y2:
        im0[y < (-1 / a * x + y1 + 1 / a * x1)] = 0
        im0[y > (-1 / a * x + y2 + 1 / a * x2)] = 0
    else:
        im0[y > (-1 / a * x + y1 + 1 / a * x1)] = 0
        im0[y < (-1 / a * x + y2 + 1 / a * x2)] = 0
        
    if JUSTLINE == False:
        
        if np.isscalar(psf_sigma):

            # make point-source gaussians at either end of the streak
            im1 = (1 / np.sqrt(2 * np.pi) / psf_sigma) * np.exp(
                -0.5 * ((x - x1) ** 2 + (y - y1) ** 2) / psf_sigma**2)
            im2 = (1 / np.sqrt(2 * np.pi) / psf_sigma) * np.exp(
                -0.5 * ((x - x2) ** 2 + (y - y2) ** 2) / psf_sigma**2)

            # "attach" the point sources by finding the maximum pixel value
            out_im = np.fmax(im0, np.fmax(im1, im2))
            
            out_im = downsample(out_im,oversample)

        else:

            if oversample > 1:

                im0 = downsample(im0,oversample)

            out_im = fftconvolve(im0,psf_sigma,mode = 'same')
            
    else:
        
        if oversample > 1:
            
            im0 = downsample(im0,oversample)
        
        out_im = im0

    #if oversample > 1:
        
        #out_im = downsample(out_im, oversample) / oversample

    # apply threshold and replace value
    if threshold is not None and replace_value is not None:
        out_im[out_im < threshold] = replace_value

    return out_im

def downsample(im, factor=2, normalization="sum"):
    """
    Scale down an image by first smoothing
    it with a square kernel, then subsampling
    the pixels in constant steps.
    Both the kernel size and steps are
    equal to the downsample factor.

    Parameters
    ----------
    im: np.array
        The image that should be downsampled.
        The original image is not altered.

    factor: scalar int
        How many pixels need to be combined
        to produce each of the output pixels.
        The input image is effectively scaled
        down by this number.

    normalization: str
        Choose "sum" or "mean" to determine
        if pixels that are combined should
        be simply summed (default) or averaged.

    Returns
    -------
        A smaller image array, where the scale
        of the image is smaller by "factor".

    """

    if factor is None or factor < 1:
        return im

    if not isinstance(factor, int):
        raise TypeError('Input "factor" must be a scalar integer. ')

    k = np.ones((factor, factor), dtype=im.dtype)
    if normalization == "mean":
        k = k / np.sum(k)
    elif normalization != "sum":
        raise KeyError('Input "normalization" must be "mean" or "sum". 'f'Got "{normalization}" instead. ')

    im_conv = convolve2d(im, k, mode="same")

    return im_conv[factor - 1 :: factor, factor - 1 :: factor]


def upsample(im, factor=2):
    """
    Use FFT interpolation (sinc interp) to up-sample
    the given image by a factor (default 2).
    """
    before = [int(np.floor(s * (factor - 1) / 2)) for s in im.shape]
    after = [int(np.ceil(s * (factor - 1) / 2)) for s in im.shape]

    im_f = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(im)))
    im_pad_f = np.pad(im_f, [(before[0], after[0]), (before[1], after[1])])
    im_new = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(im_pad_f)))

    return im_new
