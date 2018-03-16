import numpy as np
from PIL import Image
import sys

class ColourBlindness(object):

    def __init__(self,
        image,
        phosphors = './phosphors.dat',
        fundamentals = './2deg_StockmanSharpe.csv'):

        self.image = Image.open(image)

        wv1, self.phosphors = self.get_phosphors(phosphors)
        wv2, self.fundamentals = self.get_fundamentals(fundamentals)
        self.gamma = np.array([2.38, 2.30, 2.17])

        self.RGB_LMS = np.dot(self.fundamentals.T, self.phosphors)
        self.LMS_RGB = np.linalg.inv(self.RGB_LMS)

        # blue = self.rgb2lms([0,0,1])
        blue = np.array([0.54, 0.21, 0.79])
        white = self.rgb2lms([1,1,1])

        self.a = white[1] * blue[2] - blue[1] * white[2]
        self.b = white[2] * blue[0] - blue[2] * white[0]
        self.c = white[0] * blue[1] - blue[0] * white[1]



    def get_phosphors(self, filename):

        with open(filename, 'rb') as f:
            lines = [l.rstrip().split('\t') for l in f][33:]
            lines = np.array( [map(float, l) for l in lines] )
        wavelengths = lines[:,0]
        power = lines[:,1:]
        mask = np.logical_and(390 <= wavelengths, wavelengths <= 780)

        return wavelengths, power[mask, :]


    def get_fundamentals(self, filename):

        with open(filename, 'rb') as f:
            lines = [l.rstrip().split(',') for l in f]
            lines = np.array( [map(float, l) for l in lines] )
        wavelengths = lines[:,0]
        power = lines[:,1:]
        mask = np.logical_and(390 <= wavelengths, wavelengths <= 780)

        return wavelengths, power[mask, :]


    def rgb2lms(self, rgb):

        lms = np.dot(self.RGB_LMS, rgb)
        return lms


    def lms2rgb(self, lms):

        rgb = np.dot(self.LMS_RGB, lms)
        return rgb


    def protanopia(self):

        rgb_data = (np.array(self.image) / 255.0) ** self.gamma
        lms_data = np.apply_along_axis(self.rgb2lms, 2, rgb_data)

        # Apply Eqs.(9)
        lms_data[:,:,0] = -(self.b * lms_data[:,:,1] + self.c * lms_data[:,:,2]) / self.a

        rgb_protan = np.apply_along_axis(self.lms2rgb, 2, lms_data)
        rgb_final = 255 * rgb_protan**(1.0/self.gamma)
        rgb_final = np.around(rgb_final, 0)

        return Image.fromarray(rgb_final.astype(np.uint8))


    def deuteranopia(self):

        rgb_data = (np.array(self.image) / 255.0) ** self.gamma
        lms_data = np.apply_along_axis(self.rgb2lms, 2, rgb_data)

        # Apply Eqs.(10)
        lms_data[:,:,1] = -(self.a * lms_data[:,:,0] + self.c * lms_data[:,:,2]) / self.b

        rgb_protan = np.apply_along_axis(self.lms2rgb, 2, lms_data)
        rgb_final = 255 * rgb_protan**(1.0/self.gamma)
        rgb_final = np.around(rgb_final, 0)

        return Image.fromarray(rgb_final.astype(np.uint8))


    def tritanopia(self):

        rgb_data = (np.array(self.image) / 255.0) ** self.gamma
        lms_data = np.apply_along_axis(self.rgb2lms, 2, rgb_data)

        # Apply Eqs.(11)
        lms_data[:,:,2] = -(self.a * lms_data[:,:,0] + self.b * lms_data[:,:,1]) / self.c

        rgb_protan = np.apply_along_axis(self.lms2rgb, 2, lms_data)
        rgb_final = 255 * rgb_protan**(1.0/self.gamma)
        rgb_final = np.around(rgb_final, 0)

        return Image.fromarray(rgb_final.astype(np.uint8))
