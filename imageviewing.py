#!/usr/bin/python
#Filename: imageviewing.py

"""This provides some functions for converting an image to an array and
viewing an image"""

import pandas as pd
import numpy as np
from skimage.viewer import ImageViewer



def series_to_array(series,dimensions=(28,28),pixel_columns=None):
    """series is a single image + label , as imported from the digit data,
    to a 2D array of pixel values of the proper dimensions"""
    if type(pixel_columns)==type(None):
		pixel_columns=['pixel%i' % i for i in xrange(dimensions[0]*dimensions[1])]
    return np.reshape(series[pixel_columns].values, dimensions)
    
def view_series(series):
    return(ImageViewer(series_to_array(series)))




