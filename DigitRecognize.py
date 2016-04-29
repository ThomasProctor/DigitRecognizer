
#import numpy as np
#####
#####
#####	Currently, this class is called with the filename, and takes advantage
#####		of the structure of the training/test datasets. 
#####		The "readImage" method makes the "index"th image, and transforms
#####		it into the 28x28 pixel image that it should be.
#####		The "makeBinary" method turns the image into a binary image
    
    
    

class Image:
	""" Contains the image data: will have pixel number array,
		can call image number #, and read in the file ...
	"""
	
	def __init__(self,filename):
		f = open(filename,'r')
		self.data = (f.readlines())
		if ( 'train' in filename ):
			self.train = True
		else:
			self.train = False
		
	
	def readImage(self,index):
		""" Read entry number (index) from the data from file (dataset).csv,
			and convert it into a floating point image, 28x28 pixels
		"""
		import numpy as np
		if (self.train == True):
			self.image = np.array( self.data[index].split(',')[1:] )
			self.imageLabel = int( self.data[index].split(',')[0] )
		else:
			self.image = np.array( self.data[index].split(',') )
			
		self.image =  self.image.astype(int)
		
		self.image = self.image.reshape( (28,28) )
#		self.maxPixel

	def makeBinary(self):
		""" Copy image and transform image into binary """
		self.binaryImage = self.image
		self.binaryImage[ self.image > 10 ] = 1
		self.binaryImage[ self.image <=10 ] = 0

