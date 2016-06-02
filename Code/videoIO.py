import numpy as np
import cv2
import os
import time

class videoIO:
	def __init__(self):
		print "Opening Video IO"

	def convertToFrames(self,time_to_save,save_path):
		print "Converting"
		
		#Capture arguments
		directory=save_path;
		
		cap = cv2.VideoCapture(0)

		# File index
		i=0

		# If previous frames exist clear them first
		if os.path.exists(directory):
			filelist = [ f for f in os.listdir(directory)]
			for f in filelist:
				os.remove(str(directory+"\\"+f))

		#Timer start
		s_time=time.time();
				
		# Collecting and storing frames		
		while(True):
			# Capture frame-by-frame
			ret, frame = cap.read()
			
			# Create directory to store frame
			if not os.path.exists(directory):
				os.makedirs(directory)
					
			# Store the resulting frame
			os.chdir(directory);	
			cv2.imwrite('frame'+str(i)+'.png',frame)
			os.chdir("..");
			i+=1
			
			# End after user specified seconds
			if(time.time()-s_time>=time_to_save):
				break


		# When everything done, release the capture
		cap.release()
		cv2.destroyAllWindows()