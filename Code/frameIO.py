import os
from fnmatch import fnmatch
import cv2
import numpy as np 


def framesToFeatures(root='C:/Users/akshaya/Desktop/Final Year Project/Code/frames/',pattern = "*.tif"):
	#folder path --> directory
	
	#features vector
	features=[]
	#features=np.array(features)
	features1=[]
	#take all .tif files from train folder
	
	file_list=[]
	
	for path, subdirs, files in os.walk(root):
		for name in files:
			if fnmatch(name, pattern):
				#print os.path.join(path, name)
				file_list.append(str(os.path.join(path, name)))
	
	numOfFiles=len(file_list)
	print "Number of Frames imorted : ",numOfFiles
	
	#file counter
	
	i=0
	
	time=0
	number_cubes=0
	#reading 5 images at a time
	while (numOfFiles-i>=5):
		time+=1
		
		img1=cv2.cvtColor(cv2.imread(file_list[i]),cv2.COLOR_BGR2GRAY);i+=1;
		img2=cv2.cvtColor(cv2.imread(file_list[i]),cv2.COLOR_BGR2GRAY);i+=1;
		img3=cv2.cvtColor(cv2.imread(file_list[i]),cv2.COLOR_BGR2GRAY);i+=1;
		img4=cv2.cvtColor(cv2.imread(file_list[i]),cv2.COLOR_BGR2GRAY);i+=1;
		img5=cv2.cvtColor(cv2.imread(file_list[i]),cv2.COLOR_BGR2GRAY);i+=1;
		
		image_set=[img1,img2,img3,img4,img5]
		
		#Create 3 different scale for each image
		
		re_img_2020_set=[]
		re_img_4030_set=[]
		re_img_160120_set=[]
		for image in image_set:
			img_2020=cv2.resize(image,(20,20))
			img_4030=cv2.resize(image,(40,30))
			img_160120=cv2.resize(image,(160,120))
			
			re_img_2020_set.append(img_2020)
			re_img_4030_set.append(img_4030)
			re_img_160120_set.append(img_160120)
		
		resize_image_set=[re_img_2020_set,re_img_4030_set,re_img_160120_set]
		
		#Collect non-overlaping patches form all the scale 
		
		patches_all=[[],[],[]]
		i1=0
		
		for images_set in resize_image_set:
			for resize_img in images_set:
				patch_list=[]
				patch=[]
				for start in range(0,len(resize_img[0]),10):
					count=1
					for row in resize_img:
						patch.append(row[start:start+10])
					
						if(count==10):
							count=0
							patch_list.append(patch)
							patch=[]
						count+=1
				patches_all[i1].append(patch_list)
			i1+=1
		
		
		
		#Generate cubes and list of all cubes
		
		cubes=[]
		
		for resolution_patch_set in patches_all:
			for i1 in range(len(resolution_patch_set[0])):
				p_one = resolution_patch_set[0][i1];
				p_two = resolution_patch_set[1][i1];
				p_three = resolution_patch_set[2][i1];
				p_four = resolution_patch_set[3][i1];
				p_five = resolution_patch_set[4][i1];
				cubes.append([p_one,p_two,p_three,p_four,p_five])
		
		
		#Cubes are generated and in variable cubes
		#print "Number of cubes : ",len(cubes)
		number_cubes+=len(cubes)
		
		#features=[]
		for cub in cubes:
			#Calculate the x, y and t derivative for each cubes
			
			sobelx = cv2.Sobel(np.array(cub),cv2.CV_64F,1,0,ksize=-1)
			sobely = cv2.Sobel(np.array(cub),cv2.CV_64F,0,1,ksize=-1)
			sobelt = cv2.Sobel(np.array(zip(*cub)),cv2.CV_64F,0,1,ksize=-1)
			sobelt = zip(*sobelt)
			
			feature = []
			#feature=np.array(feature)
			
			#Concatinate all the x,y,t values at each pixel to generate 1500 dimension feature
			
			for time_value in range(5):
				for y_value in range(10):
					for x_value in range(10):
						feature.append(sobelx[time_value][y_value][x_value])
						feature.append(sobely[time_value][y_value][x_value])
						feature.append(sobelt[time_value][y_value][x_value])
						#np.append(feature,sobelx[time_value][y_value][x_value])
						#np.append(feature,sobely[time_value][y_value][x_value])
						#np.append(feature,sobelt[time_value][y_value][x_value])
						
			#Append the generated feature to the already existing feature list
			
			features.append(feature)
			#np.append(features,feature)
		#features1.append(features)
		
	#features=features.tolist()	
		
	# 'features' holds the list of all features generated
	print ""
	print "--------------Done Feature Extraction------------"	
	print "Number of cubes generated : ",number_cubes
	print "Number of feature generated : ",len(features)
	print "Length of each feature : ",len(features[0])
	print "-------------------------------------------------"
	print ""
	
	#Return the list of all feature generated
	
	return features
		
		
def show_image(folder,file_list_no,pattern="*.tif"):
	
	file_list=[]
	
	for path, subdirs, files in os.walk(folder):
		for name in files:
			if fnmatch(name, pattern):
				#print os.path.join(path, name)
				file_list.append(str(os.path.join(path, name)))
	
	numOfFiles=len(file_list)
	#print file_list
	file_to_print=[]
	
	for f_no in file_list_no:
		if(f_no==0):
			file_to_print.append(file_list[0])
			file_to_print.append(file_list[1])
			file_to_print.append(file_list[2])
			file_to_print.append(file_list[3])
			file_to_print.append(file_list[4])
			file_to_print.append(file_list[5])
		elif(f_no<=numOfFiles):
			file_to_print.append(file_list[f_no-1])
			if(f_no-1+1<numOfFiles):
				file_to_print.append(file_list[f_no-1+1])
			if(f_no-1+2<numOfFiles):
				file_to_print.append(file_list[f_no-1+2])
			if(f_no-1+3<numOfFiles):
				file_to_print.append(file_list[f_no-1+3])
			if(f_no-1+4<numOfFiles):
				file_to_print.append(file_list[f_no-1+4])
			if(f_no-1+5<numOfFiles):
				file_to_print.append(file_list[f_no-1+5])
		else:
			break
	
	for file in file_to_print:
		img = cv2.imread(file) #read a picture using OpenCV
		cv2.imshow('image',img) # Display the picture
		cv2.waitKey(150) # wait for closing
		#cv2.destroyAllWindows() # Ok, destroy the window
	cv2.destroyAllWindows()
#framesToFeatures()