import testing
import videoIO
import frameIO
import cv2
import numpy as np
import os
import pickle

def main():
	user_request=str(raw_input("Enter training method : "))
	
	S=[]
	
	if(user_request!="reload"):
		print "Importing training data from cached copy..."
		for file_no in range(1,14):
			if(os.path.isfile('training_cache/save'+str(file_no)+'.p')): 
				S += pickle.load(open('training_cache/save'+str(file_no)+'.p', 'rb'))
	else:
		S=preprocess_and_training()
		pickle.dump(S, open('training_cache/save13.p', 'wb')) 
	
	#print S
	#print len(S)
	
	print "\n\n"
	print "-------------------------------"
	print "TESTING PHASE"
	print "-------------------------------"
	print "\n\n"
	
	
	import_and_test_abnormal(S)
	
	
	#print "Next is testing"

def import_and_test_abnormal(S):
	
	#Input the frame of the video to be tested
	
	test_folder_list=[]
	
	#for path, subdirs, files in os.walk('C:/Users/akshaya/Desktop/Final Year Project/Code/frames/Test/Test1/'):
	for path, subdirs, files in os.walk('frames/Test/Test1/'):
		for name in subdirs:
			#print os.path.join(path, name)
			test_folder_list.append(str(os.path.join(path, name)))	
	
	feature_list=[]
	
	for folder in test_folder_list:
		if("_gt" in folder):
			feature_list=frameIO.framesToFeatures(root=folder,pattern="*.bmp")
		else:
			feature_list=frameIO.framesToFeatures(root=folder,pattern="*.tif")
		
		file_list,result=testing.testing_algorithm(feature_list,S,0.00001915625)
		
		
		print "#####################################"
		print folder," video is :: "
		
		if(len(result)==0):
			print "Normal"
		else:
			for res in result:
				print res
			continue_key=raw_input("Press enter to show the abnormal frames : ")
			key_str="1"
			while(key_str=="1"):
				if(continue_key==""):
					frameIO.show_image(folder,file_list)
					key_str=raw_input("Press Enter to continue or 1 to replay : ")
					
		print "#####################################"
			
def preprocess_and_training():
	#Generate features from it
	
	
	i=0
	file_list1=[]	
	for path, subdirs, files in os.walk('frames/Train/Train1/'):
		for name in subdirs:
		
			#print os.path.join(path, name)
			file_list1.append(str(os.path.join(path, name)))
	

	file_list2=[]	
	for path, subdirs, files in os.walk('frames/Train/Train2/'):
		for name in subdirs:
		
			#print os.path.join(path, name)
			file_list2.append(str(os.path.join(path, name)))
	
	feature_list=[]
	
	for folder in file_list1:
		feature_list.append(frameIO.framesToFeatures(root=folder,pattern = "*.tif"))
		i+=1
		print "files done:",i
		
	for folder in file_list2:
		feature_list.append(frameIO.framesToFeatures(root=folder,pattern = "*.tif"))
		i+=1
		print "files done:",i
		
	#feature_list=frameIO.framesToFeatures(root='C:/Users/akshaya/Desktop/Final Year Project/Code/frames/',pattern="*.tif")
	print "Total Feature generated : ",len(feature_list)
	
	#return
	#Training for Sparse Combination Learning
	
	S=[]
	B=[]
	
	feature_list=[i[:100] for i in feature_list]
	
	for set in feature_list:	
		S_temp,B_temp=training_algorithm(set)
		S+=S_temp
		B+=B_temp
		#S.append(S_temp)
		#B.append(S_temp)
	
	
	print "################Final S Vector###################"
	print "\n"
	print S
	print "\n"
	print "Length of S is : ",len(S)
	print "##################################################"
	
	return S
	
def training_algorithm(X):
	#Training Section using Sparse Combination Learning
	
	#######
	#Inputs
	#######
	#X=feature_list
	#######
	
	Xc=X
	S=[]
	B=[]
	gamma=[]
	i=1
	
	#Algorithm
	
	while(len(Xc)>10):
		#Create the initial dictionary Si using kmeans
		
		criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		flags = cv2.KMEANS_RANDOM_CENTERS
		print "Length of Xc is : ",len(Xc)
		compactness,labels,centers = cv2.kmeans(np.array(Xc,dtype="float32"),10,None,criteria,10,flags)
		centers=[(sum(val)/len(val)) for val in centers]
		Si=[centers]
		
		#Reset Gamma and Beta i for next Vector generation
		
		gamma=[]
		Bi=[]
		history_2=0
		history_1=0
		epoch=0
		max_epoch=10
		start=1
		
		while(start==1 or start==2 or deltaL<0):
			
			if(start==2):
				start=0
			if(start==1):
				deltaL=0
				start=2
				L2=0
				L1=0
			
			Bi=optimise_beta(Si,Xc)
			Si=np.subtract(np.array(Si),(0.0001*deltaL))
			gamma=optimise_gamma(Si,Xc,Bi,0.04)
			L1=L2
			L2=evaluate_L(Si,Xc,Bi,gamma)
			
			deltaL=L2-L1
			#Print new Values
			
			print "\n\n"
			print "*************Value in Iteration******************"
			print "L = ",L2
			epoch+=1
			print "Epoch : ",epoch
			print "DeltaL : ",deltaL
			print "*************************************************"
			print "\n\n"
			
		S.append(Si)
		B.append(Bi)
		
		print "\n\n"
		print "------New vector generated--------"
		print "Si = ",S
		print "S vector = ",len(S)
		print "----------------------------------"
		print "\n\n"
		
		#Removing computed features
		
		change_index=0
		#print gamma
		for val in range(len(gamma)):
			if(gamma[val]==0):
				del Xc[val-change_index]
				change_index+=1
		print change_index		
		#Increment counter
		i+=1
		
	#Return Generated Vector set and Beta
	
	return S,B

def optimise_beta(Si,Xc):
	#Using equation 6 optimise beta value
	
	beta=[]
	Si = np.array(Si)
	Si_transpose = np.transpose(Si)
	m=0.00000003
	
	#print Si.shape
	#print Si_transpose.shape
	
	for xj in Xc:
		numpy_xj=np.array([xj])
		Si_T_Si=np.dot(Si_transpose,Si)
		
		if(np.linalg.det(Si_T_Si)==0):
			Si_T_Si=np.add(Si_T_Si,m*np.eye(10,10))
			
		inverse_sit=np.linalg.inv(Si_T_Si)
		dot_in_si=np.dot(inverse_sit,Si_transpose)
		itr_beta=np.dot(dot_in_si,numpy_xj)
		beta.append(itr_beta)
	
	return beta

	
def optimise_gamma(Si,Xc,Bi,lamda):
	#Using equation 9 optimise gamma value
	
	gamma=[]
	Si = np.array(Si)
	
	for xj in range(len(Xc)):
		#print ((np.linalg.norm(np.subtract(np.array([Xc[xj]]),np.dot(Si,np.array(Bi[xj])))))**2)
		if((((np.linalg.norm(np.subtract(np.array([Xc[xj]]),np.dot(Si,np.array(Bi[xj])))))**2)**2)<lamda):
			gamma.append(1)
		else:
			gamma.append(0)
			
	return gamma

def evaluate_L(Si,Xc,Bi,gamma):
	#Using equation 9 optimise gamma value
	
	L=0
	Si = np.array(Si)
	temp_l=[]
	
	for xj in range(len(Xc)):
		l_iter_val=gamma[xj]*(((np.linalg.norm(np.subtract(np.array([Xc[xj]]),np.dot(Si,np.array(Bi[xj])))))**2)**2)
		temp_l.append(l_iter_val)
	
	return sum(temp_l)
	
main()