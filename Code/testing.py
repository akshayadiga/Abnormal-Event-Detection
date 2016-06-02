import cv2
import numpy as np

def testing_algorithm(x,S,T):
	#print S
	R=getR(S);
	#print R	
	return_list=[]
	file_list=[]
	#print xs
	i=0
	time=0
	flag=0
	for xi in x:
		i+=1
		flag=0
		mean=[]
		for Ri in R:
			val=np.linalg.norm(np.dot(np.array(Ri),np.array([xi])))**2
			mean.append(val)
			
			if(val<T):
				flag=1
				break		
		if(i==208):
				i=0
				min_mean=min(mean)
				if((str("Abnormal at time"+str(time)+" seconds.") not in return_list) and min_mean>0.00000045): 	
					return_list.append(str("Abnormal at time"+str(time)+" seconds."))
					file_list.append(time)
				#print "time:small : ",time,min_mean
				time+=5
				mean=[]
			
	return file_list,return_list
	
def getR(S):
	R=[];
	m=0.00000003
	
	for Si in S:
		Si = np.array(Si);
		Si_transpose = np.transpose(Si);
		Si_T_Si=np.dot(Si_transpose,Si)
		if(np.linalg.det(Si_T_Si)==0):
			Si_T_Si=np.add(Si_T_Si,m*np.eye(10,10))
		
		Ri=np.subtract(np.dot(Si,np.dot(np.linalg.inv(Si_T_Si),Si_transpose)),np.identity(len(Si)));
		R.append(Ri);

	return R;