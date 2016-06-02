Pre-Required software

1. Python 2.7x
2. Modules
	- opencv
	- numpy

Running steps:

:::::::::Training:::::::::

1 -> Copy the folder containing the frames of the video to be tested to location "/frames/Train/Train1" or "/frames/Train/Train2" 
2 -> Go to the folder which has "project.py"
3 -> Run it using the command "python project.py"
4 -> In training mode option type "reload" to train using the frames provided in "/frames/Train"

:::::::::Testing:::::::::

1-> Copy the folder containing the frames of the test video to be tested to location "/frames/Test/Test1"
2 -> Go to the folder which has "project.py"
3 -> Run it using the command "python project.py"
4 -> In training mode option type anything other than reload
5 -> It takes the chached copy of learned feature to test the new frames provided in "/frames/Test/Test1"

:::::::::Ouput:::::::::

1 -> After testing it'll prompt the user to press enter to watch the frames which are abnormal
2 -> After watching 1 can be pressed to reload the frames and watch again any number of times or any other key to test the next video
 
:::::::::::::::::::::::::::