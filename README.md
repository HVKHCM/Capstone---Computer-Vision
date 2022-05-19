# Senior Capstone - Spring 2022: Using Computer Vision to Teach Number Line
## Authors: Khang Vo Huynh, Luke Malek
## Mentor: Professor Olaf Hall-Holt
## MSCS department, St. Olaf College, Northfield, Minnesota, United States of America

### Requirement
* [Python 3.x.x](https://www.python.org/)
* [scipy](https://scipy.org/)
* [mediapipe](https://google.github.io/mediapipe/)
* [openCV](https://opencv.org/)
* [skimage](https://scikit-image.org/)
* [matplotlib](https://matplotlib.org/)
* [numpy](https://numpy.org/)

### How to run the process
First pull this repository into a folder. Then, open your favorite code editor and open up the **FullProcess.py**. Inside the file, changing the values of the variable **cap** to your desired video. Finally, save and run it by typing **python FullProcess.py** in your terminal or **F5** if you are using Python IDLE.

While the program is running, press **D** to move to the next frame of the video and press **P** to show a visualization of the algorithm. Finally, if you want to quit the current process, press **Q**.

### Driver files
#### tester.py
Within **tester.py** is a function called **process(img)**. This is where we put together many separate algorithms to produce a complete end-to-end result. The nuances of helper functions will be described below.
#### lineGeom.py
This houses many small functions which we use for line and point manipulation. 
#### imgFunction.py
The bulk of our algorithm lives in this file. Specifically, we highlight the functionality of certain important algorithms.
##### rulerLines(img)
This function takes in an image which features a number line and returns a pair of bounding lines. 
##### rulerMarks(line1, line2, img)
This function must be given a raw image and two lines within the image. Mark detection is run on the cropped region between the two provided lines. This function returns an array of 2-D coordinates, each a mark on the number line.
##### fingertip_coordinate(img)
This function takes in an image and returns the 2-D coordinate of the tip of the index finger. 



