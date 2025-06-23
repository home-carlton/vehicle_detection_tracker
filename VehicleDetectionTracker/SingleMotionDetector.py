import numpy as np
import imutils
import cv2

class SingleMotionDetector:
    def __init__(self, accumWeight=0.3, detectZone = None):
        # store the accumulated weight factor
        self.accumWeight = accumWeight
        # initialize the background model
        self.bg = None
        self._minFrames = 5
        self._frameCount = 0
        self._minx,self._miny,self._maxx,self._maxy = None,None,None,None
        
        if  isinstance(detectZone,(list, tuple)):
            self._minx,self._miny,self._maxx,self._maxy = detectZone
            
        
        
        
        
    def update(self, image):
        
        self._frameCount+=1
        # if the background model is None, initialize it
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        # update the background model by accumulating the weighted
        # average
        cv2.accumulateWeighted(image, self.bg, self.accumWeight)
        
    def detect(self, image, tVal=25):
        # compute the absolute difference between the background model
        # and the image passed in, then threshold the delta image
        
        if not self._minx is None:
            image = image[self._miny:self._maxy,self._minx:self._maxx]
        
        gray = self.gray(image)
        # capture first time
        if self._frameCount < self._minFrames:
            self.update(gray)
            return None
        
        
        delta = cv2.absdiff(self.bg.astype("uint8"), gray)
        thresh = cv2.threshold(delta, tVal, 255, cv2.THRESH_BINARY)[1]
        # perform a series of erosions and dilations to remove small
        # blobs
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # find contours in the thresholded image and initialize the
        # minimum and maximum bounding box regions for motion
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)
        
        
        
        # we could do this outside later, but we will do this anyway
        self.update(gray)
        
        
        # if no contours were found, return None
        if len(cnts) == 0:
            return None
        # otherwise, loop over the contours
        for c in cnts:
            # compute the bounding box of the contour and use it to
            # update the minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(c)
            (minX, minY) = (min(minX, x+1), min(minY, y+1))
            (maxX, maxY) = (max(maxX, x + w-1), max(maxY, y + h-1))
            
            
        if not self._minx is None:
            return (thresh,(self._minx+minX,self._miny,self._minx+maxX,self._miny+maxY))
        # otherwise, return a tuple of the thresholded image along
        # with bounding box
        return (thresh, (minX, minY, maxX, maxY))
    
    
    def gray(self, frame):
        
        #frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        
        return gray
    
    
    def applyMotionArea(self,frame, motion):

        # check to see if motion was found in the frame
        if motion is not None:
            # unpack the tuple and draw the box surrounding the
            # "motion area" on the output frame
            (thresh, (minX, minY, maxX, maxY)) = motion
            #print((minX, minY, maxX, maxY))
            cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                (0, 255, 0), 2)