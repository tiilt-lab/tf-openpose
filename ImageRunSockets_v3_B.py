import argparse
import logging
import csv
from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

import os
import threading
import subprocess
import face_recognition
from PIL import Image
from struct import unpack
import time
import sys
import glob
from collections import deque
import sys
from twisted.python import log
from twisted.internet import reactor
log.startLogging(sys.stdout)

from autobahn.twisted.websocket import WebSocketServerFactory
from autobahn.twisted.websocket import WebSocketServerProtocol


class MyServerProtocol(WebSocketServerProtocol):

    def __init__(self, process_pose=False, process_faces=False, saveLocation='D:/AimeeImageResults/PulledImages/'):
        super().__init__()
        
        self.process_pose = process_pose
        self.process_faces = process_faces
        #Deletes all previously stored images on the computer
        self.resetNewLaunch()
        #List of faces that have been seen before
        self.faces = []
        #List of images that will be processed
        self.images = []
        #Number of images required to run the processing (face recognition)
        self.imageWait = 1
        #Number of images that have not been processed
        self.imageCount = 0
        #Dictionary of ips with a cooresponding image name
        self.names = {}
        #List of people seen and the time at which they were seen 
        self.peopleSeen = open("D:/AimeeImageResults/peopleSeen.txt","w")
        self.curr_file_name= {}
        self.thread1 = threading.Thread(target = self.recognizeAndReset)
        self.thread1.start()
        #Location to save the images and results
        self.saveLocation = saveLocation

            
    def onConnect(self, request):
        ip = request.peer
        if ip not in self.names:
            self.names[ip]=[]
        if ip not in self.curr_file_name:
            self.curr_file_name[ip]=None

    def onMessage(self, payload, isBinary):
        ## echo back message verbatim
        print("message received")
        ip = self.peer

        #If the string name is recieved
        if not isBinary:
            print(payload.decode('utf8'))
            self.curr_file_name[ip] = payload.decode('utf8')

        #If the data for the image is sent     
        else:
            imageName = ip.replace(":","") + self.curr_file_name[ip]
            self.imageCount += 1
            #Save image to the specified save location
            imgFile = open(os.path.join(self.saveLocation, imageName), 'wb')
            imgFile.write(payload)
            imgFile.close()
            self.names[ip].append(imageName)
            self.curr_file_name[ip] = None
            print('image data')
            #If there are enough images to process, process them
            self.recognizeAndReset()
                
    def onClose(self, wasClean, code, reason):
      pass
      #print("WebSocket connection closed: {}".format(reason))

    def onOpen(self):
        print("connection opened")

    def recognizeAndReset(self):
        if self.imageCount >= self.imageWait:
            self.imageCount = 0
            prepImages(self)
            faceRecognize(self.images,self)
            openpose(self.images,self)
            resetImages(self)

    #Removes and resets list for the iamge obtaining and processing procedure to begin again
    def resetNewLaunch(self):
        #Delete Images in image folder
        for doc in glob.glob("D:/AimeeImageResults/Images/tcp*"):    
            os.remove(doc)
        for doc in glob.glob("D:/AimeeImageResults/PulledImages/tcp*"):    
            os.remove(doc)

#Prepares images for processing
def prepImages(self):
    for name in self.names:
        more = True
        while more:
            if len(self.names[name]) > 0:
                #take name off of name list so that it isnt processed again
                imageName = self.names[name].pop(0)
                self.images.append(imageName)
                #Move images from the pulled images folder to the images folder for openpose to process a folder
                os.rename("D:/AimeeImageResults/PulledImages/" + imageName, "D:/AimeeImageResults/Images/" + imageName)            
            else:
                 more = False

#Removes and resets list for the iamge obtaining and processing procedure to begin again
def resetImages(self):
    self.images = []
    #Delete Images in image folder
    for doc in glob.glob("D:/AimeeImageResults/Images/tcp*"):    
        os.remove(doc)


#Runs each image through openpose and saves skeletal data
def openpose(images,self):   
    logger = logging.getLogger('TfPoseEstimator-Video')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fps_time = 0
    model = 'mobilenet_thin'

    w = 432
    h = 368
    e = TfPoseEstimator(get_graph_path(self.model), target_size=(self.w, self.h))
    logger.debug('cam read+')

    #cam = cv2.VideoCapture(args.camera)

    with open('D:/AimeeImageResults/bodyparts.csv', 'a', newline='') as csvfile:
        myFields = ['frameNumber', 'personNumber', 'partNumber', 'xCoordinate', 'yCoordinate', 'score']
        partwriter = csv.DictWriter(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL, fieldnames=myFields)
        partwriter.writeheader()
        frameNum = 0
        imCount = 0
        for imageName in images:
            #Pull image from server
            #im = Image.open("D:/AimeeImageResults/Images/"+ imageName)

            image = common.read_imgfile("D:/AimeeImageResults/Images/"+ imageName, None, None)
            if image is None:
                self.logger.error('Image can not be read, path=%s' % args.image)
                sys.exit(-1)
            t = time.time()
            humans = self.e.inference(image, resize_to_default=(self.w > 0 and self.h > 0), upsample_size=4.0)
            elapsed = time.time() - t
            imCount += 1
            self.logger.info('inference image: %s in %.4f seconds.' % ("D:/AimeeImageResults/Images/"+ imageName, elapsed))
            
            for personNum in range(len(humans)):
                for partKey, part in humans[personNum].body_parts.items():
                   partwriter.writerow({'frameNumber':imageName,
                                        'personNumber':personNum,
                                        'partNumber':partKey,
                                        'xCoordinate':part.x,
                                        'yCoordinate':part.y,
                                        'score':part.score})
            
            self.logger.debug('show+')
    ##            #image = np.zeros(image.shape)#this gets rid of background picture
    ##            image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    ##            cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)),(10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    ##            cv2.imshow('tf-pose-estimation result', image)
    ##            cv2.waitKey(0)
            fps_time = time.time()
    cv2.destroyAllWindows()


#Finds a face in an image, if there is one, and determines if that face has been seen before    
def faceRecognize(images,self):
    imCount = 0
    for imageName in images:
        #Pull image from server
        time1 = time.time()
        image = face_recognition.load_image_file("D:/AimeeImageResults/Images/" + imageName)
        im = Image.open("D:/AimeeImageResults/Images/"+ imageName)
        imCount += 1
        peopleSeen = open("D:/AimeeImageResults/peopleSeen.txt",'a')
        print("Image open and retrieval took " + str(time.time() - time1) + " seconds")
        time3 = time.time()
        faceLocations = face_recognition.face_locations(image)
        print("Face Locations took " + str(time.time() - time3) + " seconds")
        #Crop and store each face found in the image
        for face in range(0,len(faceLocations)):
            top, right, bottom, left = faceLocations[face]
            cropSection = (left, top, right, bottom)
            cropped = im.crop(cropSection)
            time2 = time.time()
            cropEncodings = face_recognition.face_encodings(image)
            print("Face encoding took " + str(time.time() - time2) + " seconds")
            #See if a similar face was already found
            for unknownFace in cropEncodings:
                found = False
                for knownFace in range(0,len(self.faces)):
                    #Compare Faces to exisiting ones
                    time4 = time.time()
                    if face_recognition.compare_faces([unknownFace],self.faces[knownFace]) and not found:
                        print("Face Comparison took " + str(time.time() - time4) + " seconds")
                        found = True
                        print("Person " + str(knownFace) + ": found in image")
                        #Write which face was seen and the image it was seen in (thus giving the time the image was taken)
                        peopleSeen.write(str(knownFace) + "found at " + imageName + "\n")
                    #If no face match in the database was found, add face to database 
                if not found:
                    print("New face added")
                    self.faces.append(unknownFace)
                    peopleSeen.write("Person " + str(len(self.faces)-1) + " found (for the first time) at " + imageName + "\n")
            #cropped.save("C:/Users/TIILTMAINPC/Desktop/NoahImageResults/Faces/" + "Face " + str(face) + "-" + ips[j] + imageName)
    if not(imCount == 0):          
        im.close()



if __name__ == "__main__":
    factory = WebSocketServerFactory()
    factory.protocol = MyServerProtocol

    reactor.listenTCP(45000, factory)
    reactor.run()
