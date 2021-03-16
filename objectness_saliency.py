# USAGE
# python objectness_saliency.py --model objectness_trained_model --image images/barcelona.jpg

# import the necessary packages
import numpy as np
import argparse
import cv2



# load the input image
image = cv2.imread('pics/7.png')

# initialize OpenCV's objectness saliency detector and set the path
# to the input model files
saliency = cv2.saliency.ObjectnessBING_create()
#saliency.setTrainingPath(objectness_trained_model)

# compute the bounding box predictions used to indicate saliency
(success, saliencyMap) = saliency.computeSaliency(image)
numDetections = saliencyMap.shape[0]

# loop over the detections
for i in range(0, min(numDetections, 10)):
    # extract the bounding box coordinates
    (startX, startY, endX, endY) = saliencyMap[i].flatten()

    # randomly generate a color for the object and draw it on the image
    output = image.copy()
    color = np.random.randint(0, 255, size=(3,))
    color = [int(c) for c in color]
    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

    # show the output image
    cv2.imshow("Image", output)
    cv2.waitKey(0)