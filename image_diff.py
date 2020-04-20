.x
#%%
import argparse
import io

from google.cloud import vision
from google.cloud.vision import types

#%%
# USAGE
# python image_diff.py --first images/original_01.png --second images/modified_01.png

# import the necessary packages
import numpy as np
import urllib
from skimage import io as io_img
# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
    req = urllib.request.Request(url, headers={'User-Agent' : "Magic Browser"}) 
    resp = urllib.request.urlopen(req)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image


from skimage.measure import compare_ssim
import argparse
#import imutils
#import cv2


import argparse
import io
import os
from google.cloud import vision
from google.cloud.vision import types

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "AdVerifai-fdb018f57d00.json"

def annotate(path):
    """Returns web annotations given the path to an image."""
    client = vision.ImageAnnotatorClient()

    if path.startswith('http') or path.startswith('gs:'):
        image = types.Image()
        image.source.image_uri = path

    else:
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

    web_detection = client.web_detection(image=image).web_detection
    #print(web_detection)
    return web_detection


def report(annotations):
    """Prints detected features in the provided web annotations."""
    if annotations.pages_with_matching_images:
        print('\n{} Pages with matching images retrieved'.format(
            len(annotations.pages_with_matching_images)))

        for page in annotations.pages_with_matching_images:
            print('Url   : {}'.format(page.url))

    if annotations.full_matching_images:
        print ('\n{} Full Matches found: '.format(
               len(annotations.full_matching_images)))

        for image in annotations.full_matching_images:
            print('Url  : {}'.format(image.url))

    if annotations.partial_matching_images:
        print ('\n{} Partial Matches found: '.format(
               len(annotations.partial_matching_images)))

        for image in annotations.partial_matching_images:
            print('Url  : {}'.format(image.url))

    if annotations.web_entities:
        print ('\n{} Web entities found: '.format(
            len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print('Score      : {}'.format(entity.score))
            print('Description: {}'.format(entity.description))

res=annotate("https://africacheck.org/wp-content/uploads/2014/07/000_Par3331148.jpg")
report(res)
#%%
imageA = url_to_image("https://us-east-1.tchyn.io/snopes-production/uploads/2017/09/nfl-burns-flag.jpg")
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
for image in res.full_matching_images: 
    print('Url  : {}'.format(image.url))
    imageB = url_to_image(image.url)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    try:
        print(grayA.shape)
        print(grayB.shape)
        (score, diff) = compare_ssim(grayA, grayB, full=True)
    except Exception as e:
        print(str(e))
        continue
    print("SSIM: {}".format(score))
    if score==1.0:
        continue
    diff = (diff * 255).astype("uint8")
    # threshold the difference image, followed by finding contours to
    # obtain the regions of the two input images that differ
    thresh = cv2.threshold(diff, 0, 255,
    	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    
    # loop over the contours
    for c in cnts:
    	# compute the bounding box of the contour and then draw the
    	# bounding box on both input images to represent where the two
    	# images differ
    	(x, y, w, h) = cv2.boundingRect(c)
    	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # show the output images
    cv2.imshow("Original", imageA)
    cv2.imshow("Modified", imageB)
    cv2.imshow("Diff", diff)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

#%%
pra)x
#%%
# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--first", required=True,	help="first input image")
#ap.add_argument("-s", "--second", required=True,	help="second")
#args = vars(ap.parse_args())

# load the two input images
imageA = cv2.imread("C:/Users/olevi/.spyder2-py3/adverifai/image-difference/images/original_01.png")#args["first"])
imageB = cv2.imread("C:/Users/olevi/.spyder2-py3/adverifai/image-difference/images/modified_01.png")#args["second"])
#%%
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# loop over the contours
for c in cnts:
	# compute the bounding box of the contour and then draw the
	# bounding box on both input images to represent where the two
	# images differ
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# show the output images
cv2.imshow("Original", imageA)
cv2.imshow("Modified", imageB)
cv2.imshow("Diff", diff)
cv2.imshow("Thresh", thresh)
cv2.waitKey(0)



