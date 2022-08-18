
from tkinter import CASCADE
import cv2
import numpy as np
from PIL import Image, ImageOps
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class DigitalImaging:
    #define and initial classifiers as data members
    eye_classifier:CASCADE = \
        cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    face_classifier:CASCADE = \
        cv2.CascadeClassifier(cv2.data.haarcascades +
                              'haarcascade_frontalface_default.xml')

    def __init__(self) -> None:
        pass


    def convert_to_gs(self, path: str = "") -> Image:
        """convert_to_gs

        Args:
            path (str, optional): path to image. Defaults to "".

        Returns:
            Image: converted to grayscale
        """
        # open the image by the givven path
        with Image.open(path) as image:
            # convert to grayscale mode
            im = ImageOps.grayscale(image)
            print("image mode is: ",im.mode)
        return im

    def color_at(self, array: np.ndarray, i: int, j: int) -> tuple:
        # check if imgage is writble
        if array.flags.writeable:
            # return a tuple with the RGB in givven cordinate
            return tuple(array[i][j])
        pass

    def reduce_to(self, path: str = "", ch: str = "") -> Image:
        im = Image.open(path)
        image_array = np.array(im)
        if ch.upper() == "R":
            image_array[:, :, (1, 2)] = 0

        elif ch.upper() == "G":
            image_array[:, :, (0, 2)] = 0

        elif ch.upper() == "B":
            image_array[:, :, (0, 1)] = 0
        im = Image.fromarray(image_array)
        return im

    def make_collage(self, images: list = []) -> np.ndarray:
        # variable that holds the position in the tuple
        select_color = 0
        # colors tuple
        collors = ((1, 2), (0, 2), (0, 1))
        # initial the first color "RED"
        color = collors[0]
        # variable that hold the value of the count of the images
        count = 0
        # an array for the new image
        new_arr = []
        for image in images:
            # an expression to manage the color replace every 3 images
            if select_color % 3 == 0:
                color = collors[count]
                count = count+1 if count < 2 else 0
            # convert the current image to array
            image_array = np.array(image)
            # reduce the color of the image
            image_array[:, :, color] = 0
            select_color += 1
            # appending the new image array
            new_arr.append(image_array)
        concatenate = np.concatenate(tuple(new_arr), axis=0)
        return concatenate

    def shapes_dict(self, list_obj: list = []):
        order: OrderedDict = {}
        for index, image in enumerate(list_obj):
            # putting all images dimensions in dictionary by order
            order[index] = image.size
        return order

    def show_image(self, name):
        while True:
            cv2.imshow('image', name)
            key_pressed = cv2.waitKey(0)
            # if key_pressed & 27: # by default
            if key_pressed & ord('q'):  # q character is pressed
                break
        # cv2.destroyWindow('image') # release image window resources
        cv2.destroyAllWindows()

    def detect_obj(self, path: str = "", object: str = "",  im: Image = None) -> Image:
        # an expressin so we can handle to get image by object or by path
        if path != "" and im == None:
            img = cv2.imread(path)
        else:
            img = im
        # make a copy of the image
        copy = img.copy()
        # convert image mode to grayscale
        img_gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        # an expression to define the desire classifier by the class data members
        if object.upper() == "FACE":
            classifier = self.face_classifier
        elif object.upper() == "EYES":
            classifier = self.eye_classifier
        # the algorithm to detect an object in image
        image = classifier.detectMultiScale(img_gray,1.1,20)
        for (_x, _y, _w, _h) in image:
            cv2.rectangle(copy,
                          (_x, _y),  # upper-left corner
                          (_x+_w, _y+_h),  # lower-right corner
                          (0, 0, 0),
                          5)
        copy = Image.fromarray(copy)
        return copy

    def detect_obj_adv(self, path: str = "", face: bool = False, eyes: bool = False) -> Image:
        # first we check if the 'face' variable is true:
        if face:
            # detect the objects using are latest function
            img = di.detect_obj(path, "face")
            # then if the 'eyes' vatiable is true we detect them to
            if eyes:
                img = di.detect_obj("", "eyes", np.array(img))
        elif eyes:
            img = di.detect_obj(path, "Eyes")
        return img

    def detect_face_in_vid(self, path: str = "") -> None:
        # capturing the video using the givven path
        video = cv2.VideoCapture(path)
        while True:
            _, img = video.read()
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Detect the faces
            faces = self.face_classifier.detectMultiScale(gray, 1.1, 4)
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 0), 5)
            # Display
            cv2.imshow('img', img)
            # Stop if escape key is pressed
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        # Release the VideoCapture object
        video.release()


def __del__(self):
    print("Destroy")
    pass


if __name__ == "__main__":
    di = DigitalImaging()
    image_path = r'final_proj\data\mayim.jpeg'
    gray_img = di.convert_to_gs(image_path)
    img = Image.open(image_path)
    img.show()
    gray_img.show()
    img_arr = np.array(img)
    print(di.color_at(img_arr,4,5))
    green_img = di.reduce_to(image_path,"g")
    green_img.show()
    images_array = [img for i in range(10)]
    collage = di.make_collage(images_array)
    collage = Image.fromarray(collage)
    collage.show()
    print(di.shapes_dict(images_array))
    image_path = r'final_proj\data\pic.jpg'
    detect_eye = di.detect_obj(image_path,"Eyes")
    detect_eye.show()
    detect_face = di.detect_obj(image_path,"Face")
    detect_face.show()
    detect_adv = di.detect_obj_adv(image_path, True, True)
    detect_adv.show()
    vidoe_path = r'final_proj\data\vid.mp4'
    vid = di.detect_face_in_vid(vidoe_path)
    
    
    #video = cv2.imread(vid)
    #image_path = r'C:\Users\97252\Dropbox\cast.jpeg'
    #di.make_collage([image_path for i in range(12)])
    #cast_image = di.detect_obj_adv(image_path, True)
    #di.show_image(cast_image)
    #di.make_collage(image_path)
