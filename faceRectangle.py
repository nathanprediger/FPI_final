import cv2
import detectron2
import torch
import matplotlib
import sys
import dlib
#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image.  In
#   particular, it shows how you can take a list of images from the command
#   line and display each on the screen with red boxes overlaid on each human
#   face.
#
#   The examples/faces folder contains some jpg images of people.  You can run
#   this program on them and see the detections by executing the
#   following command:
#       ./face_detector.py ../examples/faces/*.jpg
#
#   This face detector is made using the now classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image
#   pyramid, and sliding window detection scheme.  This type of object detector
#   is fairly general and capable of detecting many types of semi-rigid objects
#   in addition to human faces.  Therefore, if you are interested in making
#   your own object detectors then read the train_object_detector.py example
#   program.  
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake installed.  On Ubuntu, this can be done easily by running the
#   command:
#       sudo apt-get install cmake
#
#   Also note that this example requires Numpy which can be installed
#   via the command:
#       pip install numpy


def get_rectangle_faces(file):
    detector = dlib.get_frontal_face_detector()
    win = dlib.image_window()
    
    print("Processing file: {}".format(file))
    img = dlib.load_rgb_image(file)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    dets = detector(img, 1)
    print(dets)
    return dets
def increase_rectangles(list_rectangles):
    modified_rectangles=[]
    for d in list_rectangles:
        # Coordenadas originais
        x1, y1 = d.left(), d.top()
        x2, y2 = d.right(), d.bottom()

        # Calcula a altura e largura originais
        height = y2 - y1
        width = x2 - x1

        # Ajusta as dimensões
        # Aumenta a altura em 2 vezes na direção do topo (diminui y1)
        new_y2 = y2 + height // 5
        new_y1 = y1 - height // 2
        # Aumenta a largura pela metade nas duas direções (expande x1 e x2)
        new_x1 = x1 - width // 2
        new_x2 = x2 + width // 2

        # Cria um novo retângulo com as dimensões ajustadas
        new_rectangle = dlib.rectangle(new_x1, new_y1, new_x2, new_y2)
        modified_rectangles.append(new_rectangle)
    return modified_rectangles
def show_rectangles(file, rectangles):
    win = dlib.image_window()
    
    print("Processing file: {}".format(file))
    img = dlib.load_rgb_image(file)
    print("Number of faces detected: {}".format(len(rectangles)))
    for i, d in enumerate(rectangles):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        
    win.clear_overlay()
    win.set_image(img)
    win.add_overlay(rectangles)
    dlib.hit_enter_to_continue()
    
rectangles=get_rectangle_faces(sys.argv[1])
rectangles=increase_rectangles(rectangles)
show_rectangles(sys.argv[1], rectangles)


# Finally, if you really want to you can ask the detector to tell you the score
# for each detection.  The score is bigger for more confident detections.
# The third argument to run is an optional adjustment to the detection threshold,
# where a negative value will return more detections and a positive value fewer.
# Also, the idx tells you which of the face sub-detectors matched.  This can be
# used to broadly identify faces in different orientations.
# if (len(sys.argv[1:]) > 0):
#     img = dlib.load_rgb_image(sys.argv[1])
#     dets, scores, idx = detector.run(img, 1, -1)
#     for i, d in enumerate(dets):
#         print("Detection {}, score: {}, face_type:{}".format(
#             d, scores[i], idx[i]))