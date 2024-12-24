
import sys
import dlib


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