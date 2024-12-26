
import sys
import dlib
import cv2

def get_rectangle_faces(image):
    detector = dlib.get_frontal_face_detector()
    
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # O argumento 1 indica que devemos ampliar a imagem uma vez para detectar mais rostos.
    dets = detector(img_rgb, 1)
    print(dets)
    return dets

def increase_rectangles(list_rectangles):
    modified_rectangles = []
    for d in list_rectangles:
        x1, y1 = d.left(), d.top()
        x2, y2 = d.right(), d.bottom()

        height = y2 - y1
        width = x2 - x1

        # Ajusta as dimensões
        # Aumenta a altura pela metade no topo e aumenta em um quinto embaixo
        new_y2 = y2 + height // 5
        new_y1 = y1 - height // 2
        # Aumenta a largura pela metade nas duas direções
        new_x1 = x1 - width // 2
        new_x2 = x2 + width // 2

        # Cria um novo retângulo com as dimensões ajustadas
        new_rectangle = dlib.rectangle(new_x1, new_y1, new_x2, new_y2)
        modified_rectangles.append(new_rectangle)
    return modified_rectangles

def show_rectangles(img, rectangles):
    # Copia a imagem para desenhar os retângulos
    img_with_rectangles = img.copy()

    print("Number of faces detected: {}".format(len(rectangles)))
    for i, d in enumerate(rectangles):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
        # Desenha o retângulo na imagem
        cv2.rectangle(img_with_rectangles, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 2)
    
    # Exibe a imagem com os retângulos
    cv2.imshow("Detected Faces", img_with_rectangles)
    cv2.waitKey(0)  # Aguarda uma tecla ser pressionada
    cv2.destroyAllWindows()  # Fecha todas as janelas

# Carrega a imagem e detecta os rostos
file=sys.argv[1]
print("Processing file: {}".format(file))
image = cv2.imread(file)
rectangles= get_rectangle_faces(image)

# Ajusta os retângulos
rectangles = increase_rectangles(rectangles)

# Exibe a imagem com os retângulos ajustados
show_rectangles(image, rectangles)

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