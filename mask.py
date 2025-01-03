import sys
import cv2
import numpy as np
import faceRectangle 
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer

def setup_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)
def rectangle_mask(image,image_shape):
    box_mask = []
    mask_list = []
    H,W,_ = image.shape
    rectangles_list=faceRectangle.get_rectangle_faces(image)
    rectangles_list=faceRectangle.increase_rectangles(rectangles_list, H, W)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for rect in rectangles_list:
        x = np.zeros(image_shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        # Preenche a área do retângulo na máscara
        box_mask.append([x1,x2,y1,y2])
        x = cv2.rectangle(x, (x1, y1), (x2, y2), 1, thickness=-1)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
        mask_list.append(x)

    return mask, box_mask, mask_list
def process_image(image_path):
    predictor=setup_model()
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    
    image_shape=list(image.shape)
    rectangles_mask, rectangle_list, box_list = rectangle_mask(image,image_shape)
    output = predictor(image_rgb)
    
    instances = output["instances"]
    masks = instances.pred_masks.cpu().numpy()
    classes = instances.pred_classes.cpu().numpy()

    person_class = 0
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for i, mask in enumerate(masks):
        if classes[i] == person_class:
            combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

    # Converter a imagem para RGBA
    highlighted_image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    alpha = 0.5  # Grau de transparência (0 = totalmente transparente, 1 = totalmente opaco)
    
    # Adicionar verde com transparência
    green_overlay = np.zeros_like(highlighted_image, dtype=np.uint8)
    green_overlay[combined_mask > 0] = [0, 255, 0, int(255 * alpha)]

    # Combinar a imagem original com o overlay verde transparente
    highlighted_image = cv2.addWeighted(highlighted_image, 1, green_overlay, alpha, 0)

    return image, cv2.bitwise_and(combined_mask,rectangles_mask), highlighted_image, rectangle_list, box_list

def display_and_save_results(image, combined_mask, highlighted_image, output_path):
    cv2.imshow("Original Image", image)
    cv2.imshow("Subject Mask", combined_mask * 255)
    cv2.imshow("Highlighted Image", highlighted_image)
    
    cv2.imwrite(output_path + "_mask.png", combined_mask * 255)
    cv2.imwrite(output_path + "_highlighted.png", highlighted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    image,combined_mask,highlighted_image,_ = process_image(sys.argv[1])
    print(combined_mask.shape, image.shape)
    display_and_save_results(image, combined_mask, highlighted_image, "imagens\output")
