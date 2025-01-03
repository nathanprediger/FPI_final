import cv2
import numpy as np
from stereographic import get_uniform_stereo_mesh
import matplotlib.pyplot as plt
from mask import process_image
from teste import Minimizer
from energy import resize_mask
import torch

import numpy as np

def resolve_mask_overlaps(box_list, rect_list):
    """
    Resolve sobreposição entre máscaras, mantendo a parte sobreposta da máscara 
    cujo centro está mais próximo da sobreposição.
    
    Args:
        box_list (list of np.ndarray): Lista de máscaras correspondentes a rect_list.
        rect_list (list of list): Lista de retângulos no formato [x1, x2, y1, y2].
    
    Returns:
        list of np.ndarray, list of list: Lista de máscaras atualizada e a lista de retângulos correspondente.
    """
    def calculate_center(rect):
        # Calcula o centro de um retângulo
        x1, x2, y1, y2 = rect
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def find_overlap_area(mask1, mask2):
        # Calcula a área de sobreposição entre duas máscaras
        return np.logical_and(mask1, mask2)

    updated_box_list = box_list.copy()

    for i in range(len(rect_list)):
        for j in range(i + 1, len(rect_list)):
            # Calcula a sobreposição entre as máscaras
            overlap = find_overlap_area(updated_box_list[i], updated_box_list[j])

            if np.any(overlap):  # Se houver sobreposição
                # Calcula os centros das duas máscaras
                center_i = calculate_center(rect_list[i])
                center_j = calculate_center(rect_list[j])

                # Calcula o centro da sobreposição
                overlap_indices = np.argwhere(overlap)
                overlap_center = np.mean(overlap_indices, axis=0)

                # Calcula as distâncias dos centros das máscaras à região de sobreposição
                distance_i = np.linalg.norm(np.array(center_i) - overlap_center)
                distance_j = np.linalg.norm(np.array(center_j) - overlap_center)

                # Determina qual máscara deve ser modificada
                if distance_i > distance_j:
                    updated_box_list[i] = np.logical_and(updated_box_list[i], ~overlap)
                else:
                    updated_box_list[j] = np.logical_and(updated_box_list[j], ~overlap)

    return updated_box_list, rect_list


def apply_mesh_warp(image, optimized_mesh, mesh_ds_ratio):
    """
    Aplica a malha otimizada para ajustar a distorção da imagem.

    Parâmetros:
        image: numpy.ndarray
            Imagem original.
        optimized_mesh: numpy.ndarray
            Malha otimizada para correção de distorção.
        mesh_ds_ratio: int
            Fator de redução da resolução da malha.

    Retorno:
        warped_image: numpy.ndarray
            Imagem corrigida.
    """
    H, W = image.shape[:2]

    # Redimensiona a malha otimizada para corresponder ao tamanho da imagem
    optimized_mesh_resized = cv2.resize(
        optimized_mesh.transpose(1, 2, 0),  # Transformar para (H_mesh, W_mesh, 2)
        (W, H),
        interpolation=cv2.INTER_LINEAR
    )

    # Divide a malha em coordenadas x e y
    map_x = optimized_mesh_resized[:, :, 0].astype(np.float32)
    map_y = optimized_mesh_resized[:, :, 1].astype(np.float32)

    # Ajusta os valores para o intervalo válido de índices de pixel
    map_x = (map_x + W // 2).clip(0, W - 1)
    map_y = (map_y + H // 2).clip(0, H - 1)

    # Aplica o remapeamento
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    return warped_image


if __name__ == "__main__":
    image_path = "data/14_116.jpg"
    mesh_ds_ratio = 10
    fov = 97
    Q = 4
    image, face_mask, highlighted_image, rect_list, box_list = process_image(image_path)
    box_list,rect_list=resolve_mask_overlaps(box_list,rect_list)
    H,W,_=image.shape
    half_diagonal = np.linalg.norm([H + 2 * Q * mesh_ds_ratio, W + 2 * Q * mesh_ds_ratio]) / 2.
    ra = half_diagonal / 2.
    rb = half_diagonal / (2 * np.log(99))
    # Gerar malhas uniforme e estereográfica
    uniform_mesh, stereo_mesh = get_uniform_stereo_mesh(image, np.pi*fov/180, Q, mesh_ds_ratio)
    # resized_mask = resize_mask(face_mask, uniform_mesh, mesh_ds_ratio)
    # resized_box_list = []  # Inicializa a lista vazia
    # for box in box_list:
    #     resized_box = resize_mask(box, uniform_mesh, mesh_ds_ratio)  # Resize mask
    #     resized_box = torch.from_numpy(resized_box)  # Converte para tensor PyTorch
    #     resized_box_list.append(resized_box)  # Adiciona à lista
    # resized_box_list = torch.stack(resized_box_list)
    _,Hm,Wm = uniform_mesh.shape 
    seg_mask = cv2.resize(face_mask.astype(np.float32), (Wm- 2 * Q, Hm- 2 * Q))
    box_masks = [cv2.resize(box_mask.astype(np.float32), (Wm- 2 * Q, Hm- 2 * Q)) for box_mask in box_list]
    box_masks = np.stack(box_masks, axis=0)
    seg_mask_padded = np.pad(seg_mask, [[Q, Q], [Q, Q]], "constant")
    box_masks_padded = np.pad(box_masks, [[0, 0], [Q, Q], [Q, Q]], "constant")
    print(seg_mask_padded.shape)
    print(box_masks_padded.shape)
    # print(seg_mask_padded.shape)
    # print(box_masks_padded[].shape)
    model = Minimizer(image, uniform_mesh, stereo_mesh, seg_mask_padded, Q, ra, rb, len(box_masks_padded), box_masks_padded)
    optim = torch.optim.Adam(model.parameters(), lr=0.5)

    # perform optimization
    print("optimizing")
    for i in range(200):
        optim.zero_grad()
        loss = model.forward()
        print("step {}, loss = {}".format(i, loss.item()))
        loss.backward()
        optim.step()
    optimized_mesh = model.vertices.detach().cpu().numpy()
    optimized_mesh = optimized_mesh[:, Q:-Q, Q:-Q]
    print(optimized_mesh.shape)
    warped_image = apply_mesh_warp(image, optimized_mesh, mesh_ds_ratio)
    
    print(rect_list)
    



    # # Empilha todos os tensores na lista em um tensor PyTorch
    

    # # Criar malha otimizada
    # optimized_mesh = create_optimized_mesh(image, uniform_mesh, stereo_mesh, resized_mask, Q, ra, rb, 20000, len(rect_list), resized_box_list)
    # # Aplicar a malha otimizada na imagem
    # Hm, Wm = optimized_mesh.shape[1:]
    # optimized_mesh = optimized_mesh[:, Q:-Q, Q:-Q]
    # X_distorted, Y_distorted = uniform_mesh
    # plt.figure(figsize=(12, 18))
    # plt.plot(X_distorted, Y_distorted, color="blue", linewidth=0.5)  # Vertical lines
    # plt.plot(X_distorted.T, Y_distorted.T, color="blue", linewidth=0.5)  # Horizontal lines
    # plt.axis("equal")
    # plt.axis("off")
    # plt.show()

    # corrected_image = apply_mesh_warp(image, optimized_mesh, mesh_ds_ratio)
    
    X_distorted, Y_distorted = optimized_mesh
    plt.figure(figsize=(12, 18))
    plt.plot(X_distorted, Y_distorted, color="blue", linewidth=0.5)  # Vertical lines
    plt.plot(X_distorted.T, Y_distorted.T, color="blue", linewidth=0.5)  # Horizontal lines
    plt.axis("equal")
    plt.axis("off")
    plt.show()
    # Exibir resultados
    cv2.imwrite("output.jpg", warped_image)
    cv2.imshow("Imagem Original", warped_image)
    #cv2.imshow("Imagem Corrigida", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
