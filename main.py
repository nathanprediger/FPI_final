import cv2
import numpy as np
from stereographic import get_uniform_stereo_mesh
import matplotlib.pyplot as plt
from mask import process_image
from teste import create_optimized_mesh, interpolate_mesh
from energy import resize_mask
import torch



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
    image_path = "imagens/teste3.jpg"
    mesh_ds_ratio = 10
    fov = (97 + 30)/2
    Q = 4
    image, face_mask, highlighted_image, rect_list, box_list = process_image(image_path)
    print(rect_list)
    H,W,_=image.shape
    half_diagonal = np.linalg.norm([H + 2 * Q * mesh_ds_ratio, W + 2 * Q * mesh_ds_ratio]) / 2.
    ra = half_diagonal / 2.
    rb = half_diagonal / (2 * np.log(99))
    # Gerar malhas uniforme e estereográfica
    uniform_mesh, stereo_mesh = get_uniform_stereo_mesh(image, np.pi*fov/180, Q, mesh_ds_ratio)
    resized_mask = resize_mask(face_mask, uniform_mesh, mesh_ds_ratio)
    resized_box_list = []  # Inicializa a lista vazia
    for box in box_list:
        resized_box = resize_mask(box, uniform_mesh, mesh_ds_ratio)  # Resize mask
        resized_box = torch.from_numpy(resized_box)  # Converte para tensor PyTorch
        resized_box_list.append(resized_box)  # Adiciona à lista

    # Empilha todos os tensores na lista em um tensor PyTorch
    resized_box_list = torch.stack(resized_box_list)

    # Criar malha otimizada
    optimized_mesh = create_optimized_mesh(image, uniform_mesh, stereo_mesh, resized_mask, Q, ra, rb, 20000, len(rect_list), resized_box_list)
    # Aplicar a malha otimizada na imagem
    Hm, Wm = optimized_mesh.shape[1:]
    optimized_mesh = optimized_mesh[:, Q:-Q, Q:-Q]
    X_distorted, Y_distorted = uniform_mesh
    plt.figure(figsize=(12, 18))
    plt.plot(X_distorted, Y_distorted, color="blue", linewidth=0.5)  # Vertical lines
    plt.plot(X_distorted.T, Y_distorted.T, color="blue", linewidth=0.5)  # Horizontal lines
    plt.axis("equal")
    plt.axis("off")
    plt.show()

    corrected_image = apply_mesh_warp(image, optimized_mesh, mesh_ds_ratio)
    
    X_distorted, Y_distorted = optimized_mesh
    plt.figure(figsize=(12, 18))
    plt.plot(X_distorted, Y_distorted, color="blue", linewidth=0.5)  # Vertical lines
    plt.plot(X_distorted.T, Y_distorted.T, color="blue", linewidth=0.5)  # Horizontal lines
    plt.axis("equal")
    plt.axis("off")
    plt.show()
    # Exibir resultados
    cv2.imwrite("output.jpg", corrected_image)
    cv2.imshow("Imagem Original", image)
    #cv2.imshow("Imagem Corrigida", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
