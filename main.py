import cv2
import numpy as np
from energy import create_optimized_mesh
from stereographic import get_uniform_stereo_mesh
from mask import process_image

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
    fov = 97
    Q = 4
    image, face_mask, highlighted_image, rect_list = process_image(image_path)
    H,W,_=image.shape
    half_diagonal = np.linalg.norm([H + 2 * Q * mesh_ds_ratio, W + 2 * Q * mesh_ds_ratio]) / 2.
    ra = half_diagonal / 2.
    rb = half_diagonal / (2 * np.log(99))
    # Gerar malhas uniforme e estereográfica
    uniform_mesh, stereo_mesh = get_uniform_stereo_mesh(image, np.pi*fov/180, Q, mesh_ds_ratio)

    # Criar malha otimizada
    optimized_mesh = create_optimized_mesh(image, uniform_mesh, stereo_mesh, face_mask, rect_list, ra, rb, 1000)
  
    # Aplicar a malha otimizada na imagem
    corrected_image = apply_mesh_warp(image, optimized_mesh, mesh_ds_ratio)

    # Exibir resultados
    cv2.imshow("Imagem Original", image)
    cv2.imshow("Imagem Corrigida", corrected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
