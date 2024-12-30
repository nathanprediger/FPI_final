import numpy as np
from mask import process_image
from stereographic import get_uniform_stereo_mesh
from scipy.optimize import minimize
import cv2
import dlib
count = 0
def calculate_similarity_transform(source_points, target_points):
    """Calcula a transformação de similaridade."""
    if not source_points.size or not target_points.size: # Verifica se os arrays estão vazios
        return np.eye(2), np.zeros(2) # Retorna identidade e zero se vazios

    mean_source = np.mean(source_points, axis=0)
    mean_target = np.mean(target_points, axis=0)

    centered_source = source_points - mean_source
    centered_target = target_points - mean_target

    cov_matrix = centered_target.T @ centered_source

    U, S, V = np.linalg.svd(cov_matrix)
    R = U @ V
    scale = np.trace(cov_matrix) / np.trace(centered_source.T @ centered_source) if np.trace(centered_source.T @ centered_source) != 0 else 1
    S = scale * R
    t = mean_target - S @ mean_source

    return S, t
import numpy as np
import dlib

def calculate_energy_f(mask, mesh_estereo, mesh_perspective, rectangle_list, r_a, r_b, W, H):
    num_rects = len(rectangle_list)
    if num_rects < 2:  # Não há pares para comparar se houver menos de 2 retângulos
        return 0.0

    # Pré-calcula os centros dos retângulos
    centers = np.zeros((num_rects, 2), dtype=np.float32)
    for i, rect in enumerate(rectangle_list):
        centers[i] = np.array([(rect.left() + rect.right()) / 2, (rect.top() + rect.bottom()) / 2])

    energia_total = 0.0
    for i in range(num_rects):
        for j in range(i + 1, num_rects):
            center_i = centers[i]
            center_j = centers[j]

            # Calcula a distância entre os centros (exemplo)
            dist = np.linalg.norm(center_i - center_j)

            # Calcula a área de interseção dos retângulos (exemplo)
            rect_i = rectangle_list[i]
            rect_j = rectangle_list[j]

            x_overlap = max(0, min(rect_i.right(), rect_j.right()) - max(rect_i.left(), rect_j.left()))
            y_overlap = max(0, min(rect_i.bottom(), rect_j.bottom()) - max(rect_i.top(), rect_j.top()))
            overlap_area = x_overlap * y_overlap
            
            # Combina distância e sobreposição na energia (exemplo)
            energia_total += (r_a * dist + r_b * overlap_area)  # Ajuste r_a e r_b conforme necessário

    return energia_total

def calculate_energy_b(vertices, mesh_uniform):
    _, H, W = mesh_uniform.shape
    energia_total = 0.0

    # Calcula as diferenças entre os vértices e seus vizinhos (vetorizado)
    diff_i = vertices[:, 1:, :] - vertices[:, :-1, :]  # Diferenças verticais
    diff_j = vertices[:, :, 1:] - vertices[:, :, :-1]  # Diferenças horizontais

    p_diff_i = mesh_uniform[:, 1:, :] - mesh_uniform[:, :-1, :]
    p_diff_j = mesh_uniform[:, :, 1:] - mesh_uniform[:, :, :-1]

    # Calcula os vetores unitários
    e_ij_i = p_diff_i / np.linalg.norm(p_diff_i, axis=0)
    e_ij_j = p_diff_j / np.linalg.norm(p_diff_j, axis=0)
    
    #Calcula a energia usando cross product vetorizado
    energia_total += np.sum(np.linalg.norm(np.cross(diff_i, e_ij_i, axis=0), axis=0)**2)
    energia_total += np.sum(np.linalg.norm(np.cross(diff_j, e_ij_j, axis=0), axis=0)**2)

    return energia_total

def calculate_energy_r(vertices):
    vertices = np.array(vertices)  # Garante que vertices seja um array NumPy
    diffs = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
    energia_total = np.sum(np.linalg.norm(diffs, axis=2)**2) / 2 #divide por dois para não contar a distancia duas vezes
    return energia_total
def calculate_energy_a(vertices, W, H):
    """
    Calcula o termo de extensão de bordas E_a.

    Parâmetros:
    - vertices: ndarray (2, H, W)
        Coordenadas otimizadas (v_i) na malha.
    - W, H: int
        Largura e altura da imagem.

    Retorno:
    - energia_total: float
        O valor total da energia \(E_a\).
    """
    # Inicializa a energia total
    energia_total = 0.0

    # Borda esquerda
    for i in range(H):
        v_i_x = vertices[0, i, 0]  # Coordenada x na borda esquerda
        if v_i_x > 0:
            energia_total += v_i_x**2

    # Borda direita
    for i in range(H):
        v_i_x = vertices[0, i, W-1]  # Coordenada x na borda direita
        if v_i_x < W:
            energia_total += (v_i_x - W)**2

    # Borda superior
    for j in range(W):
        v_i_y = vertices[1, 0, j]  # Coordenada y na borda superior
        if v_i_y > 0:
            energia_total += v_i_y**2

    # Borda inferior
    for j in range(W):
        v_i_y = vertices[1, H-1, j]  # Coordenada y na borda inferior
        if v_i_y < H:
            energia_total += (v_i_y - H)**2

    return energia_total
def extend_mesh(vertices, padding=4):
    """
    Adiciona vértices de padding nas bordas da malha.

    Parâmetros:
    - vertices: ndarray (2, H, W)
        Malha original.
    - padding: int
        Número de vértices de padding a serem adicionados em cada lado.

    Retorno:
    - vertices_extended: ndarray (2, H+2*padding, W+2*padding)
        Malha estendida.
    """
    _, H, W = vertices.shape
    H_ext, W_ext = H + 2 * padding, W + 2 * padding

    # Cria uma malha estendida preenchida inicialmente com zeros
    vertices_extended = np.zeros((2, H_ext, W_ext))

    # Copia a malha original para o centro da malha estendida
    vertices_extended[:, padding:H+padding, padding:W+padding] = vertices

    # Preenche as bordas
    vertices_extended[:, :padding, :] = vertices_extended[:, padding:padding+1, :]  # Topo
    vertices_extended[:, -padding:, :] = vertices_extended[:, -padding-1:-padding, :]  # Base
    vertices_extended[:, :, :padding] = vertices_extended[:, :, padding:padding+1]  # Esquerda
    vertices_extended[:, :, -padding:] = vertices_extended[:, :, -padding-1:-padding]  # Direita

    return vertices_extended
def gaussian_kernel(distance_squared, h=2.37):
    """
    Calcula o kernel Gaussiano dado a distância ao quadrado.

    Parâmetros:
    - distance_squared: float ou ndarray
        Distância euclidiana ao quadrado.
    - h: float
        Parâmetro de largura de banda do kernel.

    Retorno:
    - kernel_value: float ou ndarray
        Valor do kernel Gaussiano.
    """
    return np.exp(-distance_squared / (2 * h**2))

def interpolate_mesh(mesh_perspective, mesh_stereographic, mask):
    """
    Interpola entre as malhas perspectiva e estereográfica com base na equação (15).

    Parâmetros:
    - mesh_perspective: ndarray (2, H, W)
        Malha inicial da projeção perspectiva.
    - mesh_stereographic: ndarray (2, H, W)
        Malha inicial da projeção estereográfica.
    - mask: ndarray (h, w)
        Máscara binária onde 1 indica regiões faciais (dimensão menor que H, W).

    Retorno:
    - mesh_interpolated: ndarray (2, H, W)
        Malha interpolada.
    """
    H, W = mesh_perspective.shape[1:]

    # Redimensiona a máscara para o tamanho das malhas
    mask_resized = np.resize(mask, (H, W))

    mesh_interpolated = np.zeros_like(mesh_perspective)

    # Calcula os deslocamentos \Delta v_j
    delta_v = np.zeros_like(mesh_perspective)
    delta_v[:, mask_resized == 1] = mesh_stereographic[:, mask_resized == 1] - mesh_perspective[:, mask_resized == 1]

    # Itera sobre todos os pontos da malha
    for i in range(H):
        for j in range(W):
            if mask_resized[i, j] == 0:
                # Ignora pontos fora da máscara
                mesh_interpolated[:, i, j] = mesh_perspective[:, i, j]
                continue

            # Calcula a distância ao quadrado entre (i, j) e todos os outros pontos
            pi = np.array([i, j])
            pj = np.array(np.meshgrid(np.arange(H), np.arange(W), indexing='ij'))
            pj = pj.reshape(2, -1).T

            distances_squared = np.sum((pj - pi)**2, axis=1).reshape(H, W)

            # Aplica o kernel Gaussiano
            weights = gaussian_kernel(distances_squared)
            weights *= mask_resized  # Aplica a máscara

            # Calcula o numerador e o denominador
            numerator = np.sum(weights * delta_v, axis=(1, 2))
            denominator = np.sum(weights)

            if denominator > 0:
                delta_vi_0 = numerator / denominator
            else:
                delta_vi_0 = 0

            # Atualiza o ponto interpolado
            mesh_interpolated[:, i, j] = mesh_perspective[:, i, j] + delta_vi_0

    return mesh_interpolated


def calculate_total_energy(mask, mesh_estereo, mesh_perspective, vertices, rectangle_list, r_a, r_b, W, H):
    energia_f = calculate_energy_f(mask, mesh_estereo, vertices, rectangle_list, r_a, r_b, W, H)
    energia_b = calculate_energy_b(vertices, mesh_perspective)
    energia_r = calculate_energy_r(vertices)
    energia_a = calculate_energy_a(vertices, W, H)

    lambda_f, lambda_b, lambda_r, lambda_a = 4, 2, 0.5, 4
    energia_total = (
        lambda_f * energia_f +
        lambda_b * energia_b +
        lambda_r * energia_r +
        lambda_a * energia_a
    )
    print(energia_total)
    return energia_total
def energy_objective(vertices_flat, mesh_estereo, mask, mesh_perspective, rectangle_list, r_a, r_b, W, H):
    """Função objetivo para a otimização, agora com suporte a rectangle_list."""
    vertices = vertices_flat.reshape(2, H, W)
    global count
    
    # Verifica se há índices na máscara
    mask_indices = np.argwhere(mask == 1)
    if not mask_indices.size:
        return 1e10  # Retorna um valor alto de energia para evitar erros
    
    #print(count)
    count += 1

    # Calcula a energia total
    return calculate_total_energy(mask, mesh_estereo, mesh_perspective, vertices, rectangle_list, r_a, r_b, W, H)
def create_optimized_mesh(image, uniform_mesh, stereo_mesh, face_mask, rectangle_list, r_a=50, r_b=10, max_iterations=100):
    """
    Cria a malha otimizada para correção de distorção local em imagens de grande angular.

    Parâmetros:
    - image: ndarray
        Imagem original de entrada.
    - uniform_mesh: ndarray
        Malha uniforme para projeção perspectiva.
    - stereo_mesh: ndarray
        Malha estereográfica.
    - face_mask: ndarray
        Máscara binária identificando regiões faciais.
    - rectangle_list: list
        Lista de retângulos das áreas de rosto detectadas.
    - r_a, r_b: float
        Parâmetros de peso para os termos de distância e sobreposição na energia facial.
    - max_iterations: int
        Número máximo de iterações para o otimizador.

    Retorno:
    - optimized_mesh: ndarray
        Malha otimizada após minimização da energia.
    """
    # Obtém as dimensões da malha
    H_mesh, W_mesh = uniform_mesh.shape[1:]

    # Inicializa os vértices pela interpolação entre as malhas perspectiva e estereográfica
    vertices_initial = interpolate_mesh(uniform_mesh, stereo_mesh, face_mask)
    vertices_initial_flat = vertices_initial.reshape(-1)

    # Define a função objetivo para a otimização
    def energy_function(vertices_flat):
        return energy_objective(
            vertices_flat,
            stereo_mesh,
            face_mask,
            uniform_mesh,
            rectangle_list,
            r_a,
            r_b,
            W_mesh,
            H_mesh
        )

    # Executa a otimização
    result = minimize(
        energy_function,
        vertices_initial_flat,
        method='L-BFGS-B',
        jac='3-point',
        options={'ftol': 1e-6, 'gtol': 1e-6, 'maxiter': max_iterations}
    )

    # Verifica o status da otimização
    if not result.success:
        print(f"Otimizador não convergiu: {result.message}")

    # Converte os vértices otimizados para a forma da malha original
    optimized_mesh = result.x.reshape(2, H_mesh, W_mesh)
    print("Energia mínima encontrada:", result.fun)
    print("Convergiu?", result.success)
    
    return optimized_mesh
if __name__ == "__main__":
    mesh_ds_ratio = 15
    Q = 4
    image_original, mask_original, _, rectangle_list = process_image("imagens/teste1.png")
    H_image, W_image, _ = image_original.shape

    # Aplica downsampling e padding NA IMAGEM para gerar as malhas
    H_mesh = H_image // mesh_ds_ratio + 2 * Q
    W_mesh = W_image // mesh_ds_ratio + 2 * Q
    mesh_perspective, mesh_estereo = get_uniform_stereo_mesh(image_original, np.pi*97/180, Q, mesh_ds_ratio)

    # Cria uma imagem e mascara com as dimensões corretas para a malha
    #image = cv2.resize(image_original, (W_mesh, H_mesh), interpolation=cv2.INTER_LANCZOS4)
    mask = cv2.resize(mask_original.astype(np.uint8), (W_mesh, H_mesh), interpolation=cv2.INTER_NEAREST).astype(bool)


    print(f"H_image: {H_image}, W_image: {W_image}")
    print(f"H_mesh: {H_mesh}, W_mesh: {W_mesh}")
    print(f"Shape mesh_estereo: {mesh_estereo.shape}")
    print(f"Shape mesh_perspective: {mesh_perspective.shape}")
    print(f"Shape image: {image_original.shape}")
    print(f"Shape mask: {mask.shape}")
    print(mask)

    vertices_initial = interpolate_mesh(mesh_perspective, mesh_estereo, mask)
    vertices_initial_flat = vertices_initial.reshape(-1)

    r_a = 50
    r_b = 10

    result = minimize(
    lambda x: energy_objective(x, mesh_estereo, mask, mesh_perspective, rectangle_list, r_a, r_b, W_mesh, H_mesh),
    vertices_initial_flat,
    method='L-BFGS-B',  # Use L-BFGS-B (ou BFGS se não tiver limites)
    jac='2-point', # Usando diferenciação numérica para o jacobiano
    options={'ftol': 1e-3, 'gtol': 1e-3, 'maxiter': 1000} # Ajuste as tolerâncias e o número máximo de iterações
    )


    vertices_optimized = result.x.reshape(2, H_mesh, W_mesh)

    print("Energia mínima encontrada:", result.fun)
    print("Convergiu?", result.success)

    