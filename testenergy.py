import torch
import torch.nn.functional as F
from stereographic import get_uniform_stereo_mesh
from mask import process_image

# Implementação dos termos de energia ajustados
def face_energy(vertices, mesh_stereo, face_weights, scale_transform, translation_vector):
    # Transformação da malha estereográfica
    transformed_mesh = scale_transform @ mesh_stereo.reshape(2, -1)  # Forma (2, Hm * Wm)
    transformed_mesh += translation_vector.unsqueeze(-1)  # Broadcast para (2, Hm * Wm)

    # Reconstruir forma original para comparação
    transformed_mesh = transformed_mesh.reshape_as(vertices)  # Forma (2, Hm, Wm)

    # Calcular a diferença e a energia
    diff = vertices - transformed_mesh
    energy = (face_weights * diff.pow(2).sum(dim=0)).sum()
    return energy


def bending_energy(vertices):
    dx = vertices[:, :, 1:] - vertices[:, :, :-1]  # Diferença horizontal
    dy = vertices[:, 1:, :] - vertices[:, :-1, :]  # Diferença vertical
    energy = (dx.pow(2).sum() + dy.pow(2).sum())
    return energy

def regularization_energy(vertices):
    diffs_x = vertices[:, :, 1:] - vertices[:, :, :-1]
    diffs_y = vertices[:, 1:, :] - vertices[:, :-1, :]
    return (diffs_x.pow(2).sum() + diffs_y.pow(2).sum())

def asymmetric_boundary_energy(vertices, H, W):
    left = vertices[0, :, 0].pow(2).sum()
    right = (vertices[0, :, -1] - W).pow(2).sum()
    top = vertices[1, 0, :].pow(2).sum()
    bottom = (vertices[1, -1, :] - H).pow(2).sum()
    return left + right + top + bottom



# Função de energia total
def total_energy(vertices, mesh_stereo, face_weights, H, W):
    lambda_f, lambda_b, lambda_r, lambda_a = 4, 2, 0.5, 4
    Ef = face_energy(vertices, mesh_stereo, face_weights, torch.eye(2), torch.zeros(2))
    Eb = bending_energy(vertices)
    Er = regularization_energy(vertices)
    Ea = asymmetric_boundary_energy(vertices, H, W)
    return lambda_f * Ef + lambda_b * Eb + lambda_r * Er + lambda_a * Ea


def create_optimized_mesh(image, uniform_mesh, stereo_mesh, face_mask, rect_list, ra, rb, iterations, mesh_ds_ratio):
    # Converter para PyTorch
    mesh_uniform_torch = torch.tensor(uniform_mesh, dtype=torch.float32, requires_grad=False)
    mesh_stereo_torch = torch.tensor(stereo_mesh, dtype=torch.float32, requires_grad=False)

    # Inicializar malha ajustável
    vertices = mesh_uniform_torch.clone().detach().requires_grad_(True)
        
    # Otimização
    face_weights = torch.ones_like(vertices[0])  # Pesos uniformes para simplificar

    optimizer = torch.optim.Adam([vertices], lr=0.01)

    for step in range(iterations):
        optimizer.zero_grad()
        energy = total_energy(vertices, mesh_stereo_torch, face_weights, *uniform_mesh.shape[1:])
        energy.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}, Energy: {energy.item()}")

    # Retornar a malha otimizada
    return vertices.detach().cpu().numpy()
