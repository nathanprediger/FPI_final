import torch
import torch.nn as nn
import numpy as np


def face_energy(vertices, mesh_stereo, face_weights, scale_transform, translation_vector, mi, box_mask, K):
    H, W = mesh_stereo.shape[1:]
    masks = box_mask.view(K, 1, H, W) * face_weights.view(1, 1, H, W)
    face_energy = masks * mi * (torch.norm(mesh_stereo - vertices, dim=0, keepdim=True) ** 2)
    sim_reg = (scale_transform[:, 0] - 1.0) ** 2
    face_energy = torch.sum(face_energy)
    face_energy += 1e4 * torch.sum(sim_reg)

    return face_energy


def bending_energy(vertices, mesh_uniform):
    mesh_uniform = torch.tensor(mesh_uniform, dtype=vertices.dtype, device=vertices.device)

    vertx = vertices[:, :, 1:] - vertices[:, :, :-1]  # Horizontal difference
    verty = vertices[:, 1:, :] - vertices[:, :-1, :]  # Vertical difference

    unix = mesh_uniform[:, :, 1:] - mesh_uniform[:, :, :-1]
    uniy = mesh_uniform[:, 1:, :] - mesh_uniform[:, :-1, :]

    # Normalize the uniform mesh differences
    e_ij_i = unix / (torch.norm(unix, dim=0, keepdim=True) + 1e-8)
    e_ij_j = uniy / (torch.norm(uniy, dim=0, keepdim=True) + 1e-8)

    # Compute bending energy
    energy = (torch.norm(vertx[..., 0] * e_ij_i[..., 1] - vertx[..., 1] * e_ij_i[..., 0]) ** 2).sum()
    energy += (torch.norm(verty[..., 0] * e_ij_j[..., 1] - verty[..., 1] * e_ij_j[..., 0]) ** 2).sum()

    return energy


def regularization_energy(vertices):
    diffs_x = vertices[:, :, 1:] - vertices[:, :, :-1]
    diffs_y = vertices[:, 1:, :] - vertices[:, :-1, :]
    return torch.sum(diffs_x.pow(2)) + torch.sum(diffs_y.pow(2))


def indicator(condition):
    return torch.where(condition, torch.tensor(1.0, device=condition.device), torch.tensor(0.0, device=condition.device))


def asymmetric_boundary_energy(vertices, H, W, Q, mesh_uniform):

    left_condition = vertices[0, Q:H - Q, Q] < 0
    left = indicator(left_condition) * (vertices[0, Q:H - Q, Q] - mesh_uniform[0, Q:H - Q, Q]).pow(2)

    right_condition = vertices[0, Q:H - Q, W - Q - 1] > W
    right = indicator(right_condition) * (vertices[0, Q:H - Q, W - Q - 1] - mesh_uniform[0, Q:H - Q, W - Q - 1]).pow(2)

    top_condition = vertices[1, Q, Q:W - Q] < 0
    top = indicator(top_condition) * (vertices[1, Q, Q:W - Q] - mesh_uniform[1, Q, Q:W - Q]).pow(2)

    bottom_condition = vertices[1, H - Q - 1, Q:W - Q] > H
    bottom = indicator(bottom_condition) * (vertices[1, H - Q - 1, Q:W - Q] - mesh_uniform[1, H - Q - 1, Q:W - Q]).pow(2)

    return torch.sum(left) + torch.sum(right) + torch.sum(top) + torch.sum(bottom)


def total_energy(vertices, mesh_stereo, face_weights, H, W, mesh_uniform, Q, mi, matrizes, box_mask, K):
    lambda_f, lambda_b, lambda_r, lambda_a = 4, 2, 0.5, 4
    Ef = face_energy(vertices, mesh_stereo, face_weights, matrizes, torch.zeros(2), mi, box_mask, K)
    Eb = bending_energy(vertices, mesh_uniform)
    Er = regularization_energy(vertices)
    Ea = asymmetric_boundary_energy(vertices, H, W, Q, mesh_uniform)
    return lambda_f * Ef + lambda_b * Eb + lambda_r * Er + lambda_a * Ea


def create_optimized_mesh(image, uniform_mesh, stereo_mesh, face_mask, Q, ra, rb, iterations, K, box_mask):
    mesh_uniform_torch = torch.tensor(uniform_mesh, dtype=torch.float32, requires_grad=False)
    mesh_stereo_torch = torch.tensor(stereo_mesh, dtype=torch.float32, requires_grad=False)

    mi = calculate_mi(uniform_mesh, ra, rb)
    similaridade = torch.tensor([1.0, 0.0], dtype=torch.float32).unsqueeze(0).repeat(K, 1)
    vertices = interpolate_mesh(uniform_mesh, stereo_mesh, face_mask).clone().detach().requires_grad_(True)
    face_weights = torch.from_numpy(face_mask)


    similaridade.requires_grad_(True)

    optimizer = torch.optim.SGD([vertices], lr=0.01)

    for step in range(iterations):
        optimizer.zero_grad()
        matrizes = gerar_matrizes_similaridade(similaridade, K)
        energy = total_energy(vertices, mesh_stereo_torch, face_weights, *uniform_mesh.shape[1:], mesh_uniform_torch, Q, mi, matrizes, box_mask, K)
        energy.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}, Energy: {energy.item()}")

    return vertices.detach().cpu().numpy()


def calculate_mi(uniform_mesh, ra, rb):
    dist_radial = np.linalg.norm(uniform_mesh, axis=0)
    return torch.from_numpy(1 / (1 + np.exp(-(dist_radial - ra) / rb )))


def gerar_matrizes_similaridade(similaridade, K):
    matrizes = torch.zeros([K, 2, 2], dtype=torch.float32)
    matrizes[:, 0, 0] = similaridade[:, 0]
    matrizes[:, 0, 1] = similaridade[:, 1]
    matrizes[:, 1, 0] = -similaridade[:, 1]
    matrizes[:, 1, 1] = similaridade[:, 0]
    return matrizes


def gaussian_kernel(distance_squared, h=2.37):
    return np.exp(-distance_squared / (2 * h**2))


def interpolate_mesh(mesh_perspective, mesh_stereographic, mask):
    H, W = mesh_perspective.shape[1:]
    mesh_interpolated = np.zeros_like(mesh_perspective)
    delta_v = np.zeros_like(mesh_perspective)
    delta_v[:, mask == 1] = mesh_stereographic[:, mask == 1] - mesh_perspective[:, mask == 1]

    for i in range(H):
        for j in range(W):
            if mask[i, j] == 0:
                mesh_interpolated[:, i, j] = mesh_perspective[:, i, j]
            else:
                pi = np.array([i, j])
                pj = np.array(np.meshgrid(np.arange(H), np.arange(W), indexing="ij")).reshape(2, -1).T

                distances_squared = np.sum((pj - pi) ** 2, axis=1).reshape(H, W)
                weights = gaussian_kernel(distances_squared) * mask

                numerator = np.sum(weights * delta_v, axis=(1, 2))
                denominator = np.sum(weights)

                if denominator > 0:
                    delta_vi_0 = numerator / denominator
                else:
                    delta_vi_0 = 0

                mesh_interpolated[:, i, j] = mesh_perspective[:, i, j] + delta_vi_0

    return torch.from_numpy(mesh_interpolated)
