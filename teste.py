import torch
import torch.nn as nn
import numpy as np

def calculate_mi(uniform_mesh, ra, rb):
            dist_radial = np.linalg.norm(uniform_mesh, axis=0)
            return torch.from_numpy(1 / (1 + np.exp(-(dist_radial - ra) / rb )))

def gaussian_kernel(distance_squared, h=2.37):
            return np.exp(-distance_squared / (2 * h**2))


def interpolate_mesh(mesh_perspective, mesh_stereographic, mask):
    # mesh_perspective = self.mesh_uniform_torch.detach().cpu().numpy()
    # mesh_stereographic = self.mesh_stereo_torch.detach().cpu().numpy()
    # mask = self.face_weights.detach().cpu().numpy
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
class Minimizer(nn.Module):
    def __init__(self, image, uniform_mesh, stereo_mesh, face_mask, Q, ra, rb, K, box_mask):
        super(Minimizer, self).__init__()
        self.image = image
        self.mesh_uniform_torch = torch.tensor(uniform_mesh, dtype=torch.float32)
        self.mesh_stereo_torch = torch.tensor(stereo_mesh, dtype=torch.float32)
        self.vertices = torch.tensor(uniform_mesh, dtype=torch.float32)
        self.face_weights = torch.tensor(face_mask)
        self.Q=Q
        self.K=K
        self.box_mask=torch.tensor(box_mask, dtype=torch.float32)
        self.mi = calculate_mi(uniform_mesh, ra, rb)
        _,self.H,self.W=self.mesh_stereo_torch.shape
        
        self.similaridade = torch.tensor([1.0, 0.0], dtype=torch.float32).unsqueeze(0).repeat(K, 1)
        self.translation = torch.zeros([self.K, 2], dtype=torch.float32)
        
        self.mesh_uniform_torch = nn.Parameter(self.mesh_uniform_torch, requires_grad=False)
        self.mesh_stereo_torch = nn.Parameter(self.mesh_stereo_torch, requires_grad=False)
        self.vertices = nn.Parameter(self.vertices, requires_grad=True)
        self.mi = nn.Parameter(self.mi, requires_grad=False)
        self.face_weights = nn.Parameter(self.face_weights, requires_grad=False)
        self.similaridade = nn.Parameter(self.similaridade, requires_grad=True)
        self.translation = nn.Parameter(self.translation, requires_grad=True)
        self.box_mask = nn.Parameter(self.box_mask, requires_grad=False)
        
    def forward(self):
        
        def face_energy():
            def gerar_matrizes_similaridade():
                matrizes = torch.zeros([self.K, 2, 2], dtype=self.similaridade.dtype, device=self.similaridade.device)
                matrizes[:, 0, 0] = self.similaridade[:, 0]
                matrizes[:, 0, 1] = self.similaridade[:, 1]
                matrizes[:, 1, 0] = -self.similaridade[:, 1]
                matrizes[:, 1, 1] = self.similaridade[:, 0]
                return matrizes
            similarity_matrices = gerar_matrizes_similaridade()
            transformed_target = torch.matmul(similarity_matrices, self.mesh_stereo_torch.view(2, self.H * self.W))
            transformed_target = transformed_target.view(self.K, 2, self.H, self.W)
            transformed_target = transformed_target + self.translation.view(self.K, 2, 1, 1)
            target_mesh = transformed_target
            
            H, W = self.mesh_stereo_torch.shape[1:]
            masks = self.box_mask.view(self.K, 1, H, W) * self.face_weights.view(1, 1, H, W)
            face_energy = masks * self.mi * ((self.vertices-target_mesh) ** 2)
            sim_reg = ((self.similaridade[:, 0] - 1.0) ** 2)*2000
            face_energy = ((face_energy.sum())/self.K)
            face_energy += 1e4 * sim_reg.mean()

            return face_energy


        def bending_energy():

            # coordinate_padding_u = torch.zeros_like(self.mi[1:, :]).unsqueeze(0)
            # diff_u = self.vertices[:, 1:, :] - self.vertices[:, :-1, :]  # Horizontal difference
            # diff_u = torch.cat([coordinate_padding_u, diff_u], dim=0)
            
            # diff_uniform_u = self.mesh_uniform_torch[:, 1:, :] - self.mesh_uniform_torch[:, :-1, :]
            # unit_diff_u = diff_uniform_u / (torch.norm(diff_uniform_u, dim=0)).unsqueeze(0)
            # unit_diff_u = torch.cat([coordinate_padding_u, unit_diff_u], dim=0)
            # line_bending_u_loss = torch.square(torch.norm(torch.cross(diff_u, unit_diff_u, dim=0), dim=0))
            
            # coordinate_padding_v = torch.zeros_like(self.mi[:, 1:]).unsqueeze(0)
            # diff_v = self.vertices[:, :, 1:] - self.vertices[:, :, :-1]  # Vertical difference
            # diff_v = torch.cat([coordinate_padding_v, diff_v], dim=0)
            # diff_uniform_v = self.mesh_uniform_torch[:, :, 1:] - self.mesh_uniform_torch[:, :, :-1]
            # unit_diff_v = diff_uniform_v / (torch.norm(diff_uniform_v, dim=0)).unsqueeze(0)
            # unit_diff_v = torch.cat([coordinate_padding_v, unit_diff_v], dim=0)

            # line_bending_v_loss = torch.square(torch.norm(torch.cross(diff_v, unit_diff_v, dim=0), dim=0))
            
            # return (line_bending_u_loss.sum() + line_bending_v_loss.sum()) / 2
            
            vertx = self.vertices[:, :, 1:] - self.vertices[:, :, :-1]  # Horizontal difference
            verty = self.vertices[:, 1:, :] - self.vertices[:, :-1, :]  # Vertical difference
            unix = self.mesh_uniform_torch[:, :, 1:] - self.mesh_uniform_torch[:, :, :-1]
            uniy = self.mesh_uniform_torch[:, 1:, :] - self.mesh_uniform_torch[:, :-1, :]

            # # Normalize the uniform mesh differences
            e_ij_i = unix / (torch.norm(unix, dim=0, keepdim=True) + 1e-8)
            e_ij_j = uniy / (torch.norm(uniy, dim=0, keepdim=True) + 1e-8)

            # # Compute bending energy
            energy = (torch.norm(vertx[..., 0] * e_ij_i[..., 1] - vertx[..., 1] * e_ij_i[..., 0]) ** 2).sum()
            energy += (torch.norm(verty[..., 0] * e_ij_j[..., 1] - verty[..., 1] * e_ij_j[..., 0]) ** 2).sum()

            return energy


        def regularization_energy():
            diffs_x = self.vertices[:, :, 1:] - self.vertices[:, :, :-1]
            diffs_y = self.vertices[:, 1:, :] - self.vertices[:, :-1, :]
            return torch.sum(torch.norm(diffs_x).pow(2)) + torch.sum(torch.norm(diffs_y).pow(2))
        
            # coordinate_padding_u = torch.zeros_like(self.mi[1:, :]).unsqueeze(0)
            # diff_u = self.vertices[:, 1:, :] - self.vertices[:, :-1, :]  # Horizontal difference
            # diff_u = torch.cat([diff_u, coordinate_padding_u], dim=0)
            
            # coordinate_padding_v = torch.zeros_like(self.mi[:, 1:]).unsqueeze(0)
            # diff_v = self.vertices[:, :, 1:] - self.vertices[:, :, :-1]  # Vertical difference
            # diff_v = torch.cat([diff_v, coordinate_padding_v], dim=0)
            
            
            # return torch.mean((torch.square(torch.norm(diff_u)) + torch.square(torch.norm(diff_v))) / 2)

        def indicator(condition):
            return torch.where(condition, torch.tensor(1.0, device=condition.device), torch.tensor(0.0, device=condition.device))


        def asymmetric_boundary_energy():
    
            left = (self.vertices[0, self.Q:self.H-self.Q, self.Q] -
                self.mesh_uniform_torch[0, self.Q:self.H - self.Q, self.Q]) ** 2
            right = (self.vertices[0, self.Q:self.H - self.Q, self.W - self.Q - 1] -
                    self.mesh_uniform_torch[0, self.Q:self.H - self.Q, self.W - self.Q - 1]) ** 2
            top = (self.vertices[1, self.Q, self.Q:self.W-self.Q] -
                self.mesh_uniform_torch[1, self.Q, self.Q:self.W - self.Q]) ** 2
            bottom = (self.vertices[1, self.H - self.Q - 1, self.Q:self.W-self.Q] -
                    self.mesh_uniform_torch[1, self.H - self.Q - 1, self.Q:self.W - self.Q]) ** 2
            return left.sum() + right.sum() + top.sum() + bottom.sum()
            
            # left_condition = self.vertices[0, self.Q:H - self.Q, self.Q] > 0
            # left = indicator(left_condition) * (self.vertices[0, self.Q:H - self.Q, self.Q] - self.mesh_uniform_torch[0, self.Q:H - self.Q, self.Q]).pow(2)

            # right_condition = self.vertices[0, self.Q:H - self.Q, W - self.Q - 1] < W
            # right = indicator(right_condition) * (self.vertices[0, self.Q:H - self.Q, W - self.Q - 1] - self.mesh_uniform_torch[0, self.Q:H - self.Q, W - self.Q - 1]).pow(2)

            # top_condition = self.vertices[1, self.Q, self.Q:W - self.Q] > 0
            # top = indicator(top_condition) * (self.vertices[1, self.Q, self.Q:W - self.Q] - self.mesh_uniform_torch[1, self.Q, self.Q:W - self.Q]).pow(2)

            # bottom_condition = self.vertices[1, H - self.Q - 1, self.Q:W - self.Q] < H
            # bottom = indicator(bottom_condition) * (self.vertices[1, H - self.Q - 1, self.Q:W - self.Q] - self.mesh_uniform_torch[1, H - self.Q - 1, self.Q:W - self.Q]).pow(2)

            # return torch.sum(left) + torch.sum(right) + torch.sum(top) + torch.sum(bottom)

        


        lambda_f, lambda_b, lambda_r, lambda_a = 4, 2, 0.5, 4
        Ef = face_energy()
        Eb = bending_energy()
        Er = regularization_energy()
        Ea = asymmetric_boundary_energy()
        return lambda_f * Ef + lambda_b * Eb + lambda_r * Er + lambda_a * Ea


        


        