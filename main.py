import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from sklearn import svm
from copy import deepcopy
from model import *
from util import *
from e4e_projection import projection as e4e

# ============================
# Global Configuration
# ============================
device = 'cuda'
transform = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# ============================
# Path Configuration
# ============================
style_name = 'art'
test_image_path = 'test_input/My.jpeg'
test_image_name = os.path.splitext(os.path.basename(test_image_path))[0]

data_paths = {
    "stylegan_ckpt": 'models/stylegan2-ffhq-config-f.pt',
    "test_image": test_image_path,
    "gaze_latents": 'eye_dataset/eye_gaze_latents.npy',
    "gaze_labels": 'eye_dataset/labels.npy',
    "normal_vector": 'eye_dataset/normal_vector.pt',
    "gaze_latent_out": f'outputs/{test_image_name}_gaze.pt',
    "styled_output": f'outputs/{test_image_name}_{style_name}.jpg',
    "inverse_output": f'outputs/{test_image_name}_inverse.jpg',
    "comparison_output": f'outputs/{test_image_name}_comparison.jpg',
    "random_samples": 'outputs/random_samples.jpg',
    "layer_interp_dir": f'outputs/{test_image_name}_layer_interp',
    "multi_interp_path": f'outputs/{test_image_name}_multi_interp.jpg',
    "final_interp_path": f'outputs/{test_image_name}_final_interp.jpg',
    "style_model": f'models/{style_name}.pt',
}

# ============================
# Helper Functions
# ============================
def interpolate_with_target_layer(alphas, latent_base, target_layers, normal_vector, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for layer_idx in target_layers:
        print(f"Interpolating layer {layer_idx}")
        images = []
        for alpha in alphas:
            delta = torch.zeros_like(latent_base.squeeze(0))
            delta[layer_idx] = alpha * normal_vector[layer_idx]
            latent_interp = latent_base.squeeze(0) + delta.to(latent_base)
            latent_tensor = latent_interp.unsqueeze(0).float().to(device)
            img = original_generator(latent_tensor, input_is_latent=True)[0]
            images.append(img)
        grid = make_grid(torch.stack(images), nrow=len(alphas)//5, normalize=True, value_range=(-1, 1))
        save_image(grid, f'{out_dir}/layer_{layer_idx:02d}.jpg')


def interpolate_with_multiple_layers(alphas, latent_base, target_layers, normal_vector, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    images = []
    for alpha in alphas:
        delta = torch.zeros_like(latent_base.squeeze(0))
        for layer_idx in target_layers:
            delta[layer_idx] = alpha * normal_vector[layer_idx]
        latent_interp = latent_base.squeeze(0) + delta.to(latent_base)
        latent_tensor = latent_interp.unsqueeze(0).float().to(device)
        img = original_generator(latent_tensor, input_is_latent=True)[0]
        images.append(img)
    grid = make_grid(torch.stack(images), nrow=len(alphas)//5, normalize=True, value_range=(-1, 1))
    save_image(grid, out_path)


def calculate_distance(filepath, normal_vector, bias, clf):
    name = strip_path_extension(filepath) + '.pt'
    aligned = align_face(filepath)
    latent = e4e(aligned, name, device)
    w_avg = latent[6]
    normal_tensor = torch.tensor(normal_vector, dtype=torch.float32, device=device)
    distance = (torch.dot(normal_tensor, w_avg) + torch.tensor(bias).to(device)) / torch.norm(normal_tensor)
    return distance, latent

# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    # Step 1: Initialize Generator
    print("[Step 1] Loading original generator")
    original_generator = Generator(1024, 512, 8, 2).to(device)
    ckpt = torch.load(data_paths['stylegan_ckpt'], map_location='cpu')
    original_generator.load_state_dict(ckpt['g_ema'], strict=False)
    mean_latent = original_generator.mean_latent(10000)
    generator = deepcopy(original_generator)

    # Step 2: Load and Invert Test Image
    print("[Step 2] Invert test image")
    aligned = align_face(data_paths['test_image'])
    latent = e4e(aligned, data_paths['gaze_latent_out'], device).unsqueeze(0)
    recon = original_generator(latent, input_is_latent=True)[0]
    comparison = torch.cat([transform(aligned).to(device), recon], dim=2)
    save_image(comparison, data_paths['comparison_output'])

    # Step 3: Eye Gaze Latent Vector Analysis
    print("[Step 3] Eye Gaze Latent Vector Interpolation")
    save_image(recon, data_paths['inverse_output'])

    latents = np.load(data_paths['gaze_latents'])
    labels = np.load(data_paths['gaze_labels'])

    clf = svm.SVC(kernel='linear')
    clf.fit(latents, labels)
    normal_vector = clf.coef_[0]
    bias = clf.intercept_[0]
    torch.save(normal_vector, data_paths['normal_vector'])

    alphas = np.linspace(-0.2, 1.3, num=30)
    interpolate_with_target_layer(alphas, latent, list(range(18)), normal_vector, data_paths['layer_interp_dir'])
    interpolate_with_multiple_layers(alphas, latent, [5, 6], normal_vector, data_paths['multi_interp_path'])

    # Step 4: Style Change + Distance Comparison
    print("[Step 4] Style transformation and distance analysis")
    distance_original, input_w = calculate_distance(data_paths['test_image'], normal_vector, bias, clf)
    style_ckpt = torch.load(data_paths['style_model'], map_location='cpu')
    generator.load_state_dict(style_ckpt['g'], strict=False)
    styled = generator(input_w.unsqueeze(0), input_is_latent=True)[0]
    save_image(styled, data_paths['styled_output'], normalize=True, value_range=(-1, 1))

    distance_styled, output_w = calculate_distance(data_paths['styled_output'], normal_vector, bias, clf)
    print(f"Original Distance: {distance_original:.4f}, Styled Distance: {distance_styled:.4f}")

    # Step 5: Final Interpolation
    print("[Step 5] Final interpolation across SVM boundary")
    final_images = []
    for alpha in alphas:
        delta = torch.zeros_like(input_w)
        for idx in [5, 6]:
            delta[idx] = alpha * normal_vector[idx]
        w_mod = input_w + delta.to(device)
        img = generator(w_mod.unsqueeze(0), input_is_latent=True)[0]
        final_images.append(img)
    final_grid = make_grid(torch.stack(final_images), nrow=len(final_images)//5, normalize=True, value_range=(-1, 1))
    save_image(final_grid, data_paths['final_interp_path'])
    
    # Save as GIF
    gif_path = data_paths['final_interp_path'].replace('.jpg', '.gif')
    pil_images = [
        transforms.ToPILImage()(img.cpu().clamp(-1, 1) * 0.5 + 0.5)
        for img in final_images
    ]
    pil_images[0].save(
        gif_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=200,  # ms per frame
        loop=0
    )
    print(f"[INFO] Saved GIF to {gif_path}")
