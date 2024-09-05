import os
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from einops import rearrange, repeat
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    FOV_to_intrinsics, 
    get_zero123plus_input_cameras,
    get_circular_camera_poses,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl
from src.utils.infer_util import remove_background, resize_foreground, save_video

from diffusers import DiffusionPipeline
import torch
import rembg
import requests
import json


# Configuration and Load parameters

parser = argparse.ArgumentParser()
parser.add_argument('text', type=str, help='Path to input text.')
parser.add_argument('name', type=str, help='Output name.')
parser.add_argument('--output_path', type=str, default='outputs/', help='Output path.')
parser.add_argument('--MV_diffusion_steps', type=int, default=75, help='Multi_View Denoising steps.')
parser.add_argument('--view', type=int, default=6, choices=[4, 6], help='Number of Multi-view.')
parser.add_argument('--no_rembg', action='store_true', help='Do not remove input background.')
parser.add_argument('--export_omp', action='store_true', help='Export obj, mtl, png.')
parser.add_argument('--seed', type=int, default=42, help='Seed for random number generator.')
parser.add_argument('--scale', type=float, default=1.0, help='Scale for generated object.')
args = parser.parse_args()
seed_everything(args.seed)

config = OmegaConf.load("rec_config/reconstruction.yaml")
config_name = os.path.basename("rec_config/reconstruction.yaml").replace('.yaml', '')
model_config = config.model_config
infer_config = config.infer_config



device = torch.device('cuda')


pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="zero123plus",
    torch_dtype=torch.float16,
)
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)


if os.path.exists(infer_config.unet_path):
    unet_ckpt_path = infer_config.unet_path
else:
    unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
state_dict = torch.load(unet_ckpt_path, map_location='cpu')
pipeline.unet.load_state_dict(state_dict, strict=True)

pipeline = pipeline.to(device)

model = instantiate_from_config(model_config)
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename=f"{config_name.replace('-', '_')}.ckpt", repo_type="model")
state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
model.load_state_dict(state_dict, strict=False)

model = model.to(device)
model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()


image_path = os.path.join(args.output_path, "SwiftCraft3D", 'image')
mv_image_path = os.path.join(args.output_path, "SwiftCraft3D", 'mv_image')
mesh_path = os.path.join(args.output_path, "SwiftCraft3D", 'meshes')
os.makedirs(image_path, exist_ok=True)
os.makedirs(mv_image_path, exist_ok=True)
os.makedirs(mesh_path, exist_ok=True)

model_path = "/data/dongzeyi/1000(2)lora_1024/checkpoint-1250" # 根据自己设置的训练策略找到保存权重的checkpoint文件夹
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)
pipe.load_lora_weights(model_path)
pipe.to("cuda")
test = pipe("Loading", num_inference_steps=50, guidance_scale=7.5).images[0]

# Semantic-Enhanced Multi-View Prompts Generation

print("Text-to-Image Generation")
# 定义API端点
API_ENDPOINT = 'https://api.coze.com/open_api/v2/chat'


Instruction=("You are a semantic enhancement assistant. "
             "Your task is to briefly enhance the semantics of the text we input, and you only need to output the enhanced text content. You have the following enhancement schemes: "
             "\n1. If the input text can clearly describe a 3D object or 3D scene, then output the original text and add “highly detailed 4K full-body, perfectly isolated against a gray background” at the end. "
             "\n2. If the input text cannot sufficiently describe a 3D object or 3D scene, then reasonably enhance the semantics without changing the original meaning and add “highly detailed 4K full-body, perfectly isolated against a gray background” at the end.")

# 您的API Token
headers = {
    'Authorization': 'Bearer pat_aXlVsNX26LkkylmfUXM6Zapxy6qDNwuz4WnrZ7tiMunSWWae6ln4eRI6s4lg6kH0',  # 替换pat_XXXXXXXXXXXXXXXXXX为你的“Personal Access Tokens”
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': 'api.coze.com',
    'Connection': 'keep-alive',
}

# 要发送给api的数据
data = {
  "conversation_id": "1",  # 这里填入您的对话ID
  "bot_id": "7397410756176101377",  # 替换735884567716*******为你的Bot ID
  "user" : "1",  # 这里填入您的用户ID
  "query": Instruction+"Input Text: "+args.text,
  "stream": False
}
print(Instruction+"\nInput Text: "+args.text)
# 发送post请求并保存响应为响应对象
response = requests.post(url = API_ENDPOINT, headers = headers, data = json.dumps(data))
data = response.json()
# 提取以json格式的信息
answer = data['messages'][0]['content']

name=args.name
prompt=answer
input_text=[]
input_text.append(prompt)
input_files=[]
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
img_path="/path/to/outputs/SwiftCraft3D/image/"+name+".png"
image.save(img_path)  # 保存图像名
input_files.append(img_path)
del pipe

# i=1
# input_text=[]
# input_files=[]
# with open('/path/to/1.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         name=str(i)
#         line = line.strip()
#         prompt=line
#         image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
#         image_copy=image.copy()
#         img_path="/path/to/outputs/SwiftCraft3D/image/"+name+".png"
#         image.save(img_path)  # 保存图像名
#         input_files.append(img_path)
#         input_text.append(line)
#         i+=1
# del pipe


print("Image-to-Multi_View Generation")

rembg_session = None if args.no_rembg else rembg.new_session()

outputs = []
for idx, image_file in enumerate(input_files):
    name = os.path.basename(image_file).split('.')[0]
    print(f'[{idx+1}/{len(input_files)}] Imagining {name} ...')

    input_image = Image.open(image_file)
    if not args.no_rembg:
         input_image = rembg.remove(input_image)
    # sampling
    output_image = pipeline(
        input_image,
        num_inference_steps=args.MV_diffusion_steps,
    ).images[0]
    output_image.save(os.path.join(mv_image_path, f'{name}.png'))


    images = np.asarray(output_image, dtype=np.float32) / 255.0
    images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()     # (3, 960, 640)
    images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)        # (6, 3, 320, 320)
    input_image=np.asarray(output_image, dtype=np.float32) / 255.0
    input_image = torch.from_numpy(input_image).permute(2, 0, 1).contiguous().float()
    input_image = input_image.unsqueeze(0)
    outputs.append({'name': name, 'images': images,"name2":name,"main_image":input_image})


del pipeline

# Text-Infused Sparse-View 3D Reconstruction

print("3D Reconstruction")

input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0*args.scale).to(device)
chunk_size = 20

for idx, sample in enumerate(outputs):
    name = sample['name']
    print(f'[{idx+1}/{len(outputs)}] Creating {name} ...')

    images = sample['images'].unsqueeze(0).to(device)
    Main_image= sample['main_image'].unsqueeze(0).to(device)

    images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
    Main_image = v2.functional.resize(Main_image, 320, interpolation=3, antialias=True).clamp(0, 1)

    if args.view == 4:
        indices = torch.tensor([0, 2, 4, 5]).long().to(device)
        images = images[:, indices]
        input_cameras = input_cameras[:, indices]

    with torch.no_grad():
        # get triplane
        text=input_text[idx]
        planes = model.forward_planes(images, input_cameras,text)

        # get mesh
        mesh_path_idx = os.path.join(mesh_path, f'{name}.obj')

        mesh_out = model.extract_mesh(
            planes,
            use_texture_map=args.export_omp,
            **infer_config,
        )
        if args.export_omp:
            vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
            save_obj_with_mtl(
                vertices.data.cpu().numpy(),
                uvs.data.cpu().numpy(),
                faces.data.cpu().numpy(),
                mesh_tex_idx.data.cpu().numpy(),
                tex_map.permute(1, 2, 0).data.cpu().numpy(),
                mesh_path_idx,
            )
        else:
            vertices, faces, vertex_colors = mesh_out
            save_obj(vertices, faces, vertex_colors, mesh_path_idx)
        print(f"3D Mesh saved to {mesh_path_idx}")