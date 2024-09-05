# SwiftCraft3D: Efficient Text-to-3D Generation via Semantic-enhanced Sparse-view Prompting with Hybrid Reconstruction
![hrc7u-k1v2r](https://github.com/user-attachments/assets/87541ed7-8aba-4d62-bcec-5cbd8ba983a8)

# Usage

## 1. Requirements
    git clone https://github.com/OpenMICG/SwiftCraft3D.git

    cd SwiftCraft3D
    
    conda create -n SwiftCraft3D python=3.8

    conda activate SwiftCraft3D

    pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

    pip install -r requirements.txt

## 2. Train
Our training data is sourced from the [Objaverse dataset](https://objaverse.allenai.org/objaverse-1.0/), and the subtitles are derived from [Cap3D](https://github.com/crockwell/Cap3D?tab=readme-ov-file). Execute the following command to initiate the training:

    accelerate launch train.py  \
    --mixed_precision="fp16" \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
    --train_data_dir="/path/to/dataset" --caption_column="text"  \
    --resolution=1024 --random_flip  \
    --resume_from_checkpoint=latest \
    --train_batch_size=2 --num_train_epochs=10 --checkpointing_steps=1000  \
    --learning_rate=1e-06 --lr_scheduler="constant" --lr_warmup_steps=0  \
    --output_dir="/path/to/output"

## 3. Inference
The following command generates an OBJ file with vertex colors：

    python inference.py "input text" name

By executing the following command, a higher quality 3D mesh is generated and the OBJ, MTL, and PNG files are exported:

    python inference.py "input text" name --export_omp

## 4. Evaluation
We evaluated our model on the [T3Bench](https://github.com/THU-LYJ-Lab/T3Bench)

# Anckowledgement
This repo is based on [GPT-4](https://openai.com/index/gpt-4/), [Stable Diffusion XL](https://github.com/Stability-AI/StableDiffusion), [zero123plus](https://github.com/SUDO-AI-3D/zero123plus), [LRM](https://github.com/3DTopia/OpenLRM?tab=readme-ov-file), [InstantMesh](https://github.com/TencentARC/InstantMesh), and [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes). We would like to express our gratitude for their outstanding work.
