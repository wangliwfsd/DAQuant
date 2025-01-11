# DAQuant

Despite the increasing popularity of Vision Transformers (ViTs) on vision tasks, their deployment on mobile devices presents two main challenges: performance degradation due to the necessary model compression amidst computational constraints, and accuracy drop stemming from domain shift effects. Although existing post-training quantization (PTQ) methods can reduce computational load for ViTs, they often fail under extreme low-bit conditions and domain shift scenarios. To address the two challenges, this paper introduces a novel Domain Aware Post-training Quantization (**DAQuant**) approach that simultaneously tackles extreme model compression and domain adaptation for ViTs in deployment. **DAQuant** employs a distribution-aware smoothing technique to mitigate outlier effects in ViT activations and employs learnable activation clipping (LAC) to minimize quantization errors. Additionally, we propose an effective domain alignment strategy to improve the model’s generalizability, which preserves model’s optimization on source domain while enhancing generalization ability on the target domain. DAQuant demonstrates superior performance in both quantization error and generalization capacity, outperforming existing quantization methods significantly in real-device deployment scenarios.

## Usage
**We provide full script to run DAQuant. We use DeiT-S as an example here**. You can download the model weights of [deit-small-patch16-224](https://huggingface.co/facebook/deit-small-patch16-224) from [Huggingface](https://huggingface.co/).
1. Install Package
```
conda create -n daquant python=3.11.0 -y
conda activate daquant
pip install --upgrade pip  
pip install -r requirements.txt
```

2. Obtain the channel-wise scales and shifts required for initialization:

```
python generate_act_scale_shift.py --model /PATH/TO/DeiT/deit-small-patch16-224
```

3. model quantization
```
# W4A4 
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/DeiT/deit-small-patch16-224  \
--epochs 20 --output_dir ./log/deit-small-patch16-224-w4a4 \
--wbits 4 --abits 4 --dga --lwc --lac --wrc

# W6A6
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/DeiT/deit-small-patch16-224  \
--epochs 20 --output_dir ./log/deit-small-patch16-224-w6a6 \
--wbits 6 --abits 6 --dga --lwc --lac --wrc

# W4A16
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/DeiT/deit-small-patch16-224  \
--epochs 20 --output_dir ./log/deit-small-patch16-224-w4a16 \
--wbits 4 --abits 16 --dga --lwc --lac --wrc

# W3A16
CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/DeiT/deit-small-patch16-224  \
--epochs 20 --output_dir ./log/deit-small-patch16-224-w3a16 \
--wbits 3 --abits 16 --dga --lwc --lac --wrc
```

4. domain adaptation

Below is the running script for Domain Adaptation, and we will release the pre-trained model weights shortly.
```
# W4A4
CUDA_VISIBLE_DEVICES=7 python main.py \
--model /PATH/TO/DeiT/deit-small-patch16-224  \
--source_model /PATH/TO/Pre-train-in-office/DeiT/DeiT-S \
--epochs 10 --output_dir ./log/deit-small-patch16-224-w4a4-da  \
--wbits 4 --abits 4 --dga --lwc --lac --wrc --tl \
--calib_dataset amazon --target_dataset webcam \
--tl_loss --tl_weight 1.5

```

5. real quant

We utilize the kernel from [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) to enable real quantization. If you aim to accelerate and compress your model using real quantization, we can follow these steps.
```
pip install auto-gptq==0.6.0

CUDA_VISIBLE_DEVICES=0 python main.py \
--model /PATH/TO/DeiT/deit-small-patch16-224  \
--epochs 20 --output_dir ./log/deit-small-patch16-224-w4a4 \
--wbits 4 --abits 16 --lwc --lac --wrc \
--real_quant --save_dir ./real_quant/deit-small-patch16-224-w4a16
```

## Related Project
[SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models](https://github.com/mit-han-lab/smoothquant)

[OmniQuant: Omnidirectionally Calibrated Quantization for Large Language Models](https://github.com/OpenGVLab/OmniQuant.git)
