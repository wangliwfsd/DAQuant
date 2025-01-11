from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
from transformers import ViTForImageClassification
import torch
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from tqdm import tqdm
import gc   
import time
from torch import nn
from data_loaders import *
import os
import utils
from torch.cuda.amp import autocast
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def set_op_by_name(layer, name, new_module):
    levels = name.split('.')
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels)-1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)

# evaluate
def evaluate_model(model, dataloader, dev, logger):
    total_samples = len(dataloader.dataset)
    correct_predictions = 0
    model.eval()
    
    i=0
    with autocast(dtype=torch.float16):
        with torch.no_grad():
            for data in dataloader:
                i = i+1
                inputs = {"pixel_values": data[0].cuda()}
                labels = data[1].to(dev)

                outputs = model(**inputs)
                if i%5 == 0:
                    logger.info(f"gpu memory usage: {torch.cuda.max_memory_allocated(dev) / 1024**2} MB")
                logits = outputs.logits

                predictions = torch.argmax(logits, dim=1)
                correct_predictions += torch.sum(predictions == labels)

    accuracy = correct_predictions.item() / total_samples
    return accuracy
    

def main():
    device = 'cuda'
    logger = utils.create_logger("./rela_quant_log/")
    logger.info(f"empty gpu memory usage: {torch.cuda.max_memory_allocated(device) / 1024**2} MB")
    model_path = '/PATH/TO/real_quant/deit-small-patch16-224-w4a16'
    model_name = 'deit-small-patch16-224'
    wbits = 4
    group_size = -1
    with init_empty_weights():
        model = ViTForImageClassification.from_pretrained('/PATH/TO/DeiT/deit-small-patch16-224')

    time.sleep(3)
    layers = model.vit.encoder.layer

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        for name, module in named_linears.items():
            q_linear = qlinear_cuda.QuantLinear(wbits, group_size, module.in_features,module.out_features,not module.bias is None,kernel_switch_threshold=128)
            q_linear.to(next(layer.parameters()).device)
            set_op_by_name(layer, name, q_linear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    logger.info("Loading pre-computed quantized weights...")

    load_checkpoint_in_model(model,checkpoint=model_path,device_map=device_map,offload_state_dict=True)
    logger.info("Loading pre-computed quantized weights Successfully")
    
    model = model.to(device)
    logger.info(f"gpu memory usage: {torch.cuda.max_memory_allocated(device) / 1024**2} MB")
    model.eval()

    # load ImageNet dataset
    imagenet_dataloader = eval("{}DataLoader".format('ImageNet'))(
                            model_name,
                            data_dir=os.path.join('/PATH/TO/IMAGENET', 'ImageNet'),
                            image_size=224,
                            batch_size=2048,
                            num_workers=64,
                            split='val')
    begin = time.time()
    accuracy = evaluate_model(model, imagenet_dataloader, 'cuda', logger)

    logger.info(f"time: {time.time()-begin}")
    logger.info(f"Model Accuracy on ImageNet: {accuracy}")
    

if __name__ == '__main__':
    main()