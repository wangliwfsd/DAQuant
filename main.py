import os
import sys
import random
import numpy as np
from models.ViTClass import ViTClass
from models.int_vit_layer import QuantViTLayer
import torch
import time
from datautils import get_loaders
from quantize.DAQuant import DAQuant
from tqdm import tqdm
import utils
from pathlib import Path
from torch.cuda.amp import autocast
from quantize.int_linear import QuantLinear
from torchvision import transforms
from data_loaders import *
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transfer_data_loader import Amazon, DSLR, Webcam
from office_home_dataset import get_art, get_clipart, get_product, get_real_word
torch.backends.cudnn.benchmark = True

net_choices = [
    "vit-tiny-patch16-224",
    "vit-small-patch16-224",
    "vit-base-patch16-224",
    "vit-base-patch16-384",
    "vit-large-patch16-224",
    "vit-large-patch16-384",
    "deit-tiny-patch16-224",
    "deit-small-patch16-224",
    "deit-base-patch16-224"
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--source_model", type=str, help="model weight for transfer learning")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="cache dir of dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="../log/", type=str, help="direction of logging file")
    parser.add_argument("--save_dir", default=None, type=str, help="direction for saving fake quantization model")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--real_quant", default=False, action="store_true",)
    parser.add_argument("--calib_dataset",type=str,default="ImageNet",
        choices=["ImageNet", "kinetics", "amazon", "dslr", "webcam", "art", "clipart", "product", "real_word"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--tl",default=False, action="store_true",help="transfer learning")
    parser.add_argument("--tl_loss",default=False, action="store_true",help="transfer learning loss")
    parser.add_argument("--tl_weight", type= float, default=1, help="the importance for transfering loss")
    parser.add_argument("--target_dataset",type=str,default="amazon",
        choices=["amazon", "dslr", "webcam", "art", "clipart", "product", "real_word"],
        help="Where to extract target data from.",
    )
    parser.add_argument("--nsamples", type=int, default=32, help="Number of calibration data samples.")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size.")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--dga_lr", type=float, default=5e-3)
    parser.add_argument("--lwc_lr", type=float, default=1e-2)
    parser.add_argument("--lac_lr", type=float, default=1e-4)
    parser.add_argument("--wrc_lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dga",default=False, action="store_true",help="distribution-guided aware smoothing technique")
    parser.add_argument("--lwc",default=False, action="store_true",help="activate learnable weight clipping")
    parser.add_argument("--lac",default=False, action="store_true",help="activate learnable activation clipping")
    parser.add_argument("--wrc",default=False, action="store_true", help="activate learnable weight clipping")
    parser.add_argument("--symmetric",default=False, action="store_true", help="symmetric quantization")
    parser.add_argument("--a_dynamic_method", type=str, default="per_token", choices=["per_token"])
    parser.add_argument("--w_dynamic_method", type=str, default="per_channel", choices=["per_channel"])
    parser.add_argument("--deactive_amp", action="store_true", help="deactivate AMP when 8<=bits<16")
    parser.add_argument("--net", type=str, default=None, choices=net_choices)
    parser.add_argument("--act-scales", type=str, default=None)
    parser.add_argument("--act-shifts", type=str, default=None)
    print('-----------------------start----------------------------')

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # check
    if args.epochs > 0:
        assert args.dga or args.lwc or args.lac or args.wrc
        pass
        
    if (args.wbits<16 and args.wbits>=8) or (args.abits<16 and args.abits>=8):
        args.deactive_amp = True

    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    # load model
    if args.net is None:
        args.net = args.model.split('/')[-1]
    assert args.net in net_choices
    args.model_family = args.net.split('-')[0]
    vits = ViTClass(args)
    vits.model.eval()
    for param in vits.model.parameters():
        param.requires_grad = False

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": args.symmetric,
        "dynamic_method": args.w_dynamic_method,
        "group_size": None,
        "lwc": args.lwc,
        "wrc": args.wrc
    }
    args.act_quant_params = {
        "n_bits":  args.abits,
        "per_channel_axes": [],
        "symmetric":False,
        "dynamic_method": args.a_dynamic_method,
        "lac": args.lac
    }


    # act scales and shifts
    if args.act_scales is None:
        args.act_scales = f'./act_scales/{args.net}.pt'
    if args.act_shifts is None:
        args.act_shifts = f'./act_shifts/{args.net}.pt'

    # quantization
    if args.wbits < 16 or args.abits <16:
        logger.info("=== start quantization ===")
        tick = time.time()     
        # load calibration dataset
        cache_dataloader = f'{args.cache_dir}/dataloader_{args.model_family}_{args.calib_dataset}_{args.nsamples}.cache'
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            logger.info(f"load calibration from {cache_dataloader}")
        else:
            dataloader, _ = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
            )
            torch.save(dataloader, cache_dataloader)    
        act_scales = None
        act_shifts = None
        target_dataloader = None
        
        if args.tl_loss:
            target_dataloader, _ = get_loaders(
                    args.target_dataset,
                    nsamples=args.nsamples,
                )
        if args.dga:
            act_scales = torch.load(args.act_scales)
            act_shifts = torch.load(args.act_shifts)
        DAQuant(
            vits,
            args,
            dataloader,
            act_scales,
            act_shifts,
            logger,
            target_dataloader,
        )
        logger.info(time.time() - tick)
    
    if args.save_dir:
        # delete daquant parameters
        for name, module in vits.model.named_modules():
            if isinstance(module, QuantLinear):
                # del module.weight_quantizer.compensation_factor
                del module.weight_quantizer.upbound_factor
                del module.weight_quantizer.lowbound_factor
                del module.act_quantizer.upbound_activation_factor
                del module.act_quantizer.lowbound_activation_factor
            if isinstance(module, QuantViTLayer):
                if args.dga:
                    del module.qkv_smooth_scale
                    del module.qkv_smooth_shift
                    del module.fc2_smooth_scale
                    del module.fc2_smooth_shift
                    del module.fc1_smooth_scale
                    del module.fc1_smooth_shift           
        vits.model.save_pretrained(args.save_dir)  
    print('---------------------')
    torch.cuda.empty_cache()

    vits.model.eval()
    vits.model.to(vits.device)

    
    if args.tl:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.RandomCrop((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        
        if  'clipart' in args.target_dataset:
            logger.info("evaluate: clipart")
            train_loader, val_loader = get_clipart()
        elif  'art' in args.target_dataset:
            logger.info("evaluate: art")
            train_loader, val_loader = get_art()
        elif  'product' in args.target_dataset:
            logger.info("evaluate: product")
            train_loader, val_loader = get_product()
        elif  'real_word' in args.target_dataset:
            logger.info("evaluate: real_word")
            train_loader, val_loader = get_real_word()
        else:
            # Load the datasets
            file_dataset = None
            if 'amazon' in args.target_dataset:
                file_dataset = Amazon(path='/PATH/TO/OFFICE-31/AMAZON', transforms=data_transforms['test'])
            elif 'dslr' in args.target_dataset:
                file_dataset = DSLR(path='/PATH/TO/OFFICE-31/DSLR', transforms=data_transforms['test'])
            else:
                file_dataset = Webcam(path='/PATH/TO/OFFICE-31/WEBCAM', transforms=data_transforms['test'])

            val_loader = DataLoader(
                file_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=8,
                pin_memory=False,
            )
        vits.model.to(vits.device)
        transfer_learning_test(vits.model, val_loader, logger, vits.device)
    elif 'ImageNet' in args.calib_dataset:
        # load ImageNet dataset
        imagenet_dataloader = eval("{}DataLoader".format('ImageNet'))(
                            args.net,
                            data_dir=os.path.join('/PATH/TO/IMAGENET', 'ImageNet'),
                            image_size=224,
                            batch_size=32,
                            num_workers=2,
                            split='val')

        # evaluate
        begin_time = time.time()
        accuracy = evaluate_model(vits, vits.processor, imagenet_dataloader, vits.device)
        logger.info(f"consumer time: {time.time()-begin_time}")

        logger.info(f"Model Accuracy on ImageNet: {accuracy}")


# evaluate Imagenet dataset
def evaluate_model(lm, processor, dataloader, dev):
    total_samples = len(dataloader.dataset)
    correct_predictions = 0
    lm.model.eval()
    
    with autocast(dtype=torch.float16):
        for data in dataloader:
            inputs = {"pixel_values": data[0].to(lm.device)}
            labels = data[1].to(dev)
            outputs = lm.model(**inputs)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(predictions == labels)
    accuracy = correct_predictions.item() / total_samples
    return accuracy



# evaluate domain adaptation
def transfer_learning_test(model, dataset, logger, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        bar = tqdm(dataset,total=len(dataset))
        for data in bar:
            inputs, labels = data
            labels = labels.to(device)
            inputs = {"pixel_values": inputs.to(device)}
            outputs = model(**inputs)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    logger.info(f"Accuracy on test set: {accuracy}")    


if __name__ == "__main__":
    print(sys.argv)
    main()
