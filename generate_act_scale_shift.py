import torch
import os
from transformers import ViTImageProcessor, ViTForImageClassification
import argparse
import torch.nn as nn
import functools
from tqdm import tqdm
from datautils import get_loaders


def get_act_scales(model, dataloader, num_samples, model_name, calib_dataset):
    model.eval()
    device = next(model.parameters()).device
    act_scales = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).abs().detach()
        comming_std = torch.std(tensor, dim=0).float().cpu()
        if name in act_scales:
            act_scales[name] = torch.max(act_scales[name], comming_std)
        else:
            act_scales[name] = comming_std

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name)))
    
    processor = ViTImageProcessor.from_pretrained(model_name)
    for i in tqdm(range(num_samples)):
        if "ImageNet" in calib_dataset and dataloader[i].mode != 'L':
            inputs = processor(images=dataloader[i], return_tensors="pt")
            model(**inputs.to(device))
        else:
            inputs = {'pixel_values': dataloader[i].to(device)}
            model(**inputs)

    for h in hooks:
        h.remove()

    return act_scales


def get_act_shifts(model, dataloader, num_samples, model_name, calib_dataset):
    model.eval()
    device = next(model.parameters()).device
    act_shifts = {}

    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]
        tensor = tensor.view(-1, hidden_dim).detach()
        comming_mean = torch.mean(tensor, dim=0).float().cpu()
        if name in act_shifts:
            act_shifts[name] = 0.99*act_shifts[name] + 0.01 *comming_mean
        else:
            act_shifts[name] = comming_mean

    def stat_input_hook(m, x, y, name):
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            hooks.append(
                m.register_forward_hook(
                    functools.partial(stat_input_hook, name=name))
            )
            
    processor = ViTImageProcessor.from_pretrained(model_name)
    for i in tqdm(range(num_samples)):
        if "ImageNet" in calib_dataset and dataloader[i].mode != 'L':
            inputs = processor(images=dataloader[i], return_tensors="pt")
            model(**inputs.to(device))
        else:
            inputs = {'pixel_values': dataloader[i].to(device)}
            model(**inputs)

    for h in hooks:
        h.remove()

    return act_shifts

def build_vit_model(model_name):
    model = ViTForImageClassification.from_pretrained(model_name)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/PATH/TO/DEIT/DEIT-S', help='model name')
    parser.add_argument('--full_model', type=str,
                        default='/PATH/TO/PRE-TRAIN-IN/OFFICE-31/DEIT/DEIT-S', help='model name')
    parser.add_argument('--scales-output-path', type=str, default='./act_scales/',
                        help='where to save the act scales')
    parser.add_argument('--shifts-output-path', type=str, default='./act_shifts/',
                        help='where to save the act shifts')
    parser.add_argument("--calib_dataset",type=str,default="ImageNet",
        choices=["webcam", "amazon", "dslr", "ImageNet", "art", "clipart", "product", "real_word"],
        help="Where to extract calibration data from.",)
    parser.add_argument('--num-samples', type=int, default=32)
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model = build_vit_model(args.model)
    dataloader, _ = get_loaders(
    args.calib_dataset,
    nsamples=args.num_samples,
    )

    args.net = args.model.split('/')[-1]
    act_scales = get_act_scales(model, dataloader, args.num_samples, args.full_model, args.calib_dataset)
    save_path = os.path.join(args.scales_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_scales, save_path)

    act_shifts = get_act_shifts(model, dataloader, args.num_samples, args.full_model, args.calib_dataset)
    save_path = os.path.join(args.shifts_output_path,f'{args.net}.pt')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(act_shifts, save_path)

if __name__ == '__main__':
    main()
