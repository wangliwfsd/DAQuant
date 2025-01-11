import torch
import torch.nn as nn
from torch.nn import functional as F
from models.int_vit_layer import QuantViTLayer
from quantize.int_linear import QuantLinear
import auto_gptq.nn_modules.qlinear.qlinear_cuda as qlinear_cuda
from contextlib import nullcontext
import copy
import math
import utils
import os
import pdb
import gc



def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, QuantLinear)}


def DAQuant(
    vits,
    args,
    dataloader,
    act_scales,
    act_shifts,
    logger,
    target_dataloader=None,
):
    logger.info("Starting ...")
    
    # move embedding layer and first layer to target device
    model = vits.model
    processor = vits.processor
    dev = vits.device
    model.to(dev)

    if "vit" in args.net.lower() or 'deit' in args.net.lower():
        layers = model.vit.encoder.layer
        model.vit.embeddings.patch_embeddings = model.vit.embeddings.patch_embeddings.to(dev)
        model.vit.layernorm = model.vit.layernorm.to(dev)
        DecoderLayer = QuantViTLayer
        pairs = {
            "query":"qkv",
            "attention.output.dense":"fc1",
            "intermediate.dense":"fc2",
        }
        layer_name_prefix = "vit.encoder.layer"
    else:
        raise ValueError("Only support for vit/deit now")
    
    
    layers[0] = layers[0].to(dev)
    if args.deactive_amp and args.epochs>0:
        dtype = torch.float
        traincast = nullcontext
    else:
        dtype = torch.float16
        traincast = torch.cuda.amp.autocast

    input_dimension = (model.config.image_size * model.config.image_size) // (model.config.patch_size * model.config.patch_size)
    inps = torch.zeros(
            (args.nsamples, input_dimension+1, model.config.hidden_size), dtype=dtype, device=dev
        )
    inps_target = torch.zeros(
            (args.nsamples, input_dimension+1, model.config.hidden_size), dtype=dtype, device=dev
        )
    cache = {"i": 0}

    # catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.is_llama = False

        def forward(self, inp, inp2, inp3, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            raise ValueError
    
    cache_target = {"i": 0}
        
    # catch the first layer input
    class Catcher_target(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, inp2, inp3, **kwargs):
            inps_target[cache_target["i"]] = inp
            cache_target["i"] += 1
            raise ValueError

    layers[0] = Catcher(layers[0])

    with torch.no_grad():
        for batch in dataloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                if "ImageNet" in args.calib_dataset:
                    inputs = processor(images=batch, return_tensors="pt")
                    inputs = inputs.to(dev)
                else:
                    inputs = {'pixel_values': batch.to(dev)}
                model(**inputs) 
            except ValueError:
                pass
            
    # move embedding layer and first layer to cpu
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    
    if args.tl_loss:
        layers[0] = Catcher_target(layers[0])
        
        with torch.no_grad():
            for batch in target_dataloader:
                if cache_target["i"] >= args.nsamples:
                    break
                try:
                    inputs = {'pixel_values': batch.to(dev)}
                    model(**inputs) 
                except ValueError:
                    pass
    
        # move embedding layer and first layer to cpu
        layers[0] = layers[0].module
        layers[0] = layers[0].cpu()

    model.to('cpu')
    
    if "vit" in args.net.lower() or "deit" in args.net.lower():
        model.vit.embeddings.patch_embeddings = model.vit.embeddings.patch_embeddings.to('cpu')
        model.vit.layernorm = model.vit.layernorm.to('cpu')
    else:
        raise ValueError("Only support for vit/deit now")
    torch.cuda.empty_cache()

    
    # same input of first layer for fp model and quant model
    quant_inps = inps
    target_inps = copy.deepcopy(inps_target)
    fp_inps = copy.deepcopy(inps)   # take output of fp model as input
    

    if args.resume:
        daq_parameters = torch.load(args.resume)
    else:
        daq_parameters = {}

    for i in range(len(layers)):
        logger.info(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = DecoderLayer(vits.model.config, layer, args)
        qlayer = qlayer.to(dev)

        
        # obtain output of full-precision model
        qlayer.set_quant_state(weight_quant=False, act_quant=False)
        if args.epochs > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        fp_inps[j] = qlayer(fp_inps[j].unsqueeze(0))[0]
    
            if args.tl_loss:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        for j in range(args.nsamples):
                            target_inps[j] = qlayer(target_inps[j].unsqueeze(0))[0]

        # init smooth parameters
        qlayer.set_quant_state(weight_quant=False, act_quant=True)  # weight will be manually quantized before forward
        qlayer.dga = args.dga
        use_shift = True 
        
        if args.dga:
            # init channel-wise scaling and shift
            for name,module in qlayer.named_modules():
                if isinstance(module, QuantLinear):
                    for key in pairs.keys():
                        if key in name:
                            act = act_scales[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype).clamp(min=1e-5)
                            weight = module.weight.std(dim=0)[0].clamp(min=1e-5)
                            scale = (act.pow(0.5)/weight.pow(0.5)).clamp(min=1e-5)
                            if use_shift:
                                shift = act_shifts[f"{layer_name_prefix}.{i}.{name}"].to(device=dev, dtype=dtype)
                            else:
                                shift = torch.zeros_like(scale)
                            qlayer.register_parameter(f"{pairs[key]}_smooth_shift",torch.nn.Parameter(shift))
                            qlayer.register_parameter(f"{pairs[key]}_smooth_scale",torch.nn.Parameter(scale))
                        
                                
        if args.resume:
            qlayer.load_state_dict(daq_parameters[i], strict=False)
        

        if args.epochs > 0:
            with torch.no_grad():
                qlayer.float()      
            # create optimizer
            optimizer = torch.optim.AdamW(
                [{"params":qlayer.dga_parameters(use_shift),"lr":args.dga_lr}, 
                 {"params":qlayer.lwc_parameters(),"lr":args.lwc_lr}, 
                 {"params":qlayer.wrc_parameters(),"lr":args.wrc_lr},
                 {"params":qlayer.lac_parameters(),"lr":args.lac_lr}],weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            
            for epochs in range(args.epochs):
                loss_list = []
                norm_list = []
                for j in range(args.nsamples//args.batch_size):    
                    index = j * args.batch_size
                    # obtain output of quantization model
                    with traincast():
                        qlayer.smooth_and_quant_temporary()
                        quant_out = qlayer(quant_inps[index:index+args.batch_size,])[0]
                        kl_loss = nn.KLDivLoss(reduction='mean')
                        criterion = nn.SmoothL1Loss()
                        loss = criterion(fp_inps[index:index+args.batch_size,], quant_out)
                        
                        # domain adaptation
                        if args.tl_loss:
                            quant_out_log = F.log_softmax(quant_out,dim=2)
                            y = F.softmax(target_inps[index:index+args.batch_size,],dim=2)
                            loss += (torch.abs(kl_loss(quant_out_log, y))*args.tl_weight)
                        
                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                        
                    loss_list.append(loss.data)
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer,parameters=qlayer.adq_parameters(use_shift))
                    norm_list.append(norm.data)

                loss_mean = torch.stack(loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(f"layer {i} iter {epochs} loss:{loss_mean} norm:{norm_mean} max memory_allocated {torch.cuda.max_memory_allocated(vits._device) / 1024**2} ")
            qlayer.clear_temp_variable()
            del optimizer

        # real smooth and quantization
        qlayer.smooth_and_quant_inplace()
        if args.epochs>0:
            # update input of quantization model
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for j in range(args.nsamples):
                        quant_inps[j] = qlayer(quant_inps[j].unsqueeze(0))[0]
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
            daq_parameters[i] = qlayer.adq_state_dict()
            torch.save(daq_parameters, os.path.join(args.output_dir, f"daq_parameters.pth"))
        else:
            qlayer.register_scales_and_zeros()
            qlayer.half()
            layers[i] = qlayer.to("cpu")
        if args.real_quant:
            named_linears = get_named_linears(qlayer)
            for name, module in named_linears.items():
                scales = module.weight_quantizer.scales
                zeros = module.weight_quantizer.zeros
                group_size = -1
                dim0 = module.weight.shape[0]
                scales = scales.view(dim0,-1)
                zeros = zeros.view(dim0,-1)
                q_linear = qlinear_cuda.QuantLinear(args.wbits, group_size, module.in_features,module.out_features,not module.bias is None)
                q_linear.pack(module.float().cpu(),  scales.float().cpu(), zeros.float().cpu())
                
                levels = name.split('.')
                if len(levels) > 1:
                    mod_ = qlayer
                    for l_idx in range(len(levels)-1):
                        if levels[l_idx].isdigit():
                            mod_ = mod_[int(levels[l_idx])]
                        else:
                            mod_ = getattr(mod_, levels[l_idx])
                    setattr(mod_, levels[-1], q_linear)
                else:
                    setattr(qlayer, name, q_linear)        
                del module        
        del layer
        torch.cuda.empty_cache()

    del inps
    del quant_inps
    del fp_inps
    torch.cuda.empty_cache()
    gc.collect()                    
    return model

