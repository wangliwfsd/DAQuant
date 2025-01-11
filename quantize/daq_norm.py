import torch
import torch.nn as nn


'''
Modify normalization layer to adapt the training of learnable equivalent transformation
'''



class DAQLayerNorm(nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.use_act_quant = True
        self.register_buffer('weight',ori_layer_norm.weight)
        if ori_layer_norm.bias is not None:
            self.register_buffer('bias',ori_layer_norm.bias)
        else:
            self.bias = None
        self.eps = ori_layer_norm.eps
        self.norm_func = nn.functional.layer_norm
        self.normalized_shape = ori_layer_norm.normalized_shape
        self.use_temporary_parameter = False


    def forward(self, x):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        else:
            weight = self.weight
            bias = self.bias
        out = self.norm_func(x,self.normalized_shape,weight.to(torch.float16), bias.to(torch.float16), eps=self.eps)
        return out

    def set_quant_state(self, use_weight_quant, use_act_quant):
        self.use_act_quant = use_act_quant


