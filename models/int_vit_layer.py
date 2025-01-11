import torch
from torch import nn
from typing import Optional, Tuple, Optional, Set, Tuple, Union

from quantize.int_linear import QuantLinear
from collections import OrderedDict
import math
from models.transformation import *
from quantize.daq_norm import DAQLayerNorm
from transformers.models.vit.configuration_vit import ViTConfig

from transformers.pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from transformers.activations import ACT2FN

class QuantViTSelfAttention(nn.Module):
    def __init__(self, 
                 org_module:nn.Module,
                 config: ViTConfig,
                 args=None) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = QuantLinear(
            org_module.query,
            args.weight_quant_params,
            args.act_quant_params                                
            )
        self.key = QuantLinear(
            org_module.key,
            args.weight_quant_params,
            args.act_quant_params
            )
        self.value = QuantLinear(
            org_module.value,
            args.weight_quant_params,
            args.act_quant_params
            )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.use_weight_quant = False
        self.use_act_quant = False

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear)):
                m.set_quant_state(weight_quant, act_quant)


class QuantViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, 
                 org_module:nn.Module,
                 config: ViTConfig,
                 args=None) -> None:
        super().__init__()
        self.dense = QuantLinear(
            org_module.dense,
            args.weight_quant_params,
            args.act_quant_params
            )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.use_weight_quant = False
        self.use_act_quant = False

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear)):
                m.set_quant_state(weight_quant, act_quant)


class QuantViTAttention(nn.Module):
    def __init__(self,
                 org_module:nn.Module,
                 config: ViTConfig,
                 args=None) -> None:
        super().__init__()
        self.attention = QuantViTSelfAttention(
            org_module = org_module.attention,
            config=config,
            args=args
        )
        self.output = QuantViTSelfOutput(
            org_module = org_module.output,
            config=config,
            args=args
        )
        self.pruned_heads = set()
        self.use_weight_quant = False
        self.use_act_quant = False

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, (QuantLinear)):
                m.set_quant_state(weight_quant, act_quant)


class QuantViTIntermediate(nn.Module):
    def __init__(self, 
                 org_module:nn.Module,
                 config: ViTConfig,
                 args=None) -> None:
        super().__init__()
        self.dense = QuantLinear(
            org_module.dense,
            args.weight_quant_params,
            args.act_quant_params
        )
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states

class QuantViTOutput(nn.Module):
    def __init__(self, 
                 org_module:nn.Module,
                 config: ViTConfig,
                 args=None) -> None:
        super().__init__()
        self.dense = QuantLinear(
            org_module.dense,
            args.weight_quant_params,
            args.act_quant_params
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states

class QuantViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, 
                 config: ViTConfig,
                 ori_layer,
                 args) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = QuantViTAttention(
            config = config,
            org_module = ori_layer.attention,
            args = args
        )
        self.intermediate = QuantViTIntermediate(
            config = config,
            org_module = ori_layer.intermediate,
            args = args
        )
        self.output = QuantViTOutput(
            config = config,
            org_module = ori_layer.output,
            args = args            
        )
        self.layernorm_before = DAQLayerNorm(ori_layer.layernorm_before)
        self.layernorm_after = DAQLayerNorm(ori_layer.layernorm_after)


    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        names = []
        for name, m in self.named_modules():
            if isinstance(m, (QuantLinear)):
                names.append(name)
                m.set_quant_state(weight_quant, act_quant)

    def smooth_and_quant_temporary(self):
        if self.dga:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_temporary(self.layernorm_before,[self.attention.attention.query, self.attention.attention.key, self.attention.attention.value],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            
            smooth_ln_fcs_temporary(self.layernorm_after,[self.intermediate.dense],
                                    self.fc2_smooth_scale,self.fc2_smooth_shift)
            
            smooth_fc_fc_temporary(self.attention.attention.value, self.attention.output.dense,
                                self.fc1_smooth_scale, self.fc1_smooth_shift)
            
            
            self.output.dense.temp_weight = self.output.dense.weight.detach()
        else:
            for name, module in self.named_modules():
                if isinstance(module, QuantLinear):
                    module.temp_weight = module.weight.detach()


        # quant
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                if hasattr(module, "temp_weight"):
                    module.temp_weight = module.weight_quantizer(module.temp_weight)
                else:
                    module.temp_weight = module.weight_quantizer(module.weight)
                if not hasattr(module, "temp_bias"):
                    module.temp_bias = module.bias
                module.use_temporary_parameter=True

    def clear_temp_variable(self):
       for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                del module.temp_weight
                del module.temp_bias

    @torch.no_grad()
    def smooth_and_quant_inplace(self):
        if self.dga:
            with torch.no_grad():
                for name, module in self.named_parameters():
                    if "smooth_scale" in name:
                        module.data = truncate_number(module)
            smooth_ln_fcs_inplace(self.layernorm_before,[self.attention.attention.query, self.attention.attention.key, self.attention.attention.value],
                                    self.qkv_smooth_scale,self.qkv_smooth_shift)
            
            smooth_ln_fcs_inplace(self.layernorm_after,[self.intermediate.dense],
                                    self.fc2_smooth_scale,self.fc2_smooth_shift)
            
            smooth_fc_fc_inplace(self.attention.attention.value, self.attention.output.dense,
                                self.fc1_smooth_scale, self.fc1_smooth_shift)
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
                module.use_temporary_parameter=False

    def dga_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find(template) > -1:
                params.append(m)
        return iter(params)  

    def lwc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1:
                params.append(m)
        return iter(params)  
    
    def wrc_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('compensation_factor') > -1:
                params.append(m)
        return iter(params) 

    def lac_parameters(self):
        params = []
        for n, m in self.named_parameters():
            if n.find('activation_factor') > -1:
                params.append(m)
        return iter(params)

    def adq_parameters(self, use_shift=True):
        params = []
        template = "smooth" if use_shift else "smooth_scale"
        for n, m in self.named_parameters():
            if n.find('bound_factor') > -1 or n.find(template) > -1:
                params.append(m)
        return iter(params)  
    
    def adq_state_dict(self, destination=None, prefix='', keep_vars=False):
        if destination is None:
            destination = OrderedDict()
        for name, param in self.named_parameters():
            if name.find('smooth') > -1 or name.find('bound_factor') > -1:
                destination[prefix + name] = param if keep_vars else param.detach()
        return destination
    
    def register_scales_and_zeros(self):
        for name, module in self.named_modules():
            if isinstance(module, QuantLinear):
                module.weight_quantizer.register_scales_and_zeros()
