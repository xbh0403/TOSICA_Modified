#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From Uchida Takumi https://github.com/uchida-takumi/CustomizedLinear/blob/master/CustomizedLinear.py
extended torch.nn module which cusmize connection.
This code base on https://pytorch.org/docs/stable/notes/extending.html
"""
import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import pandas as pd
import pathlib

class CustomizedLinearFunction(torch.autograd.Function):
    """
    autograd function which masks it's weights by 'mask'.
    """

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias, mask is an optional argument
    def forward(ctx, input, weight, bias=None, mask=None, gene_embedding=None):
        # print("CustomizedLinearFunction.forward", "input", input.shape, "weight", weight.shape, "bias", bias.shape, "mask", mask.shape if mask is not None else "None")
        if mask is not None:
            # change weight to 0 where mask == 0
            # The shape of mask is (d, p, f) where d num_genes, p num_pathways, f num_features (768)
            # mask = (torch.expand_dims(gene_embedding, axis=2) * torch.expand_dims(mask, axis=0)).transpose(1,2,0)
            # mask = (gene_embedding.unsqueeze(2) * mask.unsqueeze(0)).transpose(1,2,0)
            # print('gene_embedding', gene_embedding.shape, 'gene_embedding.1', gene_embedding.T.unsqueeze(2).shape,'mask', mask.shape, 'mask.1', mask.T.unsqueeze(0).shape)
            mask = (gene_embedding.T.unsqueeze(2) * mask.T.unsqueeze(0))
            # mask = rearrange(mask, 'd p f -> d (p f)')
            mask = rearrange(mask, 'f d p -> d (p f)')
            # Write a equvalent np.repeat(weight, 768, axis=1) here using torch
            weight = repeat(weight, 'p d -> d (p repeat)', repeat=768) # TODO: Remove this one if we want all parameters to be trainable
            bias = repeat(bias, 'p -> (p repeat)', repeat=768)
            # print('weight in embed_layer',weight.shape, "mask", mask.shape)
            weight = weight * mask
        # weight = weight.t()
        # print('weight in embed_layer',weight.shape, "input", input.shape)
        # output = torch.einsum('nd,dpf->npf', input, mask)
        output = input.mm(weight)
        # print('output in embed_layer',output.shape)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                # change grad_weight to 0 where mask == 0
                # print('grad_weight', grad_weight.shape, 'mask', mask.shape)
                grad_weight = grad_weight * mask.t()
                # keep only when index mod 768 == 0
                grad_weight = grad_weight[::768, :]
        #if bias is not None and ctx.needs_input_grad[2]:
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)[::768]

        return grad_input, grad_weight, grad_bias, grad_mask, None


class CustomizedLinear(nn.Module):
    def __init__(self, mask, bias=True):
        """
        extended torch.nn module which mask connection.
        Args:
        mask [torch.tensor]:
            the shape is (n_input_feature, n_output_feature).
            the elements are 0 or 1 which declare un-connected or
            connected.
        bias [bool]:
            flg of bias.
        """
        super(CustomizedLinear, self).__init__()
        root = pathlib.Path(__file__).parent
        
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1] # TODO: Check if we need to * 768 here
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()

        self.mask = nn.Parameter(self.mask, requires_grad=False)

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))
        # self.gene_embedding = pd.read_csv('/Users/xbh0403/Desktop/TOSICA/TOSICA/resources/gene_embedding.csv', index_col=0).iloc[:, -768:] # TODO
        self.gene_embedding = pd.read_csv(root / 'resources/gene_embedding.csv', index_col=0).iloc[:, -768:] # TODO
        self.gene_embedding = torch.tensor(self.gene_embedding.values, dtype=torch.float).to("cuda:0")

        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)
        self.reset_parameters()

        # mask weight
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """ Same as reset_parameters, but only initialize to positive values. """
        stdv = 1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask, self.gene_embedding)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )





if __name__ == 'check grad':
    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.

    customlinear = CustomizedLinearFunction.apply

    input = (
            torch.randn(20,20,dtype=torch.double,requires_grad=True),
            torch.randn(30,20,dtype=torch.double,requires_grad=True),
            None,
            None,
            )
    test = gradcheck(customlinear, input, eps=1e-6, atol=1e-4)
    print(test)
