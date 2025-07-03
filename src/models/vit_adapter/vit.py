#!/usr/bin/env python3
"""
vit with adapter
"""
import copy
import numpy as np
import torch
import torch.nn as nn

from scipy import ndimage
from easydict import EasyDict
from torch.nn import Linear, LayerNorm
from ..vit_backbones.vit import *
from ...utils import logging

logger = logging.get_logger("FS-OOD")


######============================= Adapterformer  =============================######
class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        # _before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output


class ADPTF_Block(nn.Module):
    def __init__(self, config, vis, adapter_config):
        super(ADPTF_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.adapter_config = adapter_config
        if adapter_config.ffn_adapt:
            self.adapter_mlp = Adapter(self.adapter_config, dropout=0.1, bottleneck=self.adapter_config.ffn_num,
                                       init_option=self.adapter_config.ffn_adapter_init_option,
                                       adapter_scalar=self.adapter_config.ffn_adapter_scalar,
                                       adapter_layernorm_option=self.adapter_config.ffn_adapter_layernorm_option,
                                       )

        if self.adapter_config.vpt_on:
            assert self.adapter_config.vpt_num > 0, self.adapter_config.vpt_num
            # properly registered
            self.adapter_embedding = nn.Parameter(
                torch.empty(1, self.adapter_config.vpt_num, config.hidden_size),
                requires_grad=True
            )
            torch.nn.init.xavier_uniform_(self.adapter_embedding.data)

    def forward(self, x):

        if self.adapter_config.vpt_on:
            eee = self.adapter_embedding.expand(self.adapter_config.batch_size, -1, -1)
            x = torch.cat([eee, x], dim=1)

        # same as reguluar ViT block
        h = x
        x = self.attention_norm(x)  # layer normal
        x, weights = self.attn(x)  # attention
        x = x + h  # sum of attn and input

        residual = x  # sum of attn and input
        mlp_branch = residual
        adp_branch = residual

        # adapter begins here
        # adapter branch
        if self.adapter_config.ffn_adapt and self.adapter_config.ffn_option == 'parallel':
            adapt_x = self.adapter_mlp(adp_branch, add_residual=False)

        # mlp branch
        mlp_branch = self.ffn_norm(mlp_branch)
        mlp_branch = self.ffn(mlp_branch)

        # sum
        if self.adapter_config.ffn_adapt:
            if self.adapter_config.ffn_option == 'sequential':
                sum_adp = self.adapter_mlp(mlp_branch)
            elif self.adapter_config.ffn_option == 'parallel':
                sum_adp = mlp_branch + adapt_x
            else:
                raise ValueError(self.adapter_config.ffn_adapt)
        # adapter ends here

        x = residual + sum_adp

        if self.adapter_config.vpt_on:
            x = x[:, self.adapter_config.vpt_num:, :]
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


######============================= Adapterformer  =============================######


class ADPT_Block(nn.Module):
    def __init__(self, config, vis, adapter_config):
        super(ADPT_Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

        self.adapter_config = adapter_config

        if adapter_config.STYLE == "Pfeiffer":
            self.adapter_downsample = nn.Linear(
                config.hidden_size,
                config.hidden_size // adapter_config.REDUCATION_FACTOR
            )
            self.adapter_upsample = nn.Linear(
                config.hidden_size // adapter_config.REDUCATION_FACTOR,
                config.hidden_size
            )
            self.adapter_act_fn = ACT2FN["gelu"]

            nn.init.zeros_(self.adapter_downsample.weight)
            nn.init.zeros_(self.adapter_downsample.bias)

            nn.init.zeros_(self.adapter_upsample.weight)
            nn.init.zeros_(self.adapter_upsample.bias)
        else:
            raise ValueError("Other adapter styles are not supported.")

    def forward(self, x):
        if self.adapter_config.STYLE == "Pfeiffer":
            # same as reguluar ViT block
            h = x
            x = self.attention_norm(x)
            x, weights = self.attn(x)
            x = x + h

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)

            # start to insert adapter layers...
            adpt = self.adapter_downsample(x)
            adpt = self.adapter_act_fn(adpt)
            adpt = self.adapter_upsample(adpt)
            x = adpt + x
            # ...end

            x = x + h
            return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                   self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class ADPT_Encoder(nn.Module):
    def __init__(self, config, vis, adapter_cfg):
        super(ADPT_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)

        self.num_layers = config.transformer["num_layers"]
        for _ in range(self.num_layers):
            layer = ADPT_Block(config, vis, adapter_cfg)
            # layer = ADPTF_Block(config, vis, adapter_cfg)  ######============================= Adapterformer  =============================######
            self.layer.append(copy.deepcopy(layer))

        ######============================= Adapterformer  =============================######
        # self.adapter_cfg = adapter_cfg
        # if self.adapter_cfg.vpt_on:
        #     assert self.adapter_cfg.vpt_num > 0, self.adapter_cfg.vpt_num
        #     # properly registered
        #     self.adp_embeddings = nn.ParameterList(  # batch, num_prompt, embed_dim
        #         [nn.Parameter(torch.empty(1, self.adapter_cfg.vpt_num, config.hidden_size)) for _ in
        #          range(config.transformer.num_heads)])
        #     for eee in self.adp_embeddings:
        #         torch.nn.init.xavier_uniform_(eee.data)
        ######============================= Adapterformer  =============================######

    def forward(self, hidden_states):
        attn_weights = []
        for _, layer_block in enumerate(self.layer):
            # ######============================= Adapterformer  =============================######
            # if self.adapter_cfg.vpt_on:
            #     eee = self.adp_embeddings[idx].expand(self.adapter_cfg.batch_size, -1, -1)
            #     hidden_states = torch.cat([eee, hidden_states], dim=1)
            # ######============================= Adapterformer  =============================######

            hidden_states, weights = layer_block(hidden_states)

            # ######============================= Adapterformer  =============================######
            # if self.adapter_cfg.vpt_on:
            #     hidden_states = hidden_states[:, self.tuning_config.vpt_num:, :]
            # ######============================= Adapterformer  =============================######
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        ######============================= Adapterformer  =============================######
        # encoded = encoded[:, 0]
        ######============================= Adapterformer  =============================######
        return encoded, attn_weights


class ADPT_Transformer(nn.Module):
    def __init__(self, config, img_size, vis, adapter_cfg):
        super(ADPT_Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = ADPT_Encoder(config, vis, adapter_cfg)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)

        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class ADPT_VisionTransformer(nn.Module):
    def __init__(
            self, model_type,
            img_size=224, num_classes=21843, vis=False, adapter_cfg=None
    ):
        super(ADPT_VisionTransformer, self).__init__()
        config = CONFIGS[model_type]
        self.num_classes = num_classes
        self.classifier = config.classifier

        self.transformer = ADPT_Transformer(config, img_size, vis, adapter_cfg)
        self.head = Linear(config.hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, vis=False):
        x, attn_weights = self.transformer(x)
        # print(f'encoder output shape {x.size()}')
        logits = self.head(x[:, 0])

        if not vis:
            return logits
        return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(
                    np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)
