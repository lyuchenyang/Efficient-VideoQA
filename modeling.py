from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import CLIPProcessor, CLIPModel

from dataclasses import dataclass
from typing import Dict
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch import nn


class AADVConcat(nn.Module):
    def __init__(self, config, CLIPConfig):
        super().__init__()
        self.config = config

        self.clip = CLIPModel(CLIPConfig)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True

        self.query_multi_attention = nn.MultiheadAttention(config.transformer_width, config.attention_heads,
                                                           dropout=attn_dropout,
                                                           add_bias_kv=is_add_bias_kv,
                                                           add_zero_attn=is_add_zero_attn)

        self.video_to_multimodal = nn.Linear(in_features=config.transformer_width,
                                             out_features=config.transformer_width)
        self.text_to_multimodal = nn.Linear(in_features=config.transformer_width, out_features=config.transformer_width)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.loss_fct = CrossEntropyLoss()

    def forward(self, inputs):
        """
        :param inputs:
                    image_frames: B x N
                    question: B x 77
                    opt1: B x 77
                    opt2: B x 77
                    opt3: B x 77
                    opt4: B x 77
                    ans: B x 1

        :return: loss when training else None
        """
        image_features = self.clip.visual_projection(self.clip.vision_model(inputs['image_frames'])[0]).transpose(0, 1)

        question, options = self.encode_questions_and_options(inputs)

        question_attn_image_features = \
            self.query_multi_attention(question.transpose(0, 1), image_features, image_features)[0].transpose(0, 1)

        # n_video_features = torch.nn.functional.normalize(self.video_to_multimodal(question_attn_image_features), p=2,
        #                                                  dim=-1)
        # n_option_features = torch.nn.functional.normalize(self.text_to_multimodal(options), p=2, dim=-1)

        n_video_features = self.video_to_multimodal(question_attn_image_features)
        n_option_features = self.text_to_multimodal(options)

        logit_scale = self.logit_scale.exp()

        sim_matrix = torch.bmm(logit_scale * n_video_features, n_option_features.transpose(1, 2)).squeeze(1)

        if 'ans' in inputs:

            labels = inputs['ans']

            loss = self.loss_fct(sim_matrix, labels)

            return loss
        else:
            return sim_matrix

    def encode_questions_and_options(self, inputs):
        attn_mask = 1 - (inputs['question'] == 0).long()
        question = self.clip.get_text_features(inputs['question'], attention_mask=attn_mask).unsqueeze(1)

        attn_mask = 1 - (inputs['opt1'] == 0).long()
        opt1 = self.clip.get_text_features(inputs['opt1'], attention_mask=attn_mask).unsqueeze(1)

        attn_mask = 1 - (inputs['opt2'] == 0).long()
        opt2 = self.clip.get_text_features(inputs['opt2'], attention_mask=attn_mask).unsqueeze(1)

        attn_mask = 1 - (inputs['opt3'] == 0).long()
        opt3 = self.clip.get_text_features(inputs['opt3'], attention_mask=attn_mask).unsqueeze(1)

        attn_mask = 1 - (inputs['opt4'] == 0).long()
        opt4 = self.clip.get_text_features(inputs['opt4'], attention_mask=attn_mask).unsqueeze(1)

        options = torch.cat([opt1, opt2, opt3, opt4], dim=1)

        return question, options

class MSRVTTConcat(nn.Module):
    def __init__(self, config, CLIPConfig):
        super().__init__()
        self.config = config

        self.clip = CLIPModel(CLIPConfig)

        attn_dropout = 0.1
        is_add_bias_kv = True
        is_add_zero_attn = True

        self.query_multi_attention = nn.MultiheadAttention(self.config.transformer_width, config.attention_heads,
                                                           dropout=attn_dropout,
                                                           add_bias_kv=is_add_bias_kv,
                                                           add_zero_attn=is_add_zero_attn)

        self.video_to_multimodal = nn.Sequential(nn.Linear(config.transformer_width,
                                                           config.transformer_width))
        self.text_to_multimodal = nn.Sequential(nn.Linear(config.transformer_width,
                                                          config.transformer_width))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.loss_fct = CrossEntropyLoss()

    def forward(self, inputs):
        """
        :param inputs:
                    image_frames: B x N
                    audio: B x 2 x 10000
                    summary: B x 77
                    script: B x 77
                    dialog: B x 10 x 77
                    all_ans: B x 10 x 77
                    all_frames: B x 768/512

        :return: loss when training else None
        """
        # pre-injection of all_frame feature
        image_features = self.encode_image(inputs['image_frames'])

        attn_mask = 1 - (inputs['captions'] == 0).long()
        text_features = self.clip.get_text_features(inputs['captions'],
                                                    attention_mask=attn_mask)

        # # text_features = self.transform_text_to_inner_feature(text_features)
        # r_text_features = text_features.unsqueeze(0).repeat(image_features.size(0), 1, 1)  # added repeat
        #
        # image_features = image_features.transpose(0, 1)
        # query_to_image_attn = self.query_multi_attention(r_text_features.transpose(0, 1),
        #                                                  image_features, image_features)[0].transpose(0, 1)
        #
        # image_features = query_to_image_attn

        # with l2 norm
        n_video_features = torch.nn.functional.normalize(self.video_to_multimodal(image_features), p=2, dim=-1)
        n_text_features = torch.nn.functional.normalize(self.text_to_multimodal(text_features), p=2, dim=-1)

        logit_scale = self.logit_scale.exp()

        # original multiply
        logits = torch.mm(logit_scale * n_video_features, n_text_features.t())

        # n_text_features = n_text_features.unsqueeze(1)
        # logits = torch.bmm(logit_scale * n_video_features.transpose(0, 1).contiguous(),
        #                    n_text_features.transpose(1, 2).contiguous()).squeeze(-1)

        labels = torch.tensor([i for i in range(inputs['captions'].size(0))], dtype=torch.long,
                              device=self.config.device)

        loss_i = self.loss_fct(logits, labels)
        loss_e = self.loss_fct(logits.t(), labels)

        loss = (loss_i + loss_e) / 2

        return loss

    def encode_image(self, images):
        # simple image encoding without temporal embedding and self attention
        # image_features = self.clip.visual_projection(self.clip.vision_model(images)[0])
        # image_features = torch.max(image_features, dim=1).values
        # image_features = torch.sum(image_features, dim=1) / image_features.size(1)

        image_features = self.clip.get_image_features(images)

        return image_features


@dataclass
class ModelDimensions:
    n_mels: int
    n_audio_ctx: int
    n_audio_state: int
    n_audio_head: int
    n_audio_layer: int
    n_vocab: int
    n_text_ctx: int
    n_text_state: int
    n_text_head: int
    n_text_layer: int


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype)
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


def sinusoids(length, channels, max_timescale=10000):
    """Returns sinusoids for positional embedding"""
    assert channels % 2 == 0
    log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
    inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2))
    scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
    return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            # hooks, if installed (i.e. kv_cache is not None), will prepend the cached kv tensors;
            # otherwise, perform key/value projections for self- or cross-attention as usual.
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
        else:
            # for cross-attention, calculate keys and values once and reuse in subsequent calls.
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv = self.qkv_attention(q, k, v, mask)
        return self.out(wv)

    def qkv_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        qk = q @ k
        if mask is not None:
            qk = qk + mask[:n_ctx, :n_ctx]

        w = F.softmax(qk.float(), dim=-1).to(q.dtype)
        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x
