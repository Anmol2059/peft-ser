import torch
import argparse
from torch import nn
import transformers_models_wavlm as wavlm
import loralib as lora
from transformers import WavLMModel

class WavLMEncoderLayer(nn.Module):
    def __init__(self, config, has_relative_position_bias: bool = True):
        super().__init__()
        self.attention = wavlm.WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = wavlm.WavLMFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.config = config

        if self.config.finetune_method == "lora":
            self.feed_forward.intermediate_dense = lora.Linear(config.hidden_size, config.intermediate_size, r=config.lora_rank)
            self.feed_forward.output_dense = lora.Linear(config.intermediate_size, config.hidden_size, r=config.lora_rank)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)
        outputs = (hidden_states, position_bias)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class WavLMWrapper(nn.Module):
    def __init__(self, args):
        super(WavLMWrapper, self).__init__()
        self.args = args
        self.backbone_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus", output_hidden_states=True)
        state_dict = self.backbone_model.state_dict()
        self.model_config = self.backbone_model.config
        self.model_config.finetune_method = args.finetune_method
        self.model_config.lora_rank = args.lora_rank

        self.backbone_model.encoder.layers = nn.ModuleList(
            [WavLMEncoderLayer(self.model_config, has_relative_position_bias=(i == 0)) for i in range(self.model_config.num_hidden_layers)]
        )

        msg = self.backbone_model.load_state_dict(state_dict, strict=False)

        if self.args.finetune_method == "lora":
            for name, p in self.backbone_model.named_parameters():
                if name in msg.missing_keys: p.requires_grad = True
                else: p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.backbone_model.feature_extractor(x)
            x = x.transpose(1, 2)
            x, _ = self.backbone_model.feature_projection(x)

        x = self.backbone_model.encoder(x, output_hidden_states=True).hidden_states[-1]
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WavLM finetune experiments')
    parser.add_argument('--finetune_method', default='lora', type=str, help='finetune method: lora')
    parser.add_argument('--lora_rank', default=32, type=int, help='LoRA rank')
    args = parser.parse_args()
    model = WavLMWrapper(args)
    data = torch.zeros([1, 16000])
    output = model(data)
    print(output.shape)