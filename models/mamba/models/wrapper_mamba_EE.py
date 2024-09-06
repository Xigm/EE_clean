import json
from pathlib import Path
from typing import List, Optional, Union

import safetensors
import torch
import torch.nn as nn

from models.mamba.mistral_inference.args import MambaArgs
from models.mamba.mistral_inference.model import ModelBase

from collections import namedtuple

_is_mamba_installed = False
try:
    from mamba_ssm.models.config_mamba import MambaConfig
    from models.mamba.models.mixer_seq_simple_EE import MambaLMHeadModel

    _is_mamba_installed = True
except ImportError:
    _is_mamba_installed = False

try:
    from models.mamba.ops.triton.layer_norm import RMSNorm, layer_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn = None, None


class Mamba(ModelBase, nn.Module):
    def __init__(self, args: MambaArgs):
        super().__init__()
        self.args = args
        assert _is_mamba_installed, "Mamba is not installed. Please install it using `pip install mamba-ssm`."

        # make sure naming is consistent with `mamba_ssm`
        config = MambaConfig(
            d_model=args.dim,
            n_layer=args.n_layers,
            vocab_size=args.vocab_size,
            ssm_cfg={"ngroups": args.n_groups, "layer": "Mamba2"},
            attn_layer_idx=[],
            attn_cfg={},
            rms_norm=args.rms_norm,
            residual_in_fp32=args.residual_in_fp32,
            fused_add_norm=args.fused_add_norm,
            pad_vocab_size_multiple=args.pad_vocab_size_multiple,
            tie_embeddings=args.tie_embeddings,
        )

        config.ee_pos = args.ee_pos

        self.th = torch.ones(len(args.ee_pos))

        self.model = MambaLMHeadModel(config)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
    
    def apply_layer_norm_fn(self, hidden_states, residual):

        hidden_states = layer_norm_fn(
                hidden_states,
                self.model.backbone.norm_f.weight,
                self.model.backbone.norm_f.bias,
                eps=self.model.backbone.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.model.backbone.residual_in_fp32,
                is_rms_norm=isinstance(self.model.backbone.norm_f, RMSNorm)
            )
        
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int] = None,  # not supported for now
        cache = None,  # not supported for now
        train_EE = False,  # not supported for now
        num_last_tokens=0,
    ) -> torch.Tensor:
        
        
        if train_EE:
            k = self.k
            exits = torch.zeros((1, self.args.block_size, len(self.args.ee_pos), 2), device = input_ids.device)
            early_exits_topk = torch.zeros((1, self.args.block_size, len(self.args.ee_pos), k), device = input_ids.device)


        hidden_states = self.model.backbone.embedding(input_ids)
        residual = None
        ee_index = 0
        for i, layer in enumerate(self.model.backbone.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )

            # little patch
            if len(hidden_states.shape) != len(residual.shape):
                residual.unsqueeze_(0)

            if train_EE and i in self.args.ee_pos:
                input_ee = torch.cat((hidden_states, residual), dim = 2)
                shape_ee = input_ee.shape
                exits[0, :shape_ee[1], ee_index] = self.model.backbone.ee[ee_index].forward(input_ee)
                early_exits_topk[0, :shape_ee[1], ee_index] = torch.topk(self.model.lm_head(self.apply_layer_norm_fn(hidden_states.detach(), residual.detach())).float(), k, dim = 2)[1]
                ee_index += 1


        if not self.model.backbone.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            # hidden_states = layer_norm_fn(
            #     hidden_states,
            #     self.model.backbone.norm_f.weight,
            #     self.model.backbone.norm_f.bias,
            #     eps=self.model.backbone.norm_f.eps,
            #     residual=residual,
            #     prenorm=False,
            #     residual_in_fp32=self.model.backbone.residual_in_fp32,
            #     is_rms_norm=isinstance(self.model.backbone.norm_f, RMSNorm)
            # )

            hidden_states = self.apply_layer_norm_fn(hidden_states, residual)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states)
        
        if train_EE:
            
            targets_EE = torch.argmax(lm_logits.detach(), dim = 2).view(1, shape_ee[1], 1).repeat(1,1, len(self.args.ee_pos))
            
            # diff = (targets_EE == torch.argmax(early_exits_topk[:, :shape_ee[1], :], dim = 3)).to(torch.long)

            diff = (targets_EE.unsqueeze(3).repeat(1,1,1,k) == early_exits_topk[:, :shape_ee[1], :]).any(3).to(torch.long)

            ratio = torch.sum(diff)/torch.prod(torch.tensor(diff.shape))

            # ratios = [torch.sum(diff[:,:,i])/torch.prod(torch.tensor(diff[:,:,i].shape)) for i in range(self.args.n_layers)]

            # single loss
            # loss = F.cross_entropy(
            #                         exits[:shape_ee[0],:shape_ee[1]].view(shape_ee[0] * shape_ee[1] * self.config.n_layer, 2),
            #                         diff.view(-1)
            #                         )
            
            
            # if loss is torch.nan:
            #     print("sadge")
            

            weighted_loss = torch.ones(len(self.args.ee_pos))/len(self.args.ee_pos)
            loss = 0
            losses = []
            ee_index = 0
            for i in range(len(self.args.ee_pos)):
                loss_p = weighted_loss[i] * torch.nn.functional.cross_entropy(
                                    exits[0, :shape_ee[1], ee_index].view(shape_ee[1], 2),
                                    diff[0, :, ee_index].view(-1)
                                    )
                losses.append(loss_p)
                # loss += loss_p
                ee_index += 1

            # print(losses)
            
            # clas weighted single loss
            # loss = F.cross_entropy(
            #                     exits[:shape_ee[0],:shape_ee[1]].view(shape_ee[0] * shape_ee[1] * self.config.n_layer, 2),
            #                     diff.view(-1),
            #                     weight = torch.tensor([ratio, 1 - ratio], device = device, dtype = torch.float)
            #                     )
            

            # compute accuracy between diff and exits
            acc = (diff == torch.argmax(exits[0,:shape_ee[1]], dim = 2)).to(torch.float).mean()
            
            predicted_labels = torch.argmax(exits[0, :shape_ee[1]], dim = 2)

            true_labels = diff.view(-1)
            predicted_labels = predicted_labels.view(-1)

            # True Positives
            TP = ((true_labels == 1) & (predicted_labels == 1)).sum().item()

            # False Positives
            FP = ((true_labels == 0) & (predicted_labels == 1)).sum().item()

            # False Negatives
            FN = ((true_labels == 1) & (predicted_labels == 0)).sum().item()


            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics = torch.tensor([acc.item(), recall, precision, f1, ratio])

            return lm_logits, losses, metrics

        else:
                   
            return lm_logits, None, None

    def forward_inference(
        self,
        input_ids: torch.Tensor,
        seqlens: List[int] = None,  # not supported for now
        cache = None,  # not supported for now
        num_last_tokens=0,
        use_EE = False,
    ) -> torch.Tensor:

        hidden_states = self.model.backbone.embedding(input_ids)
        residual = None
        ee_index = 0
        for i, layer in enumerate(self.model.backbone.layers):
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )

            # little patch
            if len(hidden_states.shape) != len(residual.shape):
                residual.unsqueeze_(0)

            if use_EE and i in self.args.ee_pos:
                input_ee = torch.cat((hidden_states, residual), dim = 2)
                shape_ee = input_ee.shape
                ee = self.model.backbone.ee[ee_index].forward(input_ee)
                ee_index += 1

                if ee[0] > self.th[i]:
                    print("Early exit at layer:", i + 1)
                    break          

        hidden_states = self.apply_layer_norm_fn(hidden_states, residual)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.model.lm_head(hidden_states) 

       

        for i, layer in enumerate(self.layers):

            h, ee = layer.forward_inference(h, freqs_cis, use_EE = use_EE)

            if use_EE:

                ee = torch.nn.functional.softmax(ee[0], dim = 0)
                ind = torch.argmax(ee)

                if ind == 1 and i < self.args.n_layers - 1:

                    if ee[ind] > self.th[i]:
                        print("Early exit at layer:", i + 1, " position:", sum(seqlens))
                        
                        if not hasattr(self, 'exits_done'):
                            self.exits_done = []

                        self.exits_done.append(i + 1)

                        # propagate intermediate states
                        for j in range(i+1, self.args.n_layers):

                            norm_x = self.layers[j].attention_norm(h)
                            k = self.layers[j].attention.wk(norm_x)
                            v = self.layers[j].attention.wv(norm_x)

                            # self.transformer.h[j].attn.k[:,pos] = self.transformer.h[i].attn.k[:,pos]
                            # self.transformer.h[j].attn.v[:,pos] = self.transformer.h[i].attn.v[:,pos]

                            self.layers[j].attention.k[sum(seqlens) - 1] = k.view(self.args.n_kv_heads, self.args.head_dim)
                            self.layers[j].attention.v[sum(seqlens) - 1] = v.view(self.args.n_kv_heads, self.args.head_dim)

                            # self.intermediate_states[0, j+1, pos - 1] = x.detach()
                            # self.intermediate_states[0, j+1, pos - 1] = x.detach()
                            
                        h, _ = self.layers[-1].forward_inference(h, freqs_cis, use_EE = False)                        

                        break
                
            self.intermediate_states[i+1, sum(seqlens) - 1] = h.detach()

        return self.output(self.norm(h)).float()

            
        # if use_EE and i in self.args.ee_pos:
        #         input_ee = torch.cat((hidden_states, residual), dim = 2)
        #         shape_ee = input_ee.shape
        #         ee = self.model.backbone.ee[ee_index].forward(input_ee)
                
        #         if ee 
        #         ee_index += 1
        
    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        
    
    def from_folder(
        self,
        folder: Union[Path, str],
        max_batch_size: int = 1,
        num_pipeline_ranks: int = 1,
        device: Union[torch.device, str] = "cuda",
        dtype: Optional[torch.dtype] = None,
    ) -> "Mamba":
        # with open(Path(folder) / "params.json", "r") as f:
        #     model_args = MambaArgs.from_dict(json.load(f))

        # with torch.device("meta"):
        #     model = Mamba(model_args)

        model_file = Path(folder) / "consolidated.safetensors"

        assert model_file.exists(), f"Make sure {model_file} exists."
        loaded = safetensors.torch.load_file(str(model_file))

        # strict false to allow loading of partial models (EE's)
        self.load_state_dict(loaded, assign=True, strict=False)

        # self.to(device=device, dtype=dtype)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_EE = False):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # forward the model to get the logits for the index in the sequence
            logits = self(idx, use_EE = use_EE)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[0, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[[-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next[None,:]), dim = 1)

        return idx

    def freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if 'ee' not in name or 'feed' in name:
                param.requires_grad = False