import copy
import logging
import os
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights,tokenize
import torch
import torch.nn as nn
from .grab import TexualEmbeddingLayer, VisualEmbeddingLayer
from .prototype import PrototypeFusion, VisualPrototypeEnrichment, VisualPrototypeModule
from .prototype.losses import compute_diversity_loss
from torch.cuda.amp import autocast


logger = logging.getLogger(__name__)


def _filter_keys_by_prefix(keys, ignored_prefixes=None):
    if not ignored_prefixes:
        return list(keys)
    return [
        key for key in keys
        if not any(key.startswith(prefix) for prefix in ignored_prefixes)
    ]


def _load_matching_state_dict(module, loaded_state_dict, ignored_target_prefixes=None):
    current_state_dict = module.state_dict()
    allowed_target_keys = _filter_keys_by_prefix(current_state_dict.keys(), ignored_target_prefixes)
    filtered_state_dict = {}
    skipped_source_keys = []
    shape_mismatched_keys = []

    for key, value in loaded_state_dict.items():
        if key not in current_state_dict:
            skipped_source_keys.append(key)
            continue
        if current_state_dict[key].shape != value.shape:
            shape_mismatched_keys.append(key)
            continue
        filtered_state_dict[key] = value

    current_state_dict.update(filtered_state_dict)
    module.load_state_dict(current_state_dict, strict=True)

    return {
        "loaded_keys": sorted(filtered_state_dict.keys()),
        "skipped_source_keys": sorted(skipped_source_keys),
        "shape_mismatched_keys": sorted(shape_mismatched_keys),
        "target_keys_kept": sorted(set(allowed_target_keys) - set(filtered_state_dict.keys())),
    }


def _log_checkpoint_load_status(component_name, path, source_name, load_stats):
    logger.info(
        "Loaded %s checkpoint from %s using `%s` (%d tensors loaded, %d current tensors kept, %d checkpoint tensors skipped, %d shape mismatches)",
        component_name,
        path,
        source_name,
        len(load_stats["loaded_keys"]),
        len(load_stats["target_keys_kept"]),
        len(load_stats["skipped_source_keys"]),
        len(load_stats["shape_mismatched_keys"]),
    )


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def build_prototype_module(args, embed_dim):
    """
    Phase-2 scaffolding only: constructor hook for the prototype module.
    This path is intentionally not connected to model forward behavior yet.
    """
    prototype_total_steps = getattr(args, 'prototype_total_steps', 10 * 145)
    use_parameter_free_self_attention = getattr(args, 'use_parameter_free_self_attention', True)
    return VisualPrototypeModule(
        num_prototypes=getattr(args, 'num_prototypes', 64),
        embed_dim=embed_dim,
        tau_init=getattr(args, 'prototype_tau_init', 1.0),
        tau_min=getattr(args, 'prototype_tau_min', 0.05),
        total_steps=prototype_total_steps,
        prototype_init=getattr(args, 'prototype_init', 'random'),
        use_parameter_free_self_attention=use_parameter_free_self_attention,
        infer_hard_query=getattr(args, 'infer_hard_query', False),
    )

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        # ipdb.set_trace()
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        outputs = self.transformer([x])
        x = outputs[0]
        att = outputs[1]
        x = x.permute(1, 0, 2)  # LND -> NLD   # x,att
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_feature = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return text_feature



class ITSELF(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']
        self.grab_embed_dim = 4096
        self.args = args
        if 'cid' in args.loss_names:
            self.num_classes = num_classes + 1
            self.classifier_global = nn.Linear(self.embed_dim , self.num_classes)
            nn.init.normal_(self.classifier_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_global.bias.data, val=0.0)
            self.mlp_global = nn.Sequential(nn.Linear(2 * self.embed_dim, self.embed_dim),nn.LayerNorm(self.embed_dim),nn.GELU())
            self.classifier_id_global = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier_id_global.weight.data, std=0.001)
            nn.init.constant_(self.classifier_id_global.bias.data, val=0.0)
            if not args.only_global:
                self.classifier_grab = nn.Linear(self.grab_embed_dim, self.num_classes)
                nn.init.normal_(self.classifier_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_grab.bias.data, val=0.0)
                self.mlp_grab = nn.Sequential(nn.Linear(2 * self.grab_embed_dim, self.grab_embed_dim),nn.LayerNorm(self.grab_embed_dim),nn.GELU())
                self.classifier_id_grab = nn.Linear(self.grab_embed_dim, self.num_classes)
                nn.init.normal_(self.classifier_id_grab.weight.data, std=0.001)
                nn.init.constant_(self.classifier_id_grab.bias.data, val=0.0)
                self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
                self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        self.use_prototype = getattr(args, 'use_prototype', False)
        self.use_div_loss = getattr(args, 'use_div_loss', False)
        self.div_loss_weight = getattr(args, 'div_loss_weight', 1.0)
        self.prototype_enrich_side = getattr(args, 'prototype_enrich_side', 'text').lower()
        self.prototype_precision = getattr(args, 'prototype_precision', 'fp32').lower()
        if self.use_prototype:
            self.prototype_module = build_prototype_module(args, self.embed_dim)
            if self.prototype_enrich_side in {"text", "both"}:
                self.text_prototype_fusion = PrototypeFusion(self.embed_dim)
            if self.prototype_enrich_side in {"vision", "both"}:
                self.vision_prototype_enrichment = VisualPrototypeEnrichment(self.embed_dim)

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 

    def count_parameters(self, trainable_only=False, module_name=None):
        if module_name is None:
            parameters = self.parameters()
        else:
            module = getattr(self, module_name, None)
            if module is None:
                return 0
            parameters = module.parameters()

        if trainable_only:
            return sum(p.numel() for p in parameters if p.requires_grad)
        return sum(p.numel() for p in parameters)

    def count_backbone_parameters(self, trainable_only=False):
        if trainable_only:
            return sum(
                parameter.numel()
                for name, parameter in self.named_parameters()
                if not name.startswith("prototype_module.") and parameter.requires_grad
            )
        return sum(
            parameter.numel()
            for name, parameter in self.named_parameters()
            if not name.startswith("prototype_module.")
        )

    def get_parameter_summary(self):
        total = self.count_parameters(trainable_only=False)
        trainable = self.count_parameters(trainable_only=True)
        backbone_total = self.count_backbone_parameters(trainable_only=False)
        backbone_trainable = self.count_backbone_parameters(trainable_only=True)
        prototype_total = self.count_parameters(trainable_only=False, module_name="prototype_module")
        prototype_trainable = self.count_parameters(trainable_only=True, module_name="prototype_module")
        summary = {
            "total": total,
            "trainable": trainable,
            "backbone_total": backbone_total,
            "backbone_trainable": backbone_trainable,
            "prototype_total": prototype_total,
            "prototype_trainable": prototype_trainable,
            "other_total": total - backbone_total - prototype_total,
            "other_trainable": trainable - backbone_trainable - prototype_trainable,
        }
        return summary

    def get_backbone_parameter_breakdown(self):
        breakdown = {}
        for module_name, module in self.named_children():
            if module_name == "prototype_module":
                continue
            total = sum(parameter.numel() for parameter in module.parameters())
            trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
            if total == 0 and trainable == 0:
                continue
            breakdown[module_name] = {
                "total": total,
                "trainable": trainable,
            }
        return breakdown

    @staticmethod
    def _set_requires_grad(module, requires_grad):
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def freeze_backbone(self):
        for module_name, module in self.named_children():
            if module_name == "prototype_module":
                continue
            self._set_requires_grad(module, False)

    def unfreeze_backbone(self):
        for module_name, module in self.named_children():
            if module_name == "prototype_module":
                continue
            self._set_requires_grad(module, True)

    def freeze_prototype(self):
        if self.use_prototype:
            self._set_requires_grad(self.prototype_module, False)

    def unfreeze_prototype(self):
        if self.use_prototype:
            self._set_requires_grad(self.prototype_module, True)

    def export_prototype_state(self):
        if not self.use_prototype:
            raise RuntimeError("Prototype module is disabled; nothing to export.")

        return {
            "num_prototypes": self.prototype_module.num_prototypes,
            "embed_dim": self.prototype_module.embed_dim,
            "prototype_state_dict": self.prototype_module.state_dict(),
        }

    def save_prototype_checkpoint(self, path, extra=None):
        if not self.use_prototype:
            raise RuntimeError("Prototype module is disabled; cannot save prototype checkpoint.")

        checkpoint = self.export_prototype_state()
        if extra:
            checkpoint.update(extra)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(checkpoint, path)

    def export_backbone_state(self):
        return {
            "backbone_state_dict": {
                key: value
                for key, value in self.state_dict().items()
                if not key.startswith("prototype_module.")
            }
        }

    def save_backbone_checkpoint(self, path, extra=None):
        checkpoint = self.export_backbone_state()
        if extra:
            checkpoint.update(extra)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(checkpoint, path)

    def load_prototype_checkpoint(self, path, strict=True, map_location="cpu"):
        if not self.use_prototype:
            raise RuntimeError("Prototype module is disabled; cannot load prototype checkpoint.")

        checkpoint = torch.load(path, map_location=map_location)
        prototype_state_dict = None
        source_name = None

        if isinstance(checkpoint, dict):
            if "prototype_state_dict" in checkpoint:
                prototype_state_dict = checkpoint["prototype_state_dict"]
                source_name = "prototype_state_dict"
            elif "prototype_module" in checkpoint and isinstance(checkpoint["prototype_module"], dict):
                prototype_state_dict = checkpoint["prototype_module"]
                source_name = "prototype_module"
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                prototype_state_dict = {
                    key[len("prototype_module."):]: value
                    for key, value in checkpoint["model"].items()
                    if key.startswith("prototype_module.")
                }
                source_name = "model.prototype_module"
            elif all(isinstance(key, str) for key in checkpoint.keys()):
                prototype_state_dict = checkpoint
                source_name = "raw_state_dict"

        if not prototype_state_dict:
            raise KeyError(f"No prototype weights found in checkpoint: {path}")

        load_stats = _load_matching_state_dict(self.prototype_module, prototype_state_dict)
        _log_checkpoint_load_status("prototype", path, source_name or "unknown", load_stats)
        if strict and (
            load_stats["target_keys_kept"]
            or load_stats["skipped_source_keys"]
            or load_stats["shape_mismatched_keys"]
        ):
            raise RuntimeError(
                f"Error(s) in loading prototype checkpoint for {path}: "
                f"kept={load_stats['target_keys_kept']}, "
                f"skipped={load_stats['skipped_source_keys']}, "
                f"shape_mismatched={load_stats['shape_mismatched_keys']}"
            )
        return checkpoint

    def load_backbone_checkpoint(self, path, strict=True, map_location="cpu"):
        checkpoint = torch.load(path, map_location=map_location)
        backbone_state_dict = None
        source_name = None

        if isinstance(checkpoint, dict):
            if "backbone_state_dict" in checkpoint:
                backbone_state_dict = checkpoint["backbone_state_dict"]
                source_name = "backbone_state_dict"
            elif "base_model" in checkpoint and isinstance(checkpoint["base_model"], dict):
                backbone_state_dict = {
                    f"base_model.{key}": value
                    for key, value in checkpoint["base_model"].items()
                }
                source_name = "base_model (legacy backbone-only)"
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                backbone_state_dict = {
                    key: value
                    for key, value in checkpoint["model"].items()
                    if not key.startswith("prototype_module.")
                }
                source_name = "model (all non-prototype modules)"
            elif all(isinstance(key, str) for key in checkpoint.keys()):
                backbone_state_dict = checkpoint
                source_name = "raw_state_dict"

        if not backbone_state_dict:
            raise KeyError(f"No backbone weights found in checkpoint: {path}")

        load_stats = _load_matching_state_dict(self, backbone_state_dict, ignored_target_prefixes=["prototype_module."])
        _log_checkpoint_load_status("backbone", path, source_name or "unknown", load_stats)
        if strict and (
            load_stats["target_keys_kept"]
            or load_stats["skipped_source_keys"]
            or load_stats["shape_mismatched_keys"]
        ):
            raise RuntimeError(
                f"Error(s) in loading backbone checkpoint for {path}: "
                f"kept={load_stats['target_keys_kept']}, "
                f"skipped={load_stats['skipped_source_keys']}, "
                f"shape_mismatched={load_stats['shape_mismatched_keys']}"
            )
        return checkpoint

    def apply_prototype(self, t_feats, i_feats, training=True, current_step=None, return_stats=False):
        if not self.use_prototype:
            if return_stats:
                return t_feats, i_feats, None
            return t_feats, i_feats

        if self.prototype_precision == 'fp16':
            prototype_dtype = torch.float16
        else:
            prototype_dtype = torch.float32

        i_context = i_feats.to(dtype=prototype_dtype)
        prototype_result = self.prototype_module(
            visual_context=i_context,
            training=training,
            current_step=current_step,
            return_stats=return_stats,
        )
        if return_stats:
            prototype_query, prototype_stats = prototype_result
        else:
            prototype_query = prototype_result
            prototype_stats = None
        prototype_query = prototype_query.to(dtype=prototype_dtype)
        t_feats_enriched = t_feats
        i_feats_enriched = i_feats

        if self.prototype_enrich_side in {"text", "both"}:
            t_feats_enriched = self.text_prototype_fusion(
                t_feats,
                prototype_query,
            )
        if self.prototype_enrich_side in {"vision", "both"}:
            i_feats_enriched = self.vision_prototype_enrichment(
                i_feats,
                prototype_query,
            )

        if return_stats:
            return t_feats_enriched, i_feats_enriched, prototype_stats
        return t_feats_enriched, i_feats_enriched

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_grab(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_grab_f = self.visul_emb_layer(x, atten_i)
        return i_grab_f.float()

    def encode_text_grab(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_grab_f = self.texual_emb_layer(x, text, atten_t)
        return t_grab_f.float()
    
    def rollout(self, attentions: torch.Tensor, 
                head_fusion = 'mean', 
                discard: bool = True,
                discard_ratios: list = [0.25, 1., 1., 1., 0.25, 0.25, 1., 1., 1., 1., 0.25, 0.25], 
                start_layer: int = 4, 
                skip_layer: list = [5,6,7,8,9,10]):
        
        if len(attentions.shape) == 5:
            L, B, _, N, _ = attentions.shape
        else:
            L, B, N, _ = attentions.shape
        device = attentions.device
        result = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)  # [B, N, N]
                    
        for layer in range(start_layer, L):
            if layer in skip_layer:
                continue
            attn = attentions[layer]  
            # have H shape (L, B, H, N, N)
            if len(attentions.shape) == 5:
                with torch.no_grad():
                    if head_fusion == "mean":
                        attn = attn.mean(axis=1) # [B, H, N, N] --> axis == 1
                    elif head_fusion == "max":
                        attn = attn.max(axis=1)[0]
                    elif head_fusion == "min":
                        attn = attn.min(axis=1)[0]
                    else:
                        raise "Attention head fusion type Not supported"
            
            if discard:
                discard_ratio = discard_ratios[layer]
                flat = attn.view(B, -1)  # [B, N*N]
                num_to_discard = int(flat.size(-1) * discard_ratio)

                if num_to_discard > 0:
                    _, indices = flat.topk(num_to_discard, dim=-1, largest=False)
                    for b in range(B):
                        idx = indices[b]
                        idx = idx[idx != 0]
                        flat[b, idx] = 0
                    attn = flat.view(B, N, N)

            I = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
            attn = (attn + I) / 2.0
            attn = attn / attn.sum(dim=-1, keepdim=True)
            result = torch.bmm(attn, result)

        return result  # [B, N, N]

    def forward(self, batch, epoch=None, current_step=None, return_prototype_stats=False):
        ret = dict()
        device = "cuda"

        if 'cid' in self.current_task:
            self.mlp_global = self.mlp_global.float()
            self.classifier_global = self.classifier_global.float()
            if not self.args.only_global:
                self.mlp_grab = self.mlp_grab.float()
                self.classifier_grab = self.classifier_grab.float()
        
        ret.update({'temperature': 1 / self.logit_scale})
        images = batch['images']
        caption_ids = batch['caption_ids']
        
        if self.args.return_all:
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids, return_all=True, average_attn_weights = self.args.average_attn_weights)
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            prototype_result = self.apply_prototype(
                t_feats=t_feats,
                i_feats=i_feats,
                training=self.training,
                current_step=current_step,
                return_stats=return_prototype_stats,
            )
            if return_prototype_stats:
                t_feats, i_feats, prototype_stats = prototype_result
                ret["prototype_stats"] = prototype_stats
            else:
                t_feats, i_feats = prototype_result
            if self.args.topk_type == 'mean':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = atten_i.mean(0)
                atten_t = atten_t.mean(0) 
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'std':
                atten_i = torch.stack(atten_i, dim=0)
                atten_t = torch.stack(atten_t, dim=0)
                atten_i = atten_i.std(0, unbiased=False)
                atten_t = atten_t.std(0, unbiased=False)
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'layer_index' and self.args.layer_index is not None:
                # layer_index from 0 to 11 (12 layers)
                atten_i = atten_i[self.args.layer_index]
                atten_t = atten_t[self.args.layer_index]
                if current_step is not None:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                else:
                    i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                    t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
            elif self.args.topk_type == 'custom':
                atten_i = torch.stack(atten_i, dim=0)  # [L, B, N, N]
                atten_t = torch.stack(atten_t, dim=0)  # [L, B, N, N]

                atten_i = self.rollout(atten_i)
                atten_t = self.rollout(atten_t)
                if not self.args.only_global:
                    if current_step is not None:
                        i_grab_f = self.visul_emb_layer(image_feats, atten_i, current_step)
                        t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t, current_step)
                    else:
                        i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                        t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        else:
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            # i_feats = image_feats.float() # for CLIP ResNet visual model
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            prototype_result = self.apply_prototype(
                t_feats=t_feats,
                i_feats=i_feats,
                training=self.training,
                current_step=current_step,
                return_stats=return_prototype_stats,
            )
            if return_prototype_stats:
                t_feats, i_feats, prototype_stats = prototype_result
                ret["prototype_stats"] = prototype_stats
            else:
                t_feats, i_feats = prototype_result
            if not self.args.only_global:
                i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        if 'cid' in self.current_task:
            S = objectives.cosine_similarity_matrix(i_feats, t_feats)
            hard_negatives = objectives.sample_hard_negatives(S, batch['pids'])
            M = batch['pids'].max().item()
            new_labels = objectives.update_labels_for_negatives(batch['pids'], hard_negatives, M)
            all_pairs = objectives.create_sample_pairs(i_feats, t_feats, hard_negatives, new_labels, batch['pids'])
            ni_feats, nt_feats, nlabels = all_pairs
            z_feats1 = torch.cat([ni_feats.float(), nt_feats.float()], dim=1)
            z_feats2 = torch.cat([nt_feats.float(), ni_feats.float()], dim=1)
            z_feats1 = self.mlp_global(z_feats1.float())
            z_feats2 = self.mlp_global(z_feats2.float())
            cross_modal_logits1 = self.classifier_global(z_feats1.float())
            cross_modal_logits2 = self.classifier_global(z_feats2.float())
            device = cross_modal_logits1.device 
            nlabels = nlabels.to(device) 
            closs1 =  objectives.compute_cid(cross_modal_logits1, cross_modal_logits2,nlabels)
            image_logits = self.classifier_id_global(i_feats.half()).float()
            text_logits = self.classifier_id_global(t_feats.half()).float()
            closs3 = objectives.compute_id(image_logits, batch['pids']) + objectives.compute_id(text_logits, batch['pids'])
            
            if not self.args.only_global:
                S_ = objectives.cosine_similarity_matrix(i_grab_f, t_grab_f)
                hard_negatives_ = objectives.sample_hard_negatives(S_, batch['pids'])
                M_ = batch['pids'].max().item()
                new_labels_ = objectives.update_labels_for_negatives(batch['pids'], hard_negatives_, M_)
                all_pairs_ = objectives.create_sample_pairs(i_grab_f, t_grab_f, hard_negatives_, new_labels_, batch['pids'])
                ni_feats_, nt_feats_, nlabels_ = all_pairs_
                z_feats1_ = torch.cat([ni_feats_.float(), nt_feats_.float()], dim=1)
                z_feats2_ = torch.cat([nt_feats_.float(), ni_feats_.float()], dim=1)
                z_feats1_ = self.mlp_grab(z_feats1_.float())
                z_feats2_ = self.mlp_grab(z_feats2_.float())
                cross_modal_logits1_ = self.classifier_grab(z_feats1_.float())
                cross_modal_logits2_ = self.classifier_grab(z_feats2_.float())
                nlabels_ = nlabels_.to(device)
                closs2 =  objectives.compute_cid(cross_modal_logits1_, cross_modal_logits2_,nlabels_)
                image_logits_ = self.classifier_id_grab(i_grab_f.half()).float()
                text_logits_ = self.classifier_id_grab(t_grab_f.half()).float()
                closs4 = objectives.compute_id(image_logits_, batch['pids']) + objectives.compute_id(text_logits_, batch['pids'])
                ret.update({'cid_loss': closs1+closs2+closs3+closs4})
            else:
                ret.update({'cid_loss': closs1+closs3})

        if 'tal' in self.current_task:
            TAL_global_loss = objectives.compute_TAL(i_feats, t_feats,batch['pids'],margin=self.args.margin,tau=self.args.tau)
            if not self.args.only_global:
                TAL_grab_loss = objectives.compute_TAL(i_grab_f, t_grab_f,batch['pids'],margin=self.args.margin,tau=self.args.tau)
                ret.update({'tal_loss': TAL_global_loss + TAL_grab_loss}) 
            else:
                ret.update({'tal_loss': TAL_global_loss})

        if self.use_div_loss and self.use_prototype:
            ret.update({'div_loss': self.div_loss_weight * compute_diversity_loss(self.prototype_module._compute_query_group())})

        return ret

def build_model(args, num_classes=11003):
    model = ITSELF(args, num_classes)
    if getattr(args, "freeze_backbone", False):
        model.freeze_backbone()
    if getattr(args, "freeze_prototype", False):
        model.freeze_prototype()
    convert_weights(model)

    return model
