import copy
from model import objectives
from .clip_model import Transformer, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights,tokenize
import torch
import torch.nn as nn
from .grab import TexualEmbeddingLayer, VisualEmbeddingLayer
from .enrichment import TargetPrototypeEnricher, build_part_prototypes
from torch.cuda.amp import autocast


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


def freeze_host_parameters(model, trainable_prefix="target_enricher."):
    frozen_params = 0
    trainable_params = 0
    for name, parameter in model.named_parameters():
        keep_trainable = name.startswith(trainable_prefix)
        parameter.requires_grad = keep_trainable
        if keep_trainable:
            trainable_params += parameter.numel()
        else:
            frozen_params += parameter.numel()
    if trainable_params == 0:
        raise ValueError("--freeze_host requires target enrichment parameters to train")
    return frozen_params, trainable_params


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
        if getattr(args, "target_enrichment", False) and args.enrichment_space == "grab" and args.only_global:
            raise ValueError("--enrichment_space grab requires GRAB features; remove --only_global")
        if (
            getattr(args, "target_enrichment", False)
            and getattr(args, "topm_rank_space", "host_global") == "hybrid_global_grab"
            and args.only_global
        ):
            raise ValueError("--topm_rank_space hybrid_global_grab requires GRAB features; remove --only_global")
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

        if not args.only_global and not hasattr(self, "visul_emb_layer"):
            self.visul_emb_layer = VisualEmbeddingLayer(ratio=args.select_ratio)
            self.texual_emb_layer = TexualEmbeddingLayer(ratio=args.select_ratio)

        if getattr(args, "target_enrichment", False):
            self.target_enricher = TargetPrototypeEnricher(self.embed_dim, self.grab_embed_dim, args)

        self.logit_scale = torch.ones([]) * (1 / args.temperature)
        self.freeze_host_stats = None
        if getattr(args, "freeze_host", False):
            self.freeze_host_stats = freeze_host_parameters(self)

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

    def encode_target_image_cache(self, image, cache_prototypes=True):
        image_feats, atten_i = self.base_model.encode_image(image)
        host_features = image_feats[:, 0, :].float()
        cache = {"host_image_features": host_features}
        if not cache_prototypes:
            return cache

        grid_size = None
        if hasattr(self.base_model.visual, "num_y") and hasattr(self.base_model.visual, "num_x"):
            grid_size = (self.base_model.visual.num_y, self.base_model.visual.num_x)
        cache["prototypes"] = build_part_prototypes(
            image_feats,
            getattr(self.args, "num_parts", 6),
            grid_size=grid_size,
            mode=getattr(self.args, "extractor_mode", "global,horizontal"),
        )
        needs_grab_rank = getattr(self.args, "topm_rank_space", "host_global") == "hybrid_global_grab"
        grab_features = None
        if getattr(self.args, "enrichment_space", "global") == "grab" or needs_grab_rank:
            grab_features = self.visul_emb_layer(image_feats, atten_i).float()
        if getattr(self.args, "enrichment_space", "global") == "grab":
            cache["retrieval_features"] = grab_features
        else:
            cache["retrieval_features"] = host_features
        if needs_grab_rank:
            cache["grab_image_features"] = grab_features
        return cache

    def enrich_text_features(self, query_features, host_text_features, target_cache, grab_text_features=None):
        self.target_enricher = self.target_enricher.float()
        return self.target_enricher.enrich_only(
            query_features=query_features,
            host_text_features=host_text_features,
            pool_cache=target_cache,
            space=getattr(self.args, "enrichment_space", "global"),
            grab_text_features=grab_text_features,
        )
    
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

    def forward(self, batch, epoch=None, current_step=None, target_cache=None):
        ret = dict()
        device = "cuda"
        use_host_loss = getattr(self.args, "use_host_loss", True)

        if use_host_loss and 'cid' in self.current_task:
            self.mlp_global = self.mlp_global.float()
            self.classifier_global = self.classifier_global.float()
            if not self.args.only_global:
                self.mlp_grab = self.mlp_grab.float()
                self.classifier_grab = self.classifier_grab.float()
        
        ret.update({'temperature': 1 / self.logit_scale})
        caption_ids = batch['caption_ids']
        pnp_text_only = getattr(self.args, "pnp_text_only", False)

        if pnp_text_only:
            text_feats, _ = self.base_model.encode_text(caption_ids.long())
            t_feats = text_feats[
                torch.arange(text_feats.shape[0], device=text_feats.device),
                caption_ids.argmax(dim=-1),
            ].float()
        elif self.args.return_all:
            images = batch['images']
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids, return_all=True, average_attn_weights = self.args.average_attn_weights)
            i_feats = image_feats[:, 0, :].float()
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
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
            images = batch['images']
            image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
            i_feats = image_feats[:, 0, :].float()
            # i_feats = image_feats.float() # for CLIP ResNet visual model
            t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
            if not self.args.only_global:
                i_grab_f = self.visul_emb_layer(image_feats, atten_i)
                t_grab_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)

        zero_source = t_feats if pnp_text_only else i_feats

        if getattr(self.args, "target_enrichment", False) and target_cache is not None:
            self.target_enricher = self.target_enricher.float()
            if self.args.enrichment_space == "grab":
                target_ret = self.target_enricher(
                    query_features=t_grab_f,
                    host_text_features=t_feats,
                    grab_text_features=t_grab_f,
                    query_pids=batch["pids"],
                    pool_cache=target_cache,
                    space="grab",
                )
                t_grab_f = target_ret["enriched_features"]
            else:
                grab_text_features = None
                if getattr(self.args, "topm_rank_space", "host_global") == "hybrid_global_grab":
                    grab_text_features = t_grab_f
                target_ret = self.target_enricher(
                    query_features=t_feats,
                    host_text_features=t_feats,
                    grab_text_features=grab_text_features,
                    query_pids=batch["pids"],
                    pool_cache=target_cache,
                    space="global",
                )
                t_feats = target_ret["enriched_features"]
            target_metrics = {
                "target_enrichment_loss": target_ret["total_loss"],
            }
            loss_grad_sources = {
                "target_enrichment_loss": target_ret["total_loss"],
            }
            if self.target_enricher.use_target_retrieval_loss:
                target_metrics["target_retrieval_loss"] = target_ret["target_retrieval_loss"].detach()
                loss_grad_sources["target_retrieval_loss"] = target_ret["target_retrieval_loss"]
            if self.target_enricher.use_target_robust_loss:
                target_metrics.update({
                    "target_robust_loss": target_ret["robust_loss"].detach(),
                    "target_guard_loss": target_ret["guard_loss"].detach(),
                    "target_gain_loss": target_ret["gain_loss"].detach(),
                })
                loss_grad_sources.update({
                    "target_robust_loss": target_ret["robust_loss"],
                    "target_guard_loss": target_ret["guard_loss"],
                    "target_gain_loss": target_ret["gain_loss"],
                })
            for metric_key, metric_value in target_ret.items():
                if not (metric_key.startswith("target_") or metric_key.startswith("mixer/")):
                    continue
                if torch.is_tensor(metric_value) and metric_value.numel() == 1:
                    target_metrics[metric_key] = metric_value.detach()
            target_metrics["_loss_grad_sources"] = loss_grad_sources
            ret.update(target_metrics)

        if use_host_loss and 'cid' in self.current_task:
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

        if use_host_loss and 'tal' in self.current_task:
            TAL_global_loss = objectives.compute_TAL(i_feats, t_feats,batch['pids'],margin=self.args.margin,tau=self.args.tau)
            if not self.args.only_global:
                TAL_grab_loss = objectives.compute_TAL(i_grab_f, t_grab_f,batch['pids'],margin=self.args.margin,tau=self.args.tau)
                ret.update({'tal_loss': TAL_global_loss + TAL_grab_loss}) 
            else:
                ret.update({'tal_loss': TAL_global_loss})

        zero = zero_source.float().sum() * 0.0
        cid_loss = ret.get('cid_loss', zero)
        tal_loss = ret.get('tal_loss', zero)
        host_loss = getattr(self.args, "lambda_host", 1.0) * (cid_loss + tal_loss) if use_host_loss else zero
        target_enrichment_loss = ret.get('target_enrichment_loss', zero)
        ret.update({
            'host_loss': host_loss,
            'loss': host_loss + target_enrichment_loss,
        })

        return ret

def build_model(args, num_classes=11003):
    model = ITSELF(args, num_classes)
    convert_weights(model)

    return model
