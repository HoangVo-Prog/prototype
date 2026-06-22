from prettytable import PrettyTable
import logging
import re
import time

import torch
import torch.nn.functional as F


def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )
    pred_labels = g_pids[indices.cpu()]
    matches = pred_labels.eq(q_pids.view(-1, 1))

    all_cmc = matches[:, :max_rank].cumsum(1)
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100

    if not get_mAP:
        return all_cmc, indices

    num_rel = matches.sum(1)
    tmp_cmc = matches.cumsum(1)

    inp = [
        tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
        for i, match_row in enumerate(matches)
    ]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


def get_metrics(similarity, qids, gids, n_, retur_indices=False):
    t2i_cmc, t2i_mAP, t2i_mINP, indices = rank(
        similarity=similarity,
        q_pids=qids,
        g_pids=gids,
        max_rank=10,
        get_mAP=True,
    )
    t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.numpy(), t2i_mAP.numpy(), t2i_mINP.numpy()
    row = [
        n_,
        t2i_cmc[0],
        t2i_cmc[4],
        t2i_cmc[9],
        t2i_mAP,
        t2i_mINP,
        t2i_cmc[0] + t2i_cmc[4] + t2i_cmc[9],
    ]
    if retur_indices:
        return row, indices
    return row


def _metric_task_name(task):
    task = str(task).replace("+", "_plus_")
    task = task.replace("(", "_").replace(")", "")
    task = task.replace(".", "p")
    return re.sub(r"[^A-Za-z0-9_/-]+", "_", task).strip("_")


def _row_to_eval_metrics(row):
    task = _metric_task_name(row[0])
    return {
        f"eval/{task}/R1": float(row[1]),
        f"eval/{task}/R5": float(row[2]),
        f"eval/{task}/R10": float(row[3]),
        f"eval/{task}/mAP": float(row[4]),
        f"eval/{task}/mINP": float(row[5]),
        f"eval/{task}/rSum": float(row[6]) if len(row) > 6 else 0.0,
    }


def _scale_scores_like(scores, reference, eps=1e-12):
    score_min = scores.min(dim=1, keepdim=True).values
    score_max = scores.max(dim=1, keepdim=True).values
    ref_min = reference.min(dim=1, keepdim=True).values
    ref_max = reference.max(dim=1, keepdim=True).values

    score_range = (score_max - score_min).clamp_min(eps)
    ref_range = ref_max - ref_min
    return (scores - score_min) / score_range * ref_range + ref_min


def _global_grab_lambdas():
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.68, 0.32]


def _prototype_lambdas():
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _format_lambda(value):
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return "{:.2f}".format(value).rstrip("0").rstrip(".")


def _scaled_fuse(primary_scores, secondary_scores, primary_weight):
    scaled_secondary = _scale_scores_like(secondary_scores, primary_scores)
    return primary_weight * primary_scores + (1.0 - primary_weight) * scaled_secondary


def _ablation_lambda_from_key(key):
    match = re.search(r"\(([-+]?\d*\.?\d+)\)$", str(key))
    return float(match.group(1)) if match else 0.0


class Evaluator:
    def __init__(self, img_loader, txt_loader, args):
        self.img_loader = img_loader
        self.txt_loader = txt_loader
        self.logger = logging.getLogger("ITSELF.eval")
        self.args = args
        self.last_metrics = {}
        self.last_best_task = None
        self.progress_log_interval = max(
            0.0, float(getattr(args, "eval_log_interval", 30.0) or 0.0)
        )

    def _loader_total_items(self, loader):
        dataset = getattr(loader, "dataset", None)
        if dataset is None:
            return None
        try:
            return len(dataset)
        except TypeError:
            return None

    def _format_elapsed(self, seconds):
        seconds = int(max(0.0, seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 0:
            return "{}h{:02d}m{:02d}s".format(hours, minutes, secs)
        if minutes > 0:
            return "{}m{:02d}s".format(minutes, secs)
        return "{}s".format(secs)

    def _log_loader_progress(
        self,
        stage_name,
        batch_idx,
        num_batches,
        processed_items,
        total_items,
        start_time,
        force=False,
    ):
        if not force and self.progress_log_interval <= 0:
            return

        elapsed = self._format_elapsed(time.monotonic() - start_time)
        parts = ["{} progress".format(stage_name)]
        if num_batches is not None:
            parts.append("batch {}/{}".format(batch_idx, num_batches))
        else:
            parts.append("batch {}".format(batch_idx))

        if total_items:
            progress = min(100.0, 100.0 * processed_items / total_items)
            parts.append(
                "items {}/{} ({:.1f}%)".format(
                    processed_items,
                    total_items,
                    progress,
                )
            )
        else:
            parts.append("items {}".format(processed_items))

        parts.append("elapsed {}".format(elapsed))
        self.logger.info(" | ".join(parts))

    def _encode_loader(self, loader, stage_name, encoder_fn, device):
        ids = []
        features = []
        processed_items = 0
        total_items = self._loader_total_items(loader)
        try:
            num_batches = len(loader)
        except TypeError:
            num_batches = None

        self.logger.info(
            "Starting {}{}".format(
                stage_name,
                " for {} items".format(total_items) if total_items else "",
            )
        )
        start_time = time.monotonic()
        last_log_time = start_time

        for batch_idx, (pid, payload) in enumerate(loader, 1):
            payload = payload.to(device)
            with torch.no_grad():
                feat = encoder_fn(payload).cpu()
            ids.append(pid.view(-1))
            features.append(feat)
            processed_items += int(pid.numel())

            now = time.monotonic()
            if (
                batch_idx == 1
                or (num_batches is not None and batch_idx == num_batches)
                or (
                    self.progress_log_interval > 0
                    and now - last_log_time >= self.progress_log_interval
                )
            ):
                self._log_loader_progress(
                    stage_name,
                    batch_idx,
                    num_batches,
                    processed_items,
                    total_items,
                    start_time,
                    force=True,
                )
                last_log_time = now

        ids = torch.cat(ids, 0)
        features = torch.cat(features, 0)
        self.logger.info(
            "Finished {}: {} items in {}".format(
                stage_name,
                processed_items,
                self._format_elapsed(time.monotonic() - start_time),
            )
        )
        return features.cpu(), ids.cpu()

    def _log_task_progress(self, completed, total_tasks, start_time, force=False):
        if not force and self.progress_log_interval <= 0:
            return
        elapsed = self._format_elapsed(time.monotonic() - start_time)
        self.logger.info(
            "Scoring evaluation tasks: {}/{} complete | elapsed {}".format(
                completed,
                total_tasks,
                elapsed,
            )
        )

    def _compute_embedding(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qfeats, qids = self._encode_loader(
            self.txt_loader,
            "global text embeddings",
            model.encode_text,
            device,
        )
        gfeats, gids = self._encode_loader(
            self.img_loader,
            "global image embeddings",
            model.encode_image,
            device,
        )

        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()

    def _compute_embedding_grab(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        qfeats, qids = self._encode_loader(
            self.txt_loader,
            "GRAB text embeddings",
            model.encode_text_grab,
            device,
        )
        gfeats, gids = self._encode_loader(
            self.img_loader,
            "GRAB image embeddings",
            model.encode_image_grab,
            device,
        )
        return qfeats.cpu(), gfeats.cpu(), qids.cpu(), gids.cpu()

    def _compute_target_gallery_cache(self, model):
        model = model.eval()
        device = next(model.parameters()).device
        gids, cache_chunks = [], []
        processed_items = 0
        total_items = self._loader_total_items(self.img_loader)
        try:
            num_batches = len(self.img_loader)
        except TypeError:
            num_batches = None

        self.logger.info(
            "Starting target gallery cache build{}".format(
                " for {} items".format(total_items) if total_items else ""
            )
        )
        start_time = time.monotonic()
        last_log_time = start_time

        for batch_idx, (pid, img) in enumerate(self.img_loader, 1):
            img = img.to(device)
            with torch.no_grad():
                cache = model.encode_target_image_cache(img)
            gids.append(pid.view(-1))
            cache_chunks.append({k: v.detach().cpu() for k, v in cache.items()})
            processed_items += int(pid.numel())

            now = time.monotonic()
            if (
                batch_idx == 1
                or (num_batches is not None and batch_idx == num_batches)
                or (
                    self.progress_log_interval > 0
                    and now - last_log_time >= self.progress_log_interval
                )
            ):
                self._log_loader_progress(
                    "target gallery cache",
                    batch_idx,
                    num_batches,
                    processed_items,
                    total_items,
                    start_time,
                    force=True,
                )
                last_log_time = now

        gids = torch.cat(gids, 0)
        target_cache = {}
        for key in cache_chunks[0].keys():
            target_cache[key] = torch.cat(
                [chunk[key] for chunk in cache_chunks], dim=0
            ).to(device)
        target_cache["pids"] = gids.to(device)
        core_model = model.module if hasattr(model, "module") else model
        if hasattr(core_model, "finalize_target_cache"):
            target_cache = core_model.finalize_target_cache(target_cache)
        self.logger.info(
            "Finished target gallery cache: {} items in {}".format(
                processed_items,
                self._format_elapsed(time.monotonic() - start_time),
            )
        )
        return target_cache, gids.cpu()

    def _compute_enriched_text_embedding(self, model, target_cache):
        model = model.eval()
        device = next(model.parameters()).device

        qids, qfeats = [], []
        processed_items = 0
        total_items = self._loader_total_items(self.txt_loader)
        try:
            num_batches = len(self.txt_loader)
        except TypeError:
            num_batches = None

        self.logger.info(
            "Starting enriched text embeddings{}".format(
                " for {} items".format(total_items) if total_items else ""
            )
        )
        start_time = time.monotonic()
        last_log_time = start_time

        for batch_idx, (pid, caption) in enumerate(self.txt_loader, 1):
            caption = caption.to(device)
            with torch.no_grad():
                host_text_feat = model.encode_text(caption)
                grab_text_feat = None
                if (
                    self.args.enrichment_space == "grab"
                    or getattr(self.args, "topm_rank_space", "host_global") == "hybrid_global_grab"
                ):
                    grab_text_feat = model.encode_text_grab(caption)
                if self.args.enrichment_space == "grab":
                    query_feat = grab_text_feat
                else:
                    query_feat = host_text_feat
                text_feat = model.enrich_text_features(
                    query_feat,
                    host_text_feat,
                    target_cache,
                    grab_text_features=grab_text_feat,
                ).cpu()
            qids.append(pid.view(-1))
            qfeats.append(text_feat)
            processed_items += int(pid.numel())

            now = time.monotonic()
            if (
                batch_idx == 1
                or (num_batches is not None and batch_idx == num_batches)
                or (
                    self.progress_log_interval > 0
                    and now - last_log_time >= self.progress_log_interval
                )
            ):
                self._log_loader_progress(
                    "enriched text embeddings",
                    batch_idx,
                    num_batches,
                    processed_items,
                    total_items,
                    start_time,
                    force=True,
                )
                last_log_time = now

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        self.logger.info(
            "Finished enriched text embeddings: {} items in {}".format(
                processed_items,
                self._format_elapsed(time.monotonic() - start_time),
            )
        )
        return qfeats.cpu(), qids.cpu()

    def _iter_base_tasks(self, sims_global, sims_grab):
        yield "global", sims_global
        if self.args.only_global:
            return

        yield "grab", sims_grab
        for lambda_value in _global_grab_lambdas():
            alpha = _format_lambda(lambda_value)
            yield "global+grab({})".format(alpha), _scaled_fuse(
                sims_global,
                sims_grab,
                lambda_value,
            )

    def _iter_eval_tasks(self, sims_global, sims_grab, sims_target):
        for task_name, task_scores in self._iter_base_tasks(sims_global, sims_grab):
            yield task_name, task_scores

        if sims_target is None:
            return

        for proto_lambda in _prototype_lambdas():
            proto_value = _format_lambda(proto_lambda)
            for base_name, base_scores in self._iter_base_tasks(sims_global, sims_grab):
                fused_name = "{}+proto({})".format(base_name, proto_value)
                scaled_base_scores = _scale_scores_like(base_scores, sims_target)
                yield fused_name, (
                    (1.0 - proto_lambda) * scaled_base_scores
                    + proto_lambda * sims_target
                )

    def eval(self, model, i2t_metric=False, use_target_enrichment=None):
        if use_target_enrichment is None:
            use_target_enrichment = getattr(self.args, "target_enrichment", False)

        self.logger.info("Starting evaluation feature extraction")
        qfeats, gfeats, qids, gids = self._compute_embedding(model)
        qfeats = F.normalize(qfeats, p=2, dim=1)
        gfeats = F.normalize(gfeats, p=2, dim=1)
        sims_global = qfeats @ gfeats.t()
        self.logger.info(
            "Global similarity matrix ready: {} queries x {} gallery".format(
                qfeats.shape[0],
                gfeats.shape[0],
            )
        )

        sims_grab = None
        if not self.args.only_global:
            self.logger.info("Computing GRAB retrieval features")
            vq_feats, vg_feats, _, _ = self._compute_embedding_grab(model)
            vq_feats = F.normalize(vq_feats, p=2, dim=1)
            vg_feats = F.normalize(vg_feats, p=2, dim=1)
            sims_grab = vq_feats @ vg_feats.t()
            self.logger.info("GRAB similarity matrix ready")

        sims_target = None
        if use_target_enrichment:
            self.logger.info("Computing target-aware enrichment features")
            target_cache, target_gids = self._compute_target_gallery_cache(model)
            target_qfeats, target_qids = self._compute_enriched_text_embedding(
                model,
                target_cache,
            )
            target_qfeats = F.normalize(target_qfeats, p=2, dim=1)
            target_gfeats = F.normalize(
                target_cache["retrieval_features"].detach().cpu(),
                p=2,
                dim=1,
            )
            sims_target = target_qfeats @ target_gfeats.t()
            qids = target_qids
            gids = target_gids
            self.logger.info("Target-aware similarity matrix ready")

        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP", "rSum"])

        top1 = 0
        eval_metrics = {}
        rows_by_task = {}
        best_task = None
        best_row = None
        best_ablation_task = None
        best_ablation_row = None
        eval_tasks = list(self._iter_eval_tasks(sims_global, sims_grab, sims_target))
        self.logger.info("Scoring {} evaluation tasks".format(len(eval_tasks)))
        task_start_time = time.monotonic()
        last_task_log_time = task_start_time

        for task_idx, (key, sims) in enumerate(eval_tasks, 1):
            rs = get_metrics(sims, qids, gids, "{}-t2i".format(key), False)
            table.add_row(rs)
            rows_by_task[key] = rs
            eval_metrics.update(_row_to_eval_metrics(rs))
            if i2t_metric:
                i2t_cmc, i2t_mAP, i2t_mINP, _ = rank(
                    similarity=sims.t(),
                    q_pids=gids,
                    g_pids=qids,
                    max_rank=10,
                    get_mAP=True,
                )
                i2t_cmc, i2t_mAP, i2t_mINP = (
                    i2t_cmc.numpy(),
                    i2t_mAP.numpy(),
                    i2t_mINP.numpy(),
                )
                i2t_row = [
                    "{}-i2t".format(key),
                    i2t_cmc[0],
                    i2t_cmc[4],
                    i2t_cmc[9],
                    i2t_mAP,
                    i2t_mINP,
                    i2t_cmc[0] + i2t_cmc[4] + i2t_cmc[9],
                ]
                table.add_row(i2t_row)
                eval_metrics.update(_row_to_eval_metrics(i2t_row))

            if best_row is None or rs[1] > best_row[1]:
                best_task = key
                best_row = rs
            if "+proto(" in key and (
                best_ablation_row is None or rs[1] > best_ablation_row[1]
            ):
                best_ablation_task = key
                best_ablation_row = rs

            now = time.monotonic()
            if (
                task_idx == 1
                or task_idx == len(eval_tasks)
                or (
                    self.progress_log_interval > 0
                    and now - last_task_log_time >= self.progress_log_interval
                )
            ):
                self._log_task_progress(
                    task_idx,
                    len(eval_tasks),
                    task_start_time,
                    force=True,
                )
                last_task_log_time = now

        if best_ablation_row is not None:
            top1 = float(best_ablation_row[1])
            best_task = best_ablation_task
            eval_metrics["eval/ablation_best_R1"] = float(best_ablation_row[1])
            eval_metrics["eval/ablation_best_R5"] = float(best_ablation_row[2])
            eval_metrics["eval/ablation_best_R10"] = float(best_ablation_row[3])
            eval_metrics["eval/ablation_best_mAP"] = float(best_ablation_row[4])
            eval_metrics["eval/ablation_best_mINP"] = float(best_ablation_row[5])
            eval_metrics["eval/ablation_best_rSum"] = float(best_ablation_row[6])
            eval_metrics["eval/ablation_best_lambda"] = _ablation_lambda_from_key(
                best_ablation_task
            )
        elif best_row is not None:
            top1 = float(best_row[1])

        target_key = "global+proto(1)"
        if "global" in rows_by_task and target_key in rows_by_task:
            global_row = rows_by_task["global"]
            target_row = rows_by_task[target_key]
            eval_metrics["eval/delta_R1_target_vs_global"] = float(
                target_row[1] - global_row[1]
            )
            eval_metrics["eval/delta_R5_target_vs_global"] = float(
                target_row[2] - global_row[2]
            )
            eval_metrics["eval/delta_R10_target_vs_global"] = float(
                target_row[3] - global_row[3]
            )
            eval_metrics["eval/delta_mAP_target_vs_global"] = float(
                target_row[4] - global_row[4]
            )
            eval_metrics["eval/delta_mINP_target_vs_global"] = float(
                target_row[5] - global_row[5]
            )
            eval_metrics["eval/delta_rSum_target_vs_global"] = float(
                target_row[6] - global_row[6]
            )

        self.last_metrics = eval_metrics
        self.last_best_task = best_task

        table.custom_format["R1"] = lambda f, v: "{:.2f}".format(v)
        table.custom_format["R5"] = lambda f, v: "{:.2f}".format(v)
        table.custom_format["R10"] = lambda f, v: "{:.2f}".format(v)
        table.custom_format["mAP"] = lambda f, v: "{:.2f}".format(v)
        table.custom_format["mINP"] = lambda f, v: "{:.2f}".format(v)
        table.custom_format["rSum"] = lambda f, v: "{:.2f}".format(v)
        self.logger.info("\n" + str(table))
        self.logger.info("\n" + "best R1 = " + str(top1))
        if best_task is not None:
            self.logger.info("best R1 row = {}".format(best_task))

        return top1
