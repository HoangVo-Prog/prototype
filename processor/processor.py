import logging
import math
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, reduce_dict, synchronize
from torch.utils.tensorboard import SummaryWriter
from utils.tracking import ExperimentTracker


def _to_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "detach"):
        detached = value.detach()
        if detached.numel() == 1:
            return float(detached.item())
    return None


def _extract_prototype_scalars(prototype_stats):
    if not prototype_stats:
        return {}

    metrics = {}
    for source_key, target_key in (
        ("routing_entropy", "prototype_routing_entropy"),
        ("effective_routing", "prototype_effective_routing"),
        ("assignment_compactness", "prototype_assignment_compactness"),
    ):
        value = prototype_stats.get(source_key)
        if value is not None:
            metrics[target_key] = float(value.mean().item())

    for source_key, target_key in (
        ("prototype_usage_occupancy", "prototype_usage_occupancy"),
        ("prototype_usage_fraction", "prototype_usage_fraction"),
        ("prototype_bank_nn_distance", "prototype_bank_nn_distance"),
        ("prototype_tau", "prototype_tau"),
    ):
        value = prototype_stats.get(source_key)
        scalar = _to_float(value)
        if scalar is not None:
            metrics[target_key] = scalar

    return metrics


def _get_prototype_grad_norm(model):
    target_model = model.module if hasattr(model, "module") else model
    if not getattr(target_model, "use_prototype", False):
        return None

    grad = target_model.prototype_module.visual_meta_matrix.grad
    if grad is None:
        return None
    return float(grad.detach().norm().item())


def _get_prototype_param(model):
    target_model = model.module if hasattr(model, "module") else model
    if not getattr(target_model, "use_prototype", False):
        return None
    parameter = target_model.prototype_module.visual_meta_matrix
    if not parameter.requires_grad:
        return None
    return parameter


def _get_loss_grad_norm(loss_value, parameter):
    if loss_value is None or parameter is None or not parameter.requires_grad:
        return None

    grads = torch.autograd.grad(
        loss_value,
        parameter,
        retain_graph=True,
        allow_unused=True,
    )
    grad = grads[0]
    if grad is None:
        return None
    return float(grad.detach().norm().item())


def _get_console_metric(scalar_metrics, key, default=float("nan")):
    value = scalar_metrics.get(key, default)
    if value is None:
        return default
    return value


def _get_meter_avg(meters, key, default=float("nan")):
    meter = meters.get(key)
    if meter is None or meter.count == 0:
        return default
    return float(meter.avg)



def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0
    arguments["epoch"] = start_epoch - 1

    logger = logging.getLogger("ITSELF.train")
    logger.info('start training')

    meters = {}

    tb_writer = SummaryWriter(log_dir=args.output_dir) if get_rank() == 0 else None
    tracker = ExperimentTracker(args, tb_writer=tb_writer)

    best_top1 = 0.0
    evaluator.eval(model.eval())
    # train
    now_top1 = 0
    current_epoch = 0
    current_steps = 0 
    for epoch in range(start_epoch, num_epoch + 1):
        current_epoch += 1
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.train()
        model.epoch = epoch

        
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            should_log_step = (n_iter + 1) % log_period == 0
            if args.modify_k:
                ret = model(
                    batch,
                    epoch,
                    current_step=current_steps,
                    return_prototype_stats=should_log_step,
                )
            else:
                ret = model(
                    batch,
                    epoch,
                    return_prototype_stats=should_log_step,
                )
            total_loss = sum([v for k, v in ret.items() if "loss" in k])
            batch_size = batch['images'].shape[0]

            reduced_metrics = {"loss": total_loss.detach()}
            for key, value in ret.items():
                if "loss" in key or key == "temperature":
                    if hasattr(value, "detach"):
                        reduced_metrics[key] = value.detach()
                    elif isinstance(value, (int, float)):
                        reduced_metrics[key] = torch.tensor(value, device=device, dtype=torch.float32)

            prototype_step_metrics = _extract_prototype_scalars(ret.get("prototype_stats"))
            for key, value in prototype_step_metrics.items():
                reduced_metrics[key] = torch.tensor(value, device=device, dtype=torch.float32)

            if should_log_step:
                prototype_param = _get_prototype_param(model)
                for loss_key in ("cid_loss", "tal_loss", "div_loss"):
                    loss_grad_norm = _get_loss_grad_norm(ret.get(loss_key), prototype_param)
                    if loss_grad_norm is not None:
                        reduced_metrics[f"{loss_key}_grad_norm"] = torch.tensor(
                            loss_grad_norm, device=device, dtype=torch.float32
                        )

            optimizer.zero_grad()
            total_loss.backward()
            prototype_grad_norm = _get_prototype_grad_norm(model)
            if prototype_grad_norm is not None:
                reduced_metrics["prototype_grad_norm"] = torch.tensor(
                    prototype_grad_norm, device=device, dtype=torch.float32
                )
            optimizer.step()
            synchronize()

            reduced_metrics = reduce_dict(reduced_metrics, average=True)
            scalar_metrics = {
                key: float(value.item()) if hasattr(value, "item") else float(value)
                for key, value in reduced_metrics.items()
            }

            for key, value in scalar_metrics.items():
                meter = meters.setdefault(key, AverageMeter())
                meter.update(value, batch_size)

            if should_log_step and get_rank() == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                console_keys = (
                    "loss",
                    "tal_loss",
                    "cid_loss",
                    "div_loss",
                    "tal_loss_grad_norm",
                    "cid_loss_grad_norm",
                    "div_loss_grad_norm",
                    "prototype_grad_norm",
                )
                for key in console_keys:
                    metric_value = _get_meter_avg(meters, key)
                    if math.isnan(metric_value):
                        info_str += f", {key}: nan"
                    else:
                        info_str += f", {key}: {metric_value:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

                train_log_metrics = {
                    "train/loss": _get_meter_avg(meters, "loss"),
                    "train/lr": scheduler.get_lr()[0],
                    "train/div_loss": _get_meter_avg(meters, "div_loss"),
                    "train/tal_loss_grad_norm": _get_meter_avg(meters, "tal_loss_grad_norm"),
                    "train/cid_loss_grad_norm": _get_meter_avg(meters, "cid_loss_grad_norm"),
                    "train/div_loss_grad_norm": _get_meter_avg(meters, "div_loss_grad_norm"),
                    "train/prototype_grad_norm": _get_meter_avg(meters, "prototype_grad_norm"),
                }
                for key in scalar_metrics.keys():
                    if key == "loss":
                        continue
                    if key.endswith("_loss"):
                        train_log_metrics[f"train/{key}"] = _get_meter_avg(meters, key)
                    elif key == "temperature":
                        train_log_metrics["train/temperature"] = _get_meter_avg(meters, key)
                    elif key.startswith("prototype_"):
                        train_log_metrics[f"train/{key}"] = _get_meter_avg(meters, key)

                tracker.add_tb_scalars(train_log_metrics, current_steps)
                tracker.log_with_step_metric(train_log_metrics, "train/step", current_steps)

        if get_rank() == 0:
            epoch_metrics = {
                "epoch/lr": scheduler.get_lr()[0],
                "epoch/div_loss": meters["div_loss"].avg if "div_loss" in meters else float("nan"),
                "epoch/tal_loss_grad_norm": meters["tal_loss_grad_norm"].avg if "tal_loss_grad_norm" in meters else float("nan"),
                "epoch/cid_loss_grad_norm": meters["cid_loss_grad_norm"].avg if "cid_loss_grad_norm" in meters else float("nan"),
                "epoch/div_loss_grad_norm": meters["div_loss_grad_norm"].avg if "div_loss_grad_norm" in meters else float("nan"),
                "epoch/prototype_grad_norm": meters["prototype_grad_norm"].avg if "prototype_grad_norm" in meters else float("nan"),
            }
            if "temperature" in meters:
                epoch_metrics["epoch/temperature"] = meters["temperature"].avg
            for key, meter in meters.items():
                if meter.avg > 0 and (key.endswith("_loss") or key == "loss"):
                    epoch_metrics[f"epoch/{key}"] = meter.avg
            tracker.add_tb_scalars(epoch_metrics, epoch)
            tracker.log_with_step_metric(epoch_metrics, "epoch", epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0: 
        # if epoch % eval_period == 0 and epoch >= 61:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    eval_result = evaluator.eval(model.module.eval())
                else:
                    eval_result = evaluator.eval(model.eval())
                top1 = eval_result["best_top1"]
                now_top1 = max(now_top1,top1)
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save(
                        "best",
                        save_backbone=args.save_backbone_ckpt,
                        save_prototype=args.save_prototype_ckpt,
                        **arguments,
                    )

                val_log_metrics = {}
                for branch_name, branch_metrics in eval_result["metrics"].items():
                    for metric_name, metric_value in branch_metrics.items():
                        val_log_metrics[f"val/{branch_name}/{metric_name}"] = metric_value
                for metric_name, metric_value in eval_result.get("prototype_diagnostics", {}).items():
                    val_log_metrics[f"val/{metric_name}"] = metric_value
                val_log_metrics["val/best_R1"] = best_top1
                tracker.add_tb_scalars(val_log_metrics, epoch)
                tracker.log_with_step_metric(val_log_metrics, "epoch", epoch)
                tracker.update_summary({"best_R1": best_top1, "val/best_R1": best_top1})
                
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")
    tracker.finish()

                   
def do_inference(model, test_img_loader, test_txt_loader, args):

    logger = logging.getLogger("ITSELF.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())
