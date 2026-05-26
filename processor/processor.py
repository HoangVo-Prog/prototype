import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from utils.wandb_utils import log_wandb
from torch.utils.tensorboard import SummaryWriter


def _scalar_value(value):
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return value.detach().float().item()
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_trainable_loss(value):
    return torch.is_tensor(value) and value.numel() == 1 and value.requires_grad


def _is_loss_key(key):
    return key == "loss" or key.endswith("_loss")


def _should_track_log_scalar(key):
    return "loss" in key or key.endswith("grad_norm")


def _should_track_wandb_scalar(key):
    return (
        "loss" in key
        or key.endswith("grad_norm")
        or key.startswith("pool_")
        or key.startswith("target_")
        or key.startswith("mixer/")
    )


def _grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.float().norm(2).item()
        total += param_norm ** 2
    return total ** 0.5


def _target_enrichment_active(args, epoch):
    enrichment_start = getattr(args, "enrichment_start", 1)
    if enrichment_start < 1:
        raise ValueError("--enrichment_start must be a positive integer")
    return getattr(args, "target_enrichment", False) and epoch >= enrichment_start


def _loss_grad_norm(loss, parameters):
    if not _is_trainable_loss(loss):
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = 0.0
    for grad in grads:
        if grad is None:
            continue
        grad_norm = grad.detach().float().norm(2).item()
        total += grad_norm ** 2
    return total ** 0.5


def _iter_loss_grad_sources(ret):
    sources = {}
    explicit_sources = ret.get("_loss_grad_sources", {})
    if isinstance(explicit_sources, dict):
        for key, value in explicit_sources.items():
            if _is_loss_key(key):
                sources[key] = value

    for key, value in ret.items():
        if key == "_loss_grad_sources":
            continue
        if _is_loss_key(key) and key not in sources:
            sources[key] = value
    return sources.items()


def _update_meter(meters, key, value, batch_size):
    if key not in meters:
        meters[key] = AverageMeter()
    meters[key].update(value, batch_size)


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, target_pool=None, wandb_run=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("ITSELF.train")
    logger.info('start training')
    if target_pool is not None and getattr(args, "enrichment_start", 1) > 1:
        logger.info(
            "Target enrichment delayed until epoch {}; earlier epochs use host training only".format(
                args.enrichment_start
            )
        )

    meters = {
        "loss": AverageMeter(),
        "supid_loss": AverageMeter(),
        "cotrl_loss": AverageMeter(),
        "cid_loss": AverageMeter(),
        "tal_loss": AverageMeter(),
        "host_loss": AverageMeter(),
        "target_enrichment_loss": AverageMeter(),
        "grad_norm": AverageMeter(),
        "host_loss_grad_norm": AverageMeter(),
        "cid_loss_grad_norm": AverageMeter(),
        "tal_loss_grad_norm": AverageMeter(),
        "target_enrichment_loss_grad_norm": AverageMeter(),
    }
    wandb_meters = {}

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0
    initial_top1 = evaluator.eval(
        model.eval(),
        use_target_enrichment=_target_enrichment_active(args, start_epoch),
    )
    if get_rank() == 0:
        initial_metrics = dict(getattr(evaluator, "last_metrics", {}))
        initial_metrics["eval/top_R1"] = initial_top1
        log_wandb(wandb_run, initial_metrics, step=0, epoch=start_epoch - 1)
    # train
    now_top1 = 0
    current_epoch = 0
    current_steps = 0 
    for epoch in range(start_epoch, num_epoch + 1):
        current_epoch += 1
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        for meter in wandb_meters.values():
            meter.reset()

        model.train()
        model.epoch = epoch
        use_target_enrichment = target_pool is not None and _target_enrichment_active(args, epoch)
        if target_pool is not None and epoch == getattr(args, "enrichment_start", 1):
            logger.info("Target enrichment starts at epoch {}".format(epoch))

        
        for n_iter, batch in enumerate(train_loader):
            current_steps += 1
            batch = {k: v.to(device) for k, v in batch.items()}
            target_cache = None
            if use_target_enrichment:
                target_cache = target_pool.get_train_cache(model, batch, epoch, current_steps)
            if args.modify_k:
                ret = model(batch, epoch, current_step=current_steps, target_cache=target_cache)
            else:
                ret = model(batch, epoch, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                for diag_key, diag_value in target_cache["diagnostics"].items():
                    if isinstance(diag_value, (int, float)):
                        ret[diag_key] = diag_value
            total_loss = ret.get("loss")
            if total_loss is None:
                total_loss = sum([v for k, v in ret.items() if "loss" in k and _is_trainable_loss(v)])
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            _update_meter(wandb_meters, "loss", total_loss.item(), batch_size)
            for key, value in ret.items():
                if key == "loss":
                    continue
                scalar = _scalar_value(value)
                if scalar is None:
                    continue
                if _should_track_log_scalar(key):
                    _update_meter(meters, key, scalar, batch_size)
                if _should_track_wandb_scalar(key):
                    _update_meter(wandb_meters, key, scalar, batch_size)
            optimizer.zero_grad()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            loss_grad_sources = dict(_iter_loss_grad_sources(ret))
            loss_grad_sources["loss"] = total_loss
            for loss_key, loss_value in loss_grad_sources.items():
                grad_norm_key = f"{loss_key}_grad_norm"
                grad_norm_value = _loss_grad_norm(loss_value, trainable_params)
                _update_meter(meters, grad_norm_key, grad_norm_value, batch_size)
                _update_meter(wandb_meters, grad_norm_key, grad_norm_value, batch_size)
            total_loss.backward()
            grad_norm_value = _grad_norm(model.parameters())
            meters['grad_norm'].update(grad_norm_value, batch_size)
            _update_meter(wandb_meters, "grad_norm", grad_norm_value, batch_size)
            optimizer.step()
            synchronize()
            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.count > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
                if get_rank() == 0:
                    train_metrics = {
                        "train/{}".format(k): v.avg
                        for k, v in wandb_meters.items()
                        if v.count > 0
                    }
                    train_metrics["train/lr"] = scheduler.get_lr()[0]
                    train_metrics["train/temperature"] = _scalar_value(ret.get("temperature"))
                    log_wandb(wandb_run, train_metrics, step=current_steps, epoch=epoch)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)
        if get_rank() == 0:
            epoch_metrics = {
                "train_epoch/{}".format(k): v.avg
                for k, v in wandb_meters.items()
                if v.count > 0
            }
            epoch_metrics["train_epoch/lr"] = scheduler.get_lr()[0]
            epoch_metrics["train_epoch/temperature"] = _scalar_value(ret.get("temperature"))
            log_wandb(wandb_run, epoch_metrics, step=current_steps, epoch=epoch)

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
                    top1 = evaluator.eval(
                        model.module.eval(),
                        use_target_enrichment=_target_enrichment_active(args, epoch),
                    )
                else:
                    top1 = evaluator.eval(
                        model.eval(),
                        use_target_enrichment=_target_enrichment_active(args, epoch),
                    )
                now_top1 = max(now_top1,top1)
                eval_metrics = dict(getattr(evaluator, "last_metrics", {}))
                eval_metrics["eval/top_R1"] = top1
                eval_metrics["eval/best_R1"] = now_top1
                log_wandb(wandb_run, eval_metrics, step=current_steps, epoch=epoch)
                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)
                
 
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")

                   
def do_inference(model, test_img_loader, test_txt_loader, args):

    logger = logging.getLogger("ITSELF.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader, args)
    _ = evaluator.eval(model.eval())
