import logging
import os
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
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


def _should_track_scalar(key):
    return "loss" in key or key.endswith("grad_norm") or key == "pool_interval_reused"


def _grad_norm(parameters):
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.detach().data.float().norm(2).item()
        total += param_norm ** 2
    return total ** 0.5


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


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer, target_pool=None):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("ITSELF.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "supid_loss": AverageMeter(),
        "cotrl_loss": AverageMeter(),
        "cid_loss": AverageMeter(),
        "tal_loss": AverageMeter(),
        "host_loss": AverageMeter(),
        "target_enrichment_loss": AverageMeter(),
        "pool_interval_reused": AverageMeter(),
        "grad_norm": AverageMeter(),
        "host_loss_grad_norm": AverageMeter(),
        "cid_loss_grad_norm": AverageMeter(),
        "tal_loss_grad_norm": AverageMeter(),
        "target_enrichment_loss_grad_norm": AverageMeter(),
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

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
            target_cache = None
            if target_pool is not None:
                target_cache = target_pool.get_train_cache(model, batch, epoch, current_steps)
            if args.modify_k:
                ret = model(batch, epoch, current_step=current_steps, target_cache=target_cache)
            else:
                ret = model(batch, epoch, target_cache=target_cache)
            if target_cache is not None and "diagnostics" in target_cache:
                ret["pool_interval_reused"] = target_cache["diagnostics"]["pool_interval_reused"]
            total_loss = ret.get("loss")
            if total_loss is None:
                total_loss = sum([v for k, v in ret.items() if "loss" in k and _is_trainable_loss(v)])
            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            for key, value in ret.items():
                if key == "loss":
                    continue
                if not _should_track_scalar(key):
                    continue
                scalar = _scalar_value(value)
                if scalar is None:
                    continue
                if key not in meters:
                    meters[key] = AverageMeter()
                meters[key].update(scalar, batch_size)
            optimizer.zero_grad()
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            for loss_key in ["host_loss", "cid_loss", "tal_loss", "target_enrichment_loss"]:
                if loss_key in ret:
                    meters[f"{loss_key}_grad_norm"].update(
                        _loss_grad_norm(ret[loss_key], trainable_params),
                        batch_size,
                    )
            total_loss.backward()
            meters['grad_norm'].update(_grad_norm(model.parameters()), batch_size)
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

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.count > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

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
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())
                now_top1 = max(now_top1,top1)
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
