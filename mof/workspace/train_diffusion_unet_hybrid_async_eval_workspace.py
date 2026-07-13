if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import copy
import random
import pathlib
import multiprocessing as mp
import queue

import hydra
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mof.dataset.base_dataset import BaseImageDataset
from mof.common.checkpoint_util import TopKCheckpointManager
from mof.common.json_logger import JsonLogger
from mof.common.pytorch_util import dict_apply, optimizer_to
from mof.model.diffusion.ema_model import EMAModel
from mof.model.common.lr_scheduler import get_scheduler
from mof.workspace.async_eval_util import (
    _async_eval_worker_main,
    _resolve_async_eval_device,
    _shutdown_eval_worker,
)
from mof.workspace.train_diffusion_unet_hybrid_workspace import (
    TrainDiffusionUnetHybridWorkspace,
)

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainDiffusionUnetHybridAsyncEvalWorkspace(TrainDiffusionUnetHybridWorkspace):
    def run(self):
        cfg = copy.deepcopy(self.cfg)
        eval_process = None
        eval_req_q = None
        eval_resp_q = None

        try:
            if cfg.training.resume:
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    self.load_checkpoint(path=lastest_ckpt_path)
                    self.epoch += 1
                    self.global_step += 1

            enable_rollout = bool(cfg.training.get("enable_rollout", True))
            async_eval_every = int(cfg.training.get("async_eval_every", cfg.training.rollout_every))
            async_eval_subdir = str(cfg.training.get("async_eval_subdir", "async_eval"))
            async_eval_device, async_eval_cuda_visible_devices = _resolve_async_eval_device(cfg.training)

            eval_inflight = False
            pending_eval_req = None
            async_status = "idle"
            async_status_epoch = -1

            if enable_rollout:
                async_eval_root = pathlib.Path(self.output_dir).joinpath(async_eval_subdir)
                async_eval_root.mkdir(parents=True, exist_ok=True)
                bootstrap_ckpt = pathlib.Path(self.output_dir).joinpath(
                    "checkpoints", "async_eval_bootstrap.ckpt"
                )
                self.save_checkpoint(path=bootstrap_ckpt, use_thread=False)

                ctx = mp.get_context("spawn")
                eval_req_q = ctx.Queue()
                eval_resp_q = ctx.Queue()
                eval_process = ctx.Process(
                    target=_async_eval_worker_main,
                    args=(
                        OmegaConf.to_container(cfg, resolve=True),
                        str(bootstrap_ckpt),
                        str(async_eval_root),
                        async_eval_device,
                        async_eval_cuda_visible_devices,
                        False,
                        eval_req_q,
                        eval_resp_q,
                    ),
                    daemon=False,
                )
                eval_process.start()

            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            assert isinstance(dataset, BaseImageDataset)
            train_dataloader = DataLoader(dataset, **cfg.dataloader)
            normalizer = dataset.get_normalizer()

            val_dataset = dataset.get_validation_dataset()
            val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

            self.model.set_normalizer(normalizer)
            if cfg.training.use_ema:
                self.ema_model.set_normalizer(normalizer)

            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs
                ) // cfg.training.gradient_accumulate_every,
                last_epoch=self.global_step - 1,
            )

            ema: EMAModel = None
            if cfg.training.use_ema:
                ema = hydra.utils.instantiate(
                    cfg.ema,
                    model=self.ema_model)

            wandb_run = wandb.init(
                dir=str(self.output_dir),
                config=OmegaConf.to_container(cfg, resolve=True),
                **cfg.logging
            )
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

            device = torch.device(cfg.training.device)
            self.model.to(device)
            if self.ema_model is not None:
                self.ema_model.to(device)
            optimizer_to(self.optimizer, device)

            def _collect_train_metrics():
                if not hasattr(self.model, "get_last_train_metrics"):
                    return {}
                metrics = self.model.get_last_train_metrics()
                if not isinstance(metrics, dict):
                    return {}
                reduced_metrics = {}
                for key, value in metrics.items():
                    if not isinstance(value, torch.Tensor):
                        continue
                    reduced_metrics[key] = float(value.detach().item())
                return reduced_metrics

            def _dispatch_eval_request(eval_req):
                nonlocal eval_inflight, async_status, async_status_epoch
                if eval_req_q is None:
                    return
                eval_req_q.put(eval_req)
                eval_inflight = True
                async_status = "run"
                async_status_epoch = int(eval_req["epoch"])

            def _cleanup_async_eval_ckpt(ckpt_path):
                if ckpt_path is None:
                    return
                ckpt_path = pathlib.Path(ckpt_path)
                try:
                    if ckpt_path.is_file():
                        ckpt_path.unlink()
                except FileNotFoundError:
                    pass

            def _finalize_async_eval_checkpoint(msg):
                ckpt_path = msg.get("ckpt_path", None)
                if ckpt_path is None:
                    return

                metric_dict = {
                    key.replace('/', '_'): value
                    for key, value in msg.get("metrics", {}).items()
                }
                metric_dict["epoch"] = int(msg.get("epoch", -1))
                metric_dict["global_step"] = int(msg.get("step", self.global_step))

                if topk_manager.monitor_key not in metric_dict:
                    _cleanup_async_eval_ckpt(ckpt_path)
                    return

                topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                if topk_ckpt_path is None:
                    _cleanup_async_eval_ckpt(ckpt_path)
                    return

                src_path = pathlib.Path(ckpt_path)
                dst_path = pathlib.Path(topk_ckpt_path)
                if not src_path.is_file():
                    return

                dst_path.parent.mkdir(parents=True, exist_ok=True)
                if src_path.resolve() != dst_path.resolve():
                    if dst_path.exists():
                        dst_path.unlink()
                    os.replace(src_path, dst_path)

            def _poll_async_eval():
                nonlocal eval_inflight, pending_eval_req, async_status, async_status_epoch
                if eval_process is None or eval_resp_q is None:
                    return

                while True:
                    try:
                        msg = eval_resp_q.get_nowait()
                    except queue.Empty:
                        break

                    status = msg.get("status", "unknown")
                    if status == "ok":
                        metrics = msg.get("metrics", {})
                        ckpt_epoch = int(msg.get("epoch", -1))
                        json_log_data = dict(metrics)
                        json_log_data["epoch"] = ckpt_epoch
                        json_log_data["eval_finished"] = 1.0
                        wandb_log_data = dict(json_log_data)
                        for key, path in msg.get("video_paths", {}).items():
                            wandb_log_data[key] = wandb.Video(path)
                        step = int(msg.get("step", self.global_step))
                        if len(wandb_log_data) > 0:
                            wandb_run.log(wandb_log_data)
                        if len(json_log_data) > 0:
                            json_logger.log(dict(json_log_data, global_step=step))
                        _finalize_async_eval_checkpoint(msg)
                        async_status = "ok"
                        async_status_epoch = ckpt_epoch
                    else:
                        async_status = "err"
                        async_status_epoch = int(msg.get("epoch", -1))
                        tb = msg.get("traceback", "")
                        eval_inflight = False
                        pending_eval_req = None
                        raise RuntimeError(
                            f"Async eval worker failed at epoch={msg.get('epoch', -1)}, "
                            f"step={msg.get('step', self.global_step)}.\n{tb}"
                        )
                    eval_inflight = False

                if eval_process is not None and not eval_process.is_alive():
                    async_status = "dead"
                    eval_inflight = False
                    pending_eval_req = None
                    raise RuntimeError("Async eval worker process died unexpectedly.")

                if (not eval_inflight) and (pending_eval_req is not None):
                    req = pending_eval_req
                    pending_eval_req = None
                    _dispatch_eval_request(req)

            def _async_postfix_text():
                if not enable_rollout:
                    return "off"
                if async_status_epoch >= 0:
                    return f"{async_status}@e{async_status_epoch}"
                return async_status

            train_sampling_batch = None

            if cfg.training.debug:
                cfg.training.num_epochs = 2
                cfg.training.max_train_steps = 3
                cfg.training.max_val_steps = 3
                cfg.training.rollout_every = 1
                cfg.training.checkpoint_every = 1
                cfg.training.val_every = 1
                cfg.training.sample_every = 1

            log_path = os.path.join(self.output_dir, 'logs.json.txt')
            with JsonLogger(log_path) as json_logger:
                while self.epoch < cfg.training.num_epochs:
                    if enable_rollout:
                        _poll_async_eval()

                    step_log = dict()
                    train_losses = list()
                    with tqdm.tqdm(
                        train_dataloader,
                        desc=f"Training epoch {self.epoch}",
                        leave=False,
                        mininterval=cfg.training.tqdm_interval_sec,
                    ) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            if enable_rollout:
                                _poll_async_eval()

                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            if train_sampling_batch is None:
                                train_sampling_batch = batch

                            raw_loss = self.model.compute_loss(batch)
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            loss.backward()

                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()

                            if cfg.training.use_ema:
                                ema.step(self.model)

                            raw_loss_cpu = raw_loss.item()
                            train_metric_log = _collect_train_metrics()
                            postfix = {"loss": raw_loss_cpu}
                            postfix.update({
                                key: value
                                for key, value in train_metric_log.items()
                                if key not in {
                                    "router/entropy_coef_eff",
                                    "router/load_balance_coef_eff",
                                    "router/reg_progress",
                                    "router/reg_multiplier",
                                }
                            })
                            if enable_rollout:
                                postfix["async"] = _async_postfix_text()
                            tepoch.set_postfix(**postfix, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': lr_scheduler.get_last_lr()[0]
                            }
                            step_log.update(train_metric_log)

                            is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                            if not is_last_batch:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                                self.global_step += 1

                            if (cfg.training.max_train_steps is not None) \
                                and batch_idx >= (cfg.training.max_train_steps - 1):
                                break

                    train_loss = np.mean(train_losses)
                    step_log['train_loss'] = train_loss

                    policy = self.model
                    if cfg.training.use_ema:
                        policy = self.ema_model
                    policy.eval()

                    if (self.epoch % cfg.training.val_every) == 0:
                        with torch.no_grad():
                            val_losses = list()
                            with tqdm.tqdm(
                                val_dataloader,
                                desc=f"Validation epoch {self.epoch}",
                                leave=False,
                                mininterval=cfg.training.tqdm_interval_sec,
                            ) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    loss = self.model.compute_loss(batch)
                                    val_losses.append(loss)
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps - 1):
                                        break
                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                step_log['val_loss'] = val_loss

                    if (self.epoch % cfg.training.sample_every) == 0:
                        with torch.no_grad():
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            obs_dict = batch['obs']
                            gt_action = batch['action']

                            result = policy.predict_action(obs_dict)
                            pred_action = result['action_pred']
                            mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                            step_log['train_action_mse_error'] = mse.item()
                            del batch
                            del obs_dict
                            del gt_action
                            del result
                            del pred_action
                            del mse

                    if (self.epoch % cfg.training.checkpoint_every) == 0:
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint()
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot()

                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value

                        if topk_manager.monitor_key in metric_dict:
                            topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                            if topk_ckpt_path is not None:
                                self.save_checkpoint(path=topk_ckpt_path)

                    if enable_rollout and async_eval_every > 0 and (self.epoch % async_eval_every) == 0:
                        async_eval_dir = pathlib.Path(self.output_dir).joinpath(
                            async_eval_subdir, f"epoch_{self.epoch:04d}_step_{self.global_step:08d}"
                        )
                        async_eval_ckpt = pathlib.Path(self.output_dir).joinpath(
                            "checkpoints", f"async_eval_epoch_{self.epoch:04d}.ckpt"
                        )
                        self.save_checkpoint(path=async_eval_ckpt, use_thread=False)
                        eval_req = {
                            "cmd": "eval",
                            "ckpt_path": str(async_eval_ckpt),
                            "eval_dir": str(async_eval_dir),
                            "epoch": int(self.epoch),
                            "step": int(self.global_step),
                        }
                        if not eval_inflight:
                            _dispatch_eval_request(eval_req)
                        else:
                            if pending_eval_req is not None:
                                stale_ckpt_path = pending_eval_req.get("ckpt_path", None)
                                new_ckpt_path = eval_req.get("ckpt_path", None)
                                if stale_ckpt_path != new_ckpt_path:
                                    _cleanup_async_eval_ckpt(stale_ckpt_path)
                            pending_eval_req = eval_req
                            async_status = "queue"
                            async_status_epoch = int(eval_req["epoch"])

                    policy.train()
                    wandb_run.log(step_log, step=self.global_step)
                    json_logger.log(step_log)
                    self.global_step += 1
                    self.epoch += 1
        finally:
            if eval_process is not None:
                _shutdown_eval_worker(eval_process, eval_req_q, eval_resp_q)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridAsyncEvalWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
