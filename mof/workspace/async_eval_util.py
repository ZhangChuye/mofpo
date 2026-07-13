"""Async rollout-eval worker helpers for TrainDiffusionUnetHybridAsyncEvalWorkspace.
Extracted verbatim from the removed DDP workspace; the paper launches eval via
run_async_eval.sh, which overrides the workspace to the async-eval variant."""
import os
import traceback
import signal
import hydra
import torch
from omegaconf import OmegaConf
import random
import numpy as np
import dill


def _sanitize_eval_metrics(runner_log: dict):
    metrics = {}
    for key, value in runner_log.items():
        if isinstance(value, (int, float, np.integer, np.floating)):
            metrics[key] = float(value)
    return metrics


def _extract_eval_video_paths(runner_log: dict):
    video_paths = {}
    for key, value in runner_log.items():
        if "sim_video_" not in key:
            continue
        path = None
        if isinstance(value, str):
            path = value
        elif hasattr(value, "_path"):
            path = getattr(value, "_path")
        elif hasattr(value, "path"):
            path = getattr(value, "path")
        if path is not None:
            path = str(path)
            if os.path.isfile(path):
                video_paths[key] = path
    return video_paths


def _resolve_async_eval_device(training_cfg):
    async_eval_device = str(training_cfg.get("async_eval_device", "cuda:0"))
    async_eval_cuda_visible_devices = training_cfg.get("async_eval_cuda_visible_devices", None)
    async_eval_gpu_id = training_cfg.get("async_eval_gpu_id", None)
    if async_eval_gpu_id is not None:
        async_eval_cuda_visible_devices = str(async_eval_gpu_id)
        async_eval_device = "cuda:0"
    return async_eval_device, async_eval_cuda_visible_devices


def _instantiate_runner(cfg, output_dir, disable_tqdm=False, use_ddp_async_env=False):
    return hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir,
        disable_tqdm=bool(disable_tqdm),
        use_ddp_async_env=bool(use_ddp_async_env),
    )


def _shutdown_eval_worker(eval_process, eval_req_q, eval_resp_q):
    if eval_process is None:
        return
    try:
        if eval_req_q is not None:
            eval_req_q.put({"cmd": "stop"})
    except Exception:
        pass
    eval_process.join(timeout=10.0)
    if eval_process.is_alive():
        eval_process.terminate()
    _kill_process_group(eval_process.pid, sig=signal.SIGTERM)
    eval_process.join(timeout=2.0)
    if eval_process.is_alive():
        _kill_process_group(eval_process.pid, sig=signal.SIGKILL)
        eval_process.terminate()
    if eval_req_q is not None:
        eval_req_q.close()
        eval_req_q.join_thread()
    if eval_resp_q is not None:
        eval_resp_q.close()
        eval_resp_q.join_thread()


def _set_seed(seed, deterministic=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _kill_process_group(pid: int, sig):
    if pid is None:
        return
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def _async_eval_worker_main(
    cfg_container,
    bootstrap_ckpt_path,
    eval_output_root,
    eval_device,
    eval_cuda_visible_devices,
    disable_eval_tqdm,
    req_q,
    resp_q
):
    try:
        # Make the eval worker the leader of its own process group so cleanup
        # can tear down spawned env subprocesses on failure.
        try:
            os.setsid()
        except OSError:
            pass
        if eval_cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(eval_cuda_visible_devices)
        eval_torch_device = torch.device(eval_device)
        cfg = OmegaConf.create(cfg_container)
        with open(bootstrap_ckpt_path, "rb") as f:
            payload = torch.load(f, pickle_module=dill, map_location="cpu")
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=eval_output_root)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        policy.to(eval_torch_device)
        policy.eval()

        env_runner = _instantiate_runner(
            cfg=cfg,
            output_dir=eval_output_root,
            disable_tqdm=disable_eval_tqdm,
            use_ddp_async_env=True,
        )
        state_dict_key = "ema_model" if cfg.training.use_ema else "model"

        while True:
            req = req_q.get()
            if req is None or req.get("cmd") == "stop":
                break
            if req.get("cmd") != "eval":
                continue

            ckpt_path = req["ckpt_path"]
            eval_dir = req["eval_dir"]
            epoch = int(req["epoch"])
            step = int(req["step"])
            try:
                with open(ckpt_path, "rb") as f:
                    payload = torch.load(f, pickle_module=dill, map_location="cpu")
                policy.load_state_dict(payload['state_dicts'][state_dict_key])
                # Keep policy and normalizer buffers pinned to eval device.
                policy.to(eval_torch_device)
                policy.eval()
                env_runner.output_dir = eval_dir
                # Reset RNG before each eval so policy inference
                # (torch.randn in DDIM sampling) is deterministic.
                _set_seed(int(cfg.training.seed))
                runner_log = env_runner.run(policy)
                resp_q.put({
                    "status": "ok",
                    "epoch": epoch,
                    "step": step,
                    "ckpt_path": ckpt_path,
                    "eval_dir": eval_dir,
                    "metrics": _sanitize_eval_metrics(runner_log),
                    "video_paths": _extract_eval_video_paths(runner_log),
                })
            except Exception:
                resp_q.put({
                    "status": "error",
                    "epoch": epoch,
                    "step": step,
                    "ckpt_path": ckpt_path,
                    "eval_dir": eval_dir,
                    "traceback": traceback.format_exc(),
                })
    except Exception:
        resp_q.put({
            "status": "fatal",
            "traceback": traceback.format_exc(),
        })

