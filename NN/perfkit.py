# perfkit.py
# Kit minimalista de profiling/monitor single-GPU para entrenos tipo nanoGPT.
# Sin dependencias externas. Opcional: TensorBoard si está instalado.

from __future__ import annotations
import time, json, math, contextlib, os
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, Iterable

try:
    import torch
    _has_torch = True
except Exception:
    _has_torch = False

# ----------------------------
# Utilidades básicas
# ----------------------------

def _now_ms() -> float:
    return time.perf_counter() * 1e3

def _round(x, n=2):
    return float(f"{x:.{n}f}")

def _device_total_mem(device) -> int:
    if not _has_torch or not torch.cuda.is_available(): return 0
    props = torch.cuda.get_device_properties(device)
    return getattr(props, "total_memory", 0)

def _bytes_to_gb(x: int | float) -> float:
    return x / (1024**3)

def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())

def _get_lr(optimizer) -> float:
    return optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")

def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None: 
            continue
        param_norm = p.grad.data.norm(2)
        total += param_norm.item() ** 2
    return math.sqrt(total)

# ----------------------------
# Configuración del monitor
# ----------------------------

@dataclass
class PerfConfig:
    log_every: int = 100                  # pasos entre logs “grandes”
    grad_norm_every: int = 500            # cada cuántos steps computar grad_norm
    warmup_steps_ignore: int = 50         # steps a ignorar en promedios
    enable_tensorboard: bool = False
    tb_logdir: str = "./tb_traces"
    csv_path: Optional[str] = None        # si querés CSV con métricas por step
    estimate_mem_every: int = 1000        # re-calibrar peak mem cada tanto (opcional)
    # precisión para presupuestar (solo afecta bytes/elem de activaciones):
    dtype_bytes: int = 2                  # 2 para fp16/bf16, 4 para fp32

# ----------------------------
# Temporizador de fases
# ----------------------------

class _CudaTimer:
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end   = torch.cuda.Event(enable_timing=True)
        self._t0 = 0.0

    def start(self):
        if self.use_cuda:
            torch.cuda.synchronize()
            self._start.record()
        else:
            self._t0 = _now_ms()

    def stop_ms(self) -> float:
        if self.use_cuda:
            self._end.record()
            torch.cuda.synchronize()
            return self._start.elapsed_time(self._end)  # ms
        else:
            return _now_ms() - self._t0

@contextlib.contextmanager
def _timed_section(timers_dict: Dict[str, float], name: str, use_cuda: bool):
    t = _CudaTimer(use_cuda)
    t.start()
    try:
        yield
    finally:
        dur = t.stop_ms()
        timers_dict[name] = timers_dict.get(name, 0.0) + dur

# ----------------------------
# Contexto de step (agrupa fases)
# ----------------------------

class StepContext:
    def __init__(self, monitor: "PerfMonitor", tokens_in_step: int):
        self.monitor = monitor
        self.tokens_in_step = tokens_in_step
        self._phase_ms: Dict[str, float] = {}
        self._step_timer = _CudaTimer(monitor._use_cuda)

    def __enter__(self):
        self._step_timer.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        step_ms = self._step_timer.stop_ms()
        self.monitor._finalize_step(step_ms, self._phase_ms, self.tokens_in_step)

    @contextlib.contextmanager
    def phase(self, name: str):
        with _timed_section(self._phase_ms, name, self.monitor._use_cuda):
            yield

# ----------------------------
# Monitor principal
# ----------------------------

class PerfMonitor:
    def __init__(self, model, device, cfg: PerfConfig):
        self.model = model
        self.device = device
        self.cfg = cfg
        self._use_cuda = (_has_torch and isinstance(device, torch.device) 
                          and device.type == "cuda" and torch.cuda.is_available())
        self._step = 0
        self._tokens_total = 0
        self._tokens_since_last = 0
        self._time_since_last_ms = 0.0
        self._last_tick_ms = _now_ms()
        self._toks_s_avg_num = 0.0
        self._toks_s_avg_den = 0.0
        self._headers_written = False
        self._k_act: Optional[float] = None
        self._static_mem_bytes: Optional[int] = None
        self._tb = None
        if cfg.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                os.makedirs(cfg.tb_logdir, exist_ok=True)
                self._tb = SummaryWriter(cfg.tb_logdir)
            except Exception:
                self._tb = None  # si no está instalado, seguimos

        if cfg.csv_path:
            os.makedirs(os.path.dirname(cfg.csv_path) or ".", exist_ok=True)

        # cache: total vram
        self._vram_total = _device_total_mem(device)

    # ---- API pública ----

    def step(self, tokens_in_step: int) -> StepContext:
        """Contexto por-step. Usar:
           with monitor.step(B*T) as s:
               with s.phase('data'): ...
               with s.phase('forward'): ...
               with s.phase('backward'): ...
               with s.phase('optim'): ...
        """
        return StepContext(self, tokens_in_step)

    def log_eval(self, train_loss: float, val_loss: float, optimizer=None):
        """Llamá en tus intervals de evaluación (cada eval_freq)."""
        lr = _get_lr(optimizer) if optimizer is not None else float("nan")
        mem = self._mem_snapshot()
        msg = (f"[eval] step={self._step} train={train_loss:.3f} val={val_loss:.3f} "
               f"lr={lr:.2e} mem(alloc/res/peak)={mem['alloc_gb']:.2f}/"
               f"{mem['reserved_gb']:.2f}/{mem['peak_gb']:.2f}GB")
        print(msg)
        self._tb_write({"eval/train_loss":train_loss, "eval/val_loss":val_loss, "train/lr":lr})

    def estimate_memory_budget(self, sample_batch, emb_dim: int, n_layers: int, seq_len: int) -> Dict[str, Any]:
        """Calibra k de activaciones con 1 forward y devuelve funciones de predicción."""
        if not self._use_cuda:
            return {"note":"solo calibra en CUDA", "k_act": None}

        # 1) estática a partir de P
        P = _count_params(self.model)
        static_bytes = 16 * P  # AdamW + fp16/bf16
        self._static_mem_bytes = static_bytes

        # 2) medir delta de activaciones en un forward
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        alloc0 = torch.cuda.memory_allocated(self.device)

        self.model.eval()
        with torch.no_grad():
            _ = self.model(sample_batch.to(self.device))
        self.model.train()

        alloc1 = torch.cuda.memory_allocated(self.device)
        delta = max(0, alloc1 - alloc0)

        # 3) resolver k en: delta ≈ k * B*T*H*L * bytes/elem
        B = sample_batch.shape[0]
        T = sample_batch.shape[1]
        H = emb_dim
        L = n_layers
        bytes_elem = self.cfg.dtype_bytes
        denom = max(1, B*T*H*L*bytes_elem)
        k_act = delta / denom
        self._k_act = k_act

        def predict_peak_bytes(B_: int, T_: int, H_: int = H, L_: int = L, safety: float = 1.10):
            act = k_act * B_ * T_ * H_ * L_ * bytes_elem
            peak = static_bytes + act
            # sumar overhead de reservas/fragmentación (simple safety factor)
            return int(peak * safety)

        report = {
            "params": P,
            "static_mem_gb": _bytes_to_gb(static_bytes),
            "k_act": k_act,
            "predict_peak_bytes": predict_peak_bytes,
            "vram_total_gb": _bytes_to_gb(self._vram_total),
        }
        return report

    # ---- Internals ----

    def _finalize_step(self, step_ms: float, phases_ms: Dict[str, float], tokens_in_step: int):
        self._step += 1
        self._tokens_total += tokens_in_step
        self._tokens_since_last += tokens_in_step
        now = _now_ms()
        self._time_since_last_ms += (now - self._last_tick_ms)
        self._last_tick_ms = now

        toks_s_inst = 0.0
        if self._time_since_last_ms > 1e-6:
            toks_s_inst = (self._tokens_since_last / (self._time_since_last_ms/1e3))
        # actualizar promedio excluyendo warmup
        if self._step > self.cfg.warmup_steps_ignore:
            self._toks_s_avg_num += self._tokens_since_last
            self._toks_s_avg_den += (self._time_since_last_ms/1e3)
        toks_s_avg = (self._toks_s_avg_num / self._toks_s_avg_den) if self._toks_s_avg_den>0 else 0.0

        # snapshot de memoria
        mem = self._mem_snapshot()

        # logging “grande”
        if (self._step % self.cfg.log_every) == 0:
            msg = (f"step={self._step:06d} ms/step={_round(step_ms)} "
                   f"toks/s={int(toks_s_inst)} avg={int(toks_s_avg)} | "
                   f"data={_round(phases_ms.get('data',0))} "
                   f"fwd={_round(phases_ms.get('forward',0))} "
                   f"bwd={_round(phases_ms.get('backward',0))} "
                   f"opt={_round(phases_ms.get('optim',0))} | "
                   f"mem alloc/res/peak={mem['alloc_gb']:.2f}/"
                   f"{mem['reserved_gb']:.2f}/{mem['peak_gb']:.2f}GB "
                   f"headroom={mem['headroom_gb']:.2f}GB")
            print(msg)

        # CSV / TB
        self._csv_write({
            "step": self._step,
            "ms_step": step_ms,
            "toks_s_inst": toks_s_inst,
            "toks_s_avg": toks_s_avg,
            "data_ms": phases_ms.get("data",0.0),
            "fwd_ms": phases_ms.get("forward",0.0),
            "bwd_ms": phases_ms.get("backward",0.0),
            "opt_ms": phases_ms.get("optim",0.0),
            **{k:v for k,v in mem.items() if k.endswith("_gb")}
        })
        self._tb_write({
            "train/ms_step": step_ms,
            "train/toks_s_inst": toks_s_inst,
            "train/toks_s_avg": toks_s_avg,
            "time/data_ms": phases_ms.get("data",0.0),
            "time/forward_ms": phases_ms.get("forward",0.0),
            "time/backward_ms": phases_ms.get("backward",0.0),
            "time/optim_ms": phases_ms.get("optim",0.0),
            "mem/alloc_gb": mem["alloc_gb"],
            "mem/reserved_gb": mem["reserved_gb"],
            "mem/peak_gb": mem["peak_gb"],
            "mem/headroom_gb": mem["headroom_gb"],
        })

        # reset ventana
        self._tokens_since_last = 0
        self._time_since_last_ms = 0.0

    def _mem_snapshot(self) -> Dict[str, float]:
        if not self._use_cuda:
            return {"alloc_gb":0.0, "reserved_gb":0.0, "peak_gb":0.0, "headroom_gb":0.0}
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev)
        reserved = torch.cuda.memory_reserved(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        headroom = max(0, self._vram_total - reserved)
        return {
            "alloc_gb": _bytes_to_gb(alloc),
            "reserved_gb": _bytes_to_gb(reserved),
            "peak_gb": _bytes_to_gb(peak),
            "headroom_gb": _bytes_to_gb(headroom),
        }

    def _csv_write(self, row: Dict[str, Any]):
        if not self.cfg.csv_path:
            return
        # escribimos cabecera la primera vez
        if not self._headers_written:
            with open(self.cfg.csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(row.keys()) + "\n")
            self._headers_written = True
        with open(self.cfg.csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")

    def _tb_write(self, scalars: Dict[str, float]):
        if self._tb is None:
            return
        for k, v in scalars.items():
            try:
                self._tb.add_scalar(k, v, self._step)
            except Exception:
                pass



# --- GPU Telemetría con NVML -> TensorBoard (CUDA) ---
# Requiere: pip install nvidia-ml-py3
import threading, time
try:
    import pynvml
    _has_nvml = True
except Exception:
    _has_nvml = False

class GPUSystemMonitor:
    def __init__(self, tb_logdir="./tb_traces", device_index=0, period_sec=1.0):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(tb_logdir)
        self.dev_index = device_index
        self.period = period_sec
        self._stop = threading.Event()
        self._t = None

    def start(self):
        if not _has_nvml or not torch.cuda.is_available():
            print("[sysmon] NVML no disponible; no se logueará sys/*")
            return
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.dev_index)
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        print("[sysmon] NVML telemetry ON")

    def stop(self):
        if not _has_nvml or self._t is None:
            return
        self._stop.set()
        self._t.join(timeout=2)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
        print("[sysmon] NVML telemetry OFF")

    def _loop(self):
        step = 0
        while not self._stop.is_set():
            try:
                util  = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem   = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                power = None
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0 # W
                except Exception:
                    power = float("nan")
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)

                # clocks
                try:
                    sm_clock  = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    sm_clock, mem_clock = float("nan"), float("nan")

                # pstate
                try:
                    pstate = pynvml.nvmlDeviceGetPerformanceState(self.handle)
                except Exception:
                    pstate = -1

                self.writer.add_scalar("sys/gpu_util", util.gpu, step)
                self.writer.add_scalar("sys/mem_util", util.memory, step)
                self.writer.add_scalar("sys/vram_used_gb", mem.used / (1024**3), step)
                self.writer.add_scalar("sys/power_w", power, step)
                self.writer.add_scalar("sys/temp_c", temp, step)
                self.writer.add_scalar("sys/sm_clock_mhz", sm_clock, step)
                self.writer.add_scalar("sys/mem_clock_mhz", mem_clock, step)
                self.writer.add_scalar("sys/pstate", pstate, step)
                step += 1
            except Exception:
                pass
            time.sleep(self.period)
