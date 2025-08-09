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
    @property
    def step(self) -> int:
        return getattr(self, "_step", 0)

    @property
    def global_step(self) -> int:
        # exposición segura del contador interno
        return getattr(self, "_step", 0)

    def log_eval(self, train_loss: float, val_loss: float, optimizer=None):
        """Llamá en tus intervals de evaluación (cada eval_freq)."""
        lr = _get_lr(optimizer) if optimizer is not None else float("nan")
        mem = self._mem_snapshot()
        msg = (
            f"EVAL | step {self._step:>6d} | train {train_loss:.3f} | "
            f"val {val_loss:.3f} | lr {lr:.2e} | "
            f"mem alloc/res/peak={mem['alloc_gb']:.2f}/{mem['reserved_gb']:.2f}/"
            f"{mem['peak_gb']:.2f} GB"
        )
        print(msg)
        self._tb_write({
            "eval/train_loss": train_loss,
            "eval/val_loss": val_loss,
            "train/lr": lr
        })

    def _log_grad_health_if_needed(self):
        """Muestrea stats de gradientes cada grad_norm_every pasos."""
        try:
            every = getattr(self.cfg, "grad_norm_every", None)
            if not every or self._step % every != 0 or self._step == 0:
                return
            # grad stats
            total_sq = 0.0
            n_elems = 0
            gmax = float("-inf")
            zeros = 0
            nan_inf = 0
            for p in self.model.parameters():
                if p.grad is None: 
                    continue
                g = p.grad.detach()
                # conteos
                zeros += torch.count_nonzero(g == 0).item()
                nan_inf += torch.count_nonzero(~torch.isfinite(g)).item()
                # normas
                total_sq += torch.sum(g[g.isfinite()]**2).item()
                n_elems += g.numel()
                gmax = max(gmax, float(torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()))
            grad_norm = (total_sq ** 0.5) if total_sq > 0 else 0.0
            zero_frac = (zeros / n_elems) if n_elems > 0 else 0.0

            self._tb_write({
                "health/grad_norm": grad_norm,
                "health/grad_max": gmax if gmax != float("-inf") else 0.0,
                "health/zero_grad_frac": zero_frac,
                "health/grad_nan_inf": float(nan_inf),
            })
        except Exception:
            pass  # jamás tirar abajo el entrenamiento por telemetría


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

        toks_s_inst = (self._tokens_since_last / (self._time_since_last_ms/1e3)) if self._time_since_last_ms > 1e-6 else 0.0
        if self._step > self.cfg.warmup_steps_ignore:
            self._toks_s_avg_num += self._tokens_since_last
            self._toks_s_avg_den += (self._time_since_last_ms/1e3)
        toks_s_avg = (self._toks_s_avg_num / self._toks_s_avg_den) if self._toks_s_avg_den > 0 else 0.0

        mem = self._mem_snapshot()

        # ---- proporciones por fase (suman 1) y grouping para TB ----
        total = max(1e-6, step_ms)
        d = phases_ms.get('data', 0.0)
        f = phases_ms.get('forward', 0.0)
        b = phases_ms.get('backward', 0.0)
        o = phases_ms.get('optim', 0.0)

        # carta única "time/share" con 4 series
        if self._tb is not None:
            try:
                self._tb.add_scalars(
                    "time/share",
                    {"data": d/total, "fwd": f/total, "bwd": b/total, "opt": o/total},
                    self._step
                )
            except Exception:
                pass

        # (además dejamos las ms crudas como antes)
        self._tb_write({
            "time/data_ms": d, "time/forward_ms": f, "time/backward_ms": b, "time/optim_ms": o,
            # ... y el resto de tu logging (ms_step, toks_s, mem/*) ...
        })

        # ---- salud de gradientes cada N pasos ----
        self._log_grad_health_if_needed()


        # ---- pretty print: porcentajes por fase y unidades legibles ----
        total = max(1e-6, step_ms)
        d = phases_ms.get('data', 0.0)
        f = phases_ms.get('forward', 0.0)
        b = phases_ms.get('backward', 0.0)
        o = phases_ms.get('optim', 0.0)

        def pct(x): return 100.0 * x / total
        def kfmt(x):  # 12345 -> "12.3k"
            return f"{x/1000:.1f}k" if x >= 10_000 else f"{int(x)}"

        if (self._step % self.cfg.log_every) == 0:
            line = (
                f"STEP {self._step:>6d} | {step_ms:>6.2f} ms/step | "
                f"{kfmt(toks_s_inst)} tok/s (avg {kfmt(toks_s_avg)}) | "
                f"data {d:>5.2f} ms ({pct(d):>4.1f}%) | "
                f"fwd {f:>5.2f} ms ({pct(f):>4.1f}%) | "
                f"bwd {b:>5.2f} ms ({pct(b):>4.1f}%) | "
                f"opt {o:>5.2f} ms ({pct(o):>4.1f}%) | "
                f"mem alloc/res/peak={mem['alloc_gb']:.2f}/{mem['reserved_gb']:.2f}/"
                f"{mem['peak_gb']:.2f} GB | headroom={mem['headroom_gb']:.2f} GB"
            )
            print(line)

        self._csv_write({
            "step": self._step,
            "ms_step": step_ms,
            "toks_s_inst": toks_s_inst,
            "toks_s_avg": toks_s_avg,
            "data_ms": d, "fwd_ms": f, "bwd_ms": b, "opt_ms": o,
            **{k: v for k, v in mem.items() if k.endswith("_gb")}
        })
        self._tb_write({
            "train/ms_step": step_ms,
            "train/toks_s_inst": toks_s_inst,
            "train/toks_s_avg": toks_s_avg,
            "time/data_ms": d, "time/forward_ms": f, "time/backward_ms": b, "time/optim_ms": o,
            "mem/alloc_gb": mem["alloc_gb"], "mem/reserved_gb": mem["reserved_gb"],
            "mem/peak_gb": mem["peak_gb"], "mem/headroom_gb": mem["headroom_gb"],
        })

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

    def close(self):
        """Flush + close del SummaryWriter (si existe). Llamar al final del run."""
        try:
            if self._tb is not None:
                self._tb.flush()
                self._tb.close()
        except Exception:
            pass


    class _StepCtx:
        def __init__(self, mon, tokens):
            self.mon = mon
            self.tokens = tokens
            self.t0 = None
            self._phases = {"data":0.0, "forward":0.0, "backward":0.0, "optim":0.0}
            self._cur = None
            self._tphase = None
        def __enter__(self):
            self.t0 = _now_ms()
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = _now_ms() - self.t0
            self.mon._finalize_step(dt, self._phases, self.tokens)
        def phase(self, name):
            # sub-contexto por fase
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                t = _now_ms()
                yield
                self._phases[name] = self._phases.get(name,0.0) + (_now_ms()-t)
            return _cm()

    def step(self, tokens_in_step:int):
        """Usado por train(): with monitor.step(tokens) as s: ..."""
        return self._StepCtx(self, tokens_in_step)





# --- GPU Telemetría con NVML o fallback a nvidia-smi ---
import threading, time, shutil, os, sys, subprocess
try:
    import pynvml
    _has_nvml = True
except Exception:
    _has_nvml = False

class GPUSystemMonitor:
    def __init__(self, tb_writer=None, tb_logdir="./tb_traces", device_index=0, period_sec=1.0, get_step_fn=None):
        from torch.utils.tensorboard import SummaryWriter
        self.writer = tb_writer or SummaryWriter(tb_logdir)
        self.dev_index = int(device_index)
        self.period = float(period_sec)
        self._get_step = get_step_fn
        self._local_step = 0
        self._stop = threading.Event()
        self._t = None
        self._mode = None  # "nvml" o "smi"

    def _step(self):
        if callable(self._get_step):
            s = int(self._get_step())
            if s < self._local_step:
                s = self._local_step
            self._local_step = s
        else:
            self._local_step += 1
        return self._local_step

    def _ensure_nvml_path(self):
        if not sys.platform.startswith("win"):
            return
        # agrega dirs típicos + donde esté nvidia-smi
        candidates = [
            r"C:\Windows\System32",
            r"C:\Program Files\NVIDIA Corporation\NVSMI",
        ]
        smi = shutil.which("nvidia-smi")
        if smi:
            candidates.append(os.path.dirname(smi))
        for p in candidates:
            try:
                if os.path.isdir(p):
                    os.add_dll_directory(p)
            except Exception:
                pass

    def start(self):
        import torch
        if not torch.cuda.is_available():
            print("[sysmon] CUDA no disponible; sys/* off")
            return
        # intenta NVML primero
        if _has_nvml:
            self._ensure_nvml_path()
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.dev_index)
                self._mode = "nvml"
                self._t = threading.Thread(target=self._loop_nvml, daemon=True); self._t.start()
                print("[sysmon] NVML telemetry ON")
                return
            except Exception as e:
                print(f"[sysmon] NVML init falló: {e}. Probando fallback nvidia-smi...")
        # fallback: nvidia-smi
        if shutil.which("nvidia-smi"):
            self._mode = "smi"
            self._t = threading.Thread(target=self._loop_smi, daemon=True); self._t.start()
            print("[sysmon] nvidia-smi telemetry ON (fallback)")
        else:
            print("[sysmon] ni NVML ni nvidia-smi disponibles; sys/* off")

    def stop(self):
        if self._t is None: return
        self._stop.set(); self._t.join(timeout=2)
        if self._mode == "nvml":
            try: pynvml.nvmlShutdown()
            except Exception: pass
        print("[sysmon] telemetry OFF")

    # --- NVML loop ---
    def _loop_nvml(self):
        while not self._stop.is_set():
            s = self._step()
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                try:   power_w = pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle)/1000.0
                except Exception: power_w = float("nan")
                temp_c = pynvml.nvmlDeviceGetTemperature(self._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    sm_clock  = pynvml.nvmlDeviceGetClockInfo(self._nvml_handle, pynvml.NVML_CLOCK_SM)
                    mem_clock = pynvml.nvmlDeviceGetClockInfo(self._nvml_handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    sm_clock, mem_clock = float("nan"), float("nan")
                self.writer.add_scalars("sys/util", {"gpu": util.gpu, "mem": util.memory}, s)
                self.writer.add_scalars("sys/clocks_mhz", {"sm": sm_clock, "mem": mem_clock}, s)
                self.writer.add_scalars("sys/thermals_power", {"temp_c": temp_c, "power_w": power_w}, s)
                self.writer.add_scalar("sys/vram_used_gb", mem.used/(1024**3), s)
            except Exception:
                pass
            time.sleep(self.period)

    # --- nvidia-smi loop (fallback) ---
    def _loop_smi(self):
        # query en CSV, sin unidades
        query = "utilization.gpu,utilization.memory,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used"
        cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits", "-i", str(self.dev_index)]
        while not self._stop.is_set():
            s = self._step()
            try:
                out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
                # ej: "65, 55, 2250, 8750, 67, 110.5, 3121"
                parts = [p.strip() for p in out.split(",")]
                gpu, memu, sm, mem, temp, power, used = parts
                self.writer.add_scalars("sys/util", {"gpu": float(gpu), "mem": float(memu)}, s)
                self.writer.add_scalars("sys/clocks_mhz", {"sm": float(sm), "mem": float(mem)}, s)
                self.writer.add_scalars("sys/thermals_power", {"temp_c": float(temp), "power_w": float(power)}, s)
                self.writer.add_scalar("sys/vram_used_gb", float(used)/(1024), s)  # MiB -> GiB aprox
            except Exception:
                pass
            time.sleep(self.period)




def debug_tensorboard_files(tb_base="./tb_traces"):
    import glob, os
    tb_base = os.path.abspath(tb_base)
    runs = glob.glob(os.path.join(tb_base, "*"))
    print(f"[TB DEBUG] base={tb_base}")
    if not runs:
        print("  (no hay subcarpetas de runs)")
    for r in runs:
        ev = glob.glob(os.path.join(r, "events.out.tfevents.*"))
        print(f"  run={os.path.basename(r)}  event_files={len(ev)}")
        for p in ev[:3]:
            print(f"    - {os.path.basename(p)}")