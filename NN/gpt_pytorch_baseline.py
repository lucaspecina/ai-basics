import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import platform


############################################################
# CONFIGURACION

gpt_config = {
        # "vocab_size": 50257,     # Vocabulary size
        "context_length": 256,  # Input tokens per training example
        "emb_dim": 512,          # Embedding dimension
        "n_heads": 8,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": False,        # Query-key-value bias
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

settings = {
    "learning_rate": 5e-4,
    # "num_epochs": 2,
    "batch_size": 16,
    "weight_decay": 0.1,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 20
}

def get_checkpoint_filename(cfg):
    return f"model_checkpoints/gpt_checkpoint_ctx{cfg['context_length']}_emb{cfg['emb_dim']}_heads{cfg['n_heads']}_layers{cfg['n_layers']}_tokchar.pth"
checkpoint_path = get_checkpoint_filename(gpt_config)
print(checkpoint_path)


############################################################
# DATASET Y DATALOADER

# Tokenizer
class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


# Dataset
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y


# Load and preprocess
torch.manual_seed(settings["seed"])
device = settings["device"]

with open("data/input.txt", 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
gpt_config["vocab_size"] = vocab_size

# Encode full text into tensor
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

# Split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

train_dataset = CharDataset(train_data, gpt_config["context_length"])
val_dataset   = CharDataset(val_data,   gpt_config["context_length"])

use_cuda   = torch.cuda.is_available()
on_windows = (platform.system() == "Windows")

def make_loader(dataset, shuffle):
    # En Windows arrancamos con 0 workers para evitar spawn/pickling issues.
    # Luego, si todo ok, podés probar subir a 2/4/8 y agregar prefetch/persistent.
    if on_windows:
        return DataLoader(
            dataset,
            batch_size=settings["batch_size"],
            shuffle=shuffle,
            num_workers=0,
            pin_memory=use_cuda
        )
    else:
        return DataLoader(
            dataset,
            batch_size=settings["batch_size"],
            shuffle=shuffle,
            num_workers=(4 if use_cuda else 0),
            pin_memory=use_cuda,
            prefetch_factor=(2 if use_cuda else None),
            persistent_workers=(True if use_cuda else False)
        )

train_loader = make_loader(train_dataset, shuffle=True)
val_loader   = make_loader(val_dataset,   shuffle=False)



############################################################
# MODEL COMPONENTS

import torch
from torch.autograd.profiler import record_function

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        with record_function("attn.qkv_proj"):
            keys    = self.W_key(x)
            queries = self.W_query(x)
            values  = self.W_value(x)

        with record_function("attn.reshape_split_heads"):
            keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            values  = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        with record_function("attn.scores_softmax"):
            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

        with record_function("attn.weighted_sum_outproj"):
            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.reshape(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)

        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            nn.GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = cfg["device"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # nn.Sequential arma la secuencia de transformer_blocks
        # Ejecuta una secuencia de submódulos en orden, pasando la salida de uno como input al siguiente
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # nn.LayerNorm: antes usabamos F.LayerNorm (funcion). Ahora usamos una clase que es una layer
        self.final_norm = nn.LayerNorm(cfg["emb_dim"])
        # nn.Linear: es una capa lineal (en vez de un tensor de pesos y un bias)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # in_idx -> input batch (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape
        # input embeddings + positional
        tok_embeds = self.tok_emb(in_idx) 
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x) # Dropout
        x = self.trf_blocks(x) # Transformer blocks
        x = self.final_norm(x) # Layer norm
        logits = self.out_head(x) # Linear layer



############################################################
# UTILS

# Checkpointing functions
def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'gpt_config': gpt_config  # Save config for verification
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=model.device)
        # Verify config matches
        if checkpoint['gpt_config'] != gpt_config:
            print(f"Warning: Config mismatch in checkpoint {filename}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {filename}, starting from epoch {epoch + 1}")
        return epoch
    else:
        print(f"No checkpoint found at {filename}, starting from scratch")
        return -1

def load_model_for_inference(gpt_config, filename, device):
    model = GPTModel(gpt_config).to(device)
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        # Verify config matches
        if checkpoint['gpt_config'] != gpt_config:
            print(f"Warning: Config mismatch in checkpoint {filename}")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {filename} for inference")
    else:
        print(f"No checkpoint found at {filename}, returning untrained model")
    return model


import time
import os
from IPython.display import display, Markdown

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context size
        # E.g., if LLM supports only 5 tokens, and the context size is 10
        # then only the last 5 tokens are used as context
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx

def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=200):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    # Use tokenizer's encode method
    encoded = torch.tensor([tokenizer.encode(start_context)], dtype=torch.long, device=device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=max_new_tokens, context_size=context_size
        )
        # Use tokenizer's decode method
        decoded_text = tokenizer.decode(token_ids[0].tolist())
        # decoded_text = decoded_text.replace("\n", " ")  # Compact print format
    model.train()
    display(Markdown(f"**Generated Output:**\n\n{decoded_text}"))
    return decoded_text


def calc_loss_batch(input_batch, target_batch, model, device):
    # Tensores ya están en device
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i >= num_batches:
            break
        input_batch = input_batch.to(device, non_blocking=True)
        target_batch = target_batch.to(device, non_blocking=True)
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += loss.item()

    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss   = calc_loss_loader(val_loader,   model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# --- Ventana de profiling con PyTorch Profiler -> TensorBoard ---
def run_profiler_window(model, loader, device, steps=10, tb_logdir="./tb_traces/profile"):
    """
    Corre una ventanita (wait=1, warmup=2, active=steps-3) y exporta el trace a TensorBoard.
    Abrí TB -> Profile -> Trace viewer para ver CPU+CUDA kernels y tus record_function().
    """
    from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler
    model.train()
    sched = schedule(wait=1, warmup=2, active=max(1, steps-3), repeat=1)

    it = iter(loader)
    def _one_step():
        inp, tgt = next(it)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        loss = calc_loss_batch(inp, tgt, model, device)
        loss.backward()
        # no optim.step() a propósito, es suficiente para el trace breve
        model.zero_grad(set_to_none=True)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        schedule=sched,
        on_trace_ready=tensorboard_trace_handler(tb_logdir)
    ) as prof:
        total = steps
        for _ in range(total):
            _one_step()
            prof.step()
    print(f"[profiler] Trace guardado en {tb_logdir}")


from torch.utils.data import DataLoader

def get_calibration_batch(dataset, batch_size, device):
    """
    Devuelve 1 batch para calibración usando num_workers=0 (sin multiprocessing).
    Evita errores de spawn/pickling en Windows y surfacing claro de excepciones.
    """
    cal_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    x, _ = next(iter(cal_loader))
    # NO lo movemos a device acá; perf.estimate_memory_budget lo mueve internamente.
    return x



############################################################
# INIT

torch.manual_seed(123)

##############################
# Config prints
##############################
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Using {device}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")

# TF32 en CUDA: suele dar speed-up sin perder calidad en matmuls
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

print()
print(f'Settings:\n{settings}')
print(f'GPT config:\n{gpt_config}')
print(f'Number of batches train (per epoch): {len(train_loader)}')
print(f'Checkpoint path: {checkpoint_path}')

##############################
# Initialize model
##############################

model = GPTModel(gpt_config)
model.to(device)  # no assignment model = model.to(device) necessary for nn.Module classes
optimizer = torch.optim.AdamW(model.parameters(), 
                              lr=settings["learning_rate"], 
                              weight_decay=settings["weight_decay"]
                              )


import os, time
from perfkit import PerfMonitor, PerfConfig

# 1) carpeta absoluta y única para este run
base_tb = os.path.abspath("./tb_traces")
os.makedirs(base_tb, exist_ok=True)
run_dir = os.path.join(base_tb, time.strftime("%Y%m%d-%H%M%S"))  # p.ej. 20250809-152540
os.makedirs(run_dir, exist_ok=True)
print(f"[TB] run_dir = {run_dir}")

# 2) PerfMonitor con TensorBoard encendido hacia run_dir
perf = PerfMonitor(
    model, device,
    PerfConfig(
        log_every=100,
        grad_norm_every=500,
        warmup_steps_ignore=50,
        enable_tensorboard=True,      # <--- ON
        tb_logdir=run_dir,            # <--- usamos el run_dir absoluto
        csv_path="./perf_logs/train_metrics.csv",
        dtype_bytes=2
    )
)


#### Estimacion de memoria (robusto en Windows)
try:
    sample_batch = get_calibration_batch(train_dataset, settings["batch_size"], device)
except Exception as e:
    print(f"[mem] Falló obtener batch de calibración con num_workers=0: {e}")
    raise

mem_report = perf.estimate_memory_budget(
    sample_batch=sample_batch,
    emb_dim=gpt_config["emb_dim"],
    n_layers=gpt_config["n_layers"],
    seq_len=gpt_config["context_length"]
)
print("MemBudget:", {k: v for k, v in mem_report.items() if k != "predict_peak_bytes"})

# Predicción de pico para otra config (opcional)
peak_bytes = mem_report["predict_peak_bytes"](B_=16, T_=512, safety=1.10)
print("Peak ~", peak_bytes / 1e9, "GB")





############################################################
# TRAIN

def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    max_steps=None,
    checkpoint_path=None,
    save_chkpt=False,
    monitor=None  # <-- PerfMonitor opcional (de perfkit)
):
    """
    Entrenamiento single-GPU con instrumentación desacoplada.
    - Si monitor es None, corre sin medir fases/recursos.
    - Evalúa cada `eval_freq` steps usando evaluate_model(...).
    - Mantiene la firma y retornos de tu versión: (train_losses, val_losses, tokens_seen).
    """
    import time
    import torch
    import os
    from contextlib import contextmanager

    # -------- Helpers para no duplicar código cuando monitor = None --------
    @contextmanager
    def _null_cm():
        yield

    class _NullStepCtx:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
        def phase(self, name): return _null_cm()

    def _step_ctx(tokens_in_step):
        if monitor is None:
            return _NullStepCtx()
        return monitor.step(tokens_in_step)

    # ----------------------------------------------------------------------
    train_losses, val_losses, track_tokens = [], [], []
    global_step = -1
    stop_training = False

    # (opcional) cargar checkpoint
    start_epoch = 0
    if checkpoint_path is not None:
        try:
            last_epoch = load_checkpoint(model, optimizer, checkpoint_path)
            if isinstance(last_epoch, int) and last_epoch >= 0:
                start_epoch = last_epoch + 1
        except Exception as e:
            print(f"[warn] no se pudo cargar checkpoint ({e}). iniciando desde 0.")

    model.train()

    for epoch in range(num_epochs):
        for inp_batch, tgt_batch in train_loader:
            global_step += 1

            # tokens del step (entrada)
            B, T = inp_batch.shape
            tokens_in_step = B * T

            with _step_ctx(tokens_in_step) as s:

                # ---- DATA PHASE: mover a device (mide H2D) ----
                with s.phase("data"):
                    inp_batch = inp_batch.to(device, non_blocking=True)
                    tgt_batch = tgt_batch.to(device, non_blocking=True)

                # ---- FORWARD PHASE ----
                with s.phase("forward"):
                    loss = calc_loss_batch(inp_batch, tgt_batch, model, device)

                # ---- BACKWARD PHASE ----
                with s.phase("backward"):
                    optimizer.zero_grad(set_to_none=True)  # limpiar grads antes del backward
                    loss.backward()

                # ---- OPTIMIZER PHASE ----
                with s.phase("optim"):
                    optimizer.step()

            # ---- Intervalo de evaluación ----
            if eval_freq and (global_step % eval_freq == 0):
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    device=device,
                    eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(tokens_in_step if len(track_tokens)==0
                                    else track_tokens[-1] + tokens_in_step)

                # logging del monitor (si está activo)
                if monitor is not None:
                    monitor.log_eval(train_loss, val_loss, optimizer)
                else:
                    lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")
                    print(f"[eval] ep={epoch+1} step={global_step:06d} "
                          f"train={train_loss:.3f} val={val_loss:.3f} lr={lr:.2e}")

            # ---- corte por max_steps ----
            if (max_steps is not None) and (global_step >= max_steps):
                print(f"Reached max steps: {max_steps}")
                stop_training = True
                break

        # ---- checkpoint por epoch ----
        if save_chkpt and checkpoint_path is not None:
            try:
                save_checkpoint(model, optimizer, epoch, checkpoint_path)
            except Exception as e:
                print(f"[warn] no se pudo guardar checkpoint ({e})")

        if stop_training:
            break

    return train_losses, val_losses, track_tokens


############################################################
# RUN

from perfkit import GPUSystemMonitor

# --- parche para que pynvml use el DLL de System32 ---
import os, ctypes, importlib
import pynvml  # importa primero el módulo

def _load_nvml_from_system32():
    dll = r"C:\Windows\System32\nvml.dll"
    if not os.path.isfile(dll):
        raise FileNotFoundError(dll)
    # igual que hace pynvml internamente, pero apuntando a System32
    pynvml.nvmlLib = ctypes.CDLL(dll)

# sobrescribimos la rutina de carga por la nuestra
pynvml._LoadNvmlLibrary = _load_nvml_from_system32

# test
pynvml.nvmlInit()
print("NVML OK (parche System32)")
pynvml.nvmlShutdown()



# tensorboard --logdir ./tb_traces   # <- apuntá SIEMPRE a la carpeta base, no a la del run
tb_writer = getattr(perf, "_tb", None)
sysmon = GPUSystemMonitor(
    tb_writer=tb_writer,
    tb_logdir=run_dir,
    device_index=0,
    period_sec=1.0,
    get_step_fn=lambda: perf.global_step   # <-- OK ahora
)

print("monitor.step ->", type(getattr(perf,"step")))
print("monitor.global_step ->", perf.global_step)
# debe imprimir <class 'method'> y un entero que crece durante el train

import ctypes
ctypes.WinDLL(r"C:\Windows\System32\nvml.dll")  # no debe tirar excepción
from pynvml import nvmlInit, nvmlShutdown
nvmlInit(); print("NVML OK"); nvmlShutdown()


#####################################


# tensorboard --logdir ./tb_trace
# http://localhost:6006/

sysmon.start()

train_losses, val_losses, tokens_seen = train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=1,
    eval_freq=100,
    eval_iter=10,
    max_steps=1000,  # lo que uses
    checkpoint_path=checkpoint_path,
    save_chkpt=True,
    monitor=perf,
)

sysmon.stop()
# MUY IMPORTANTE: flush/close del writer para asegurar event files en disco
try:
    perf.close()
except Exception:
    pass






