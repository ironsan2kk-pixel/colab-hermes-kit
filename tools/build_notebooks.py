#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Генератор ноутбуков Colab -> Hermes Kit (LLM_Server.ipynb, ComfyUI_Server.ipynb).
Запуск: python3 tools/build_notebooks.py
Проверяет: валидный JSON + компиляцию каждой code-ячейки.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells):
    return {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


# =====================================================================
#  LLM_Server.ipynb
# =====================================================================

LLM_MD_INTRO = r"""
# 🧠 Colab → Hermes: LLM-сервер (vLLM)

Этот ноутбук поднимает на видеокарте Google Colab **настоящий LLM-сервер** (открытая модель Qwen2.5), делает для него публичную ссылку и сам себя проверяет. Ссылку потом подключаем к Hermes — и модель из Colab доступна как обычная модель.

**Как запустить:**
1. Меню **Среда выполнения → Сменить тип среды выполнения → аппаратный ускоритель: T4 GPU** → Сохранить.
2. Меню **Среда выполнения → Выполнить все** (или Ctrl+F9). Появится предупреждение, что ноутбук «не от Google» — жми **Run anyway**.
3. Первый запуск: 5–10 минут (ставятся библиотеки, качается модель ~5 ГБ). Дождись в конце блока **«СКОПИРУЙ ЭТО…»** и пришли его в чат Hermes.

**Что важно знать:**
- Бесплатная Colab иногда не даёт GPU. Если в ячейке «2. Проверка GPU» написано «НЕ НАЙДЕН» — проверь шаг 1.
- Сессия живёт до ~12 часов (обычно меньше). Умерла сессия → просто «Выполнить все» заново, ссылка будет новая.
- Ноутбук — на T4 (16 ГБ) крутится модель 7B. На платных GPU (L4 / A100) ноутбук сам выберет модель крупнее.
- Одновременно только один GPU-ноутбук: либо этот, либо «картиночный». Оба сразу — только на платных GPU.

**Пресеты моделей (меняются в ячейке 1):**

| пресет | модель | для кого |
|---|---|---|
| `auto` | 7B / 14B / 32B по видеопамяти | по умолчанию |
| `7b` | Qwen2.5-7B-Instruct-AWQ | T4 (бесплатная) |
| `14b` | Qwen2.5-14B-Instruct-AWQ | L4 / A100 |
| `32b` | Qwen2.5-32B-Instruct-AWQ | A100 40GB |
| `vision` | Qwen2.5-VL-7B (понимает картинки) | T4 |

Если vLLM категорически не заводится на T4 — в конце есть запасная ячейка **«АЛЬТЕРНАТИВА: Ollama»**, выполни только её.
"""

LLM_CELL_SETTINGS = r"""
# ==== НАСТРОЙКИ (обычно менять не нужно) ====
MODEL_PRESET = "auto"   # "auto" | "7b" | "14b" | "32b" | "vision"
MAX_MODEL_LEN = 32768   # контекст в токенах (снизь до 16384, если совсем мало видеопамяти)
PORT = 8000             # локальный порт сервера

print("Настройки загружены. Пресет модели:", MODEL_PRESET)
"""

LLM_CELL_GPU = r"""
import subprocess, re

gpu_info = subprocess.run(
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    shell=True, capture_output=True, text=True,
).stdout.strip()
print("GPU:", gpu_info if gpu_info else "НЕ НАЙДЕН")
if not gpu_info:
    raise RuntimeError(
        "Нет GPU. Сделай: Среда выполнения → Сменить тип среды выполнения → T4 GPU → Сохранить, "
        "затем: Среда выполнения → Выполнить все."
    )

m = re.search(r"([0-9]+)\s*MiB", gpu_info)
VRAM_GB = (int(m.group(1)) / 1024.0) if m else 16.0

PRESETS = {
    "7b":     ("Qwen/Qwen2.5-7B-Instruct-AWQ",    32768),
    "14b":    ("Qwen/Qwen2.5-14B-Instruct-AWQ",   32768),
    "32b":    ("Qwen/Qwen2.5-32B-Instruct-AWQ",   32768),
    "vision": ("Qwen/Qwen2.5-VL-7B-Instruct-AWQ", 32768),
}
if MODEL_PRESET == "auto":
    _preset = "7b" if VRAM_GB < 20 else ("14b" if VRAM_GB < 38 else "32b")
else:
    _preset = MODEL_PRESET

MODEL_ID, _native_len = PRESETS[_preset]
MAX_LEN = min(MAX_MODEL_LEN, _native_len)
MODEL_NAME = "colab-llm"
print("Видеопамять ≈ %.0f ГБ → пресет '%s' → %s (контекст %d)" % (VRAM_GB, _preset, MODEL_ID, MAX_LEN))
"""

LLM_CELL_INSTALL = r"""
import subprocess, sys

print("Ставлю vLLM (2–5 минут, один раз за сессию). Подожди, ничего не нажимай...")
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "vllm", "huggingface_hub[hf_transfer]"],
    capture_output=True, text=True, timeout=1800,
)
if r.returncode == 0:
    print("✅ vLLM установлен")
else:
    print("❌ pip завершился с ошибкой. Хвост вывода:")
    print((r.stdout or "")[-1500:])
    print((r.stderr or "")[-1500:])
    raise RuntimeError("Не удалось установить vLLM — попробуй ячейку «АЛЬТЕРНАТИВА: Ollama» в конце ноутбука.")
"""

LLM_CELL_START = r"""
import os, sys, time, subprocess, secrets

API_KEY = globals().get("API_KEY") or ("sk-colab-" + secrets.token_hex(12))
os.makedirs("/content/logs", exist_ok=True)
VLLM_LOG = "/content/logs/vllm.log"

env = os.environ.copy()
env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
env["VLLM_NO_USAGE_STATS"] = "1"

cmd = [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server",
    "--model", MODEL_ID,
    "--served-model-name", MODEL_NAME,
    "--dtype", "float16",
    "--max-model-len", str(MAX_LEN),
    "--gpu-memory-utilization", "0.90",
    "--api-key", API_KEY,
    "--enable-auto-tool-choice",
    "--tool-call-parser", "hermes",
    "--enforce-eager",
    "--port", str(PORT),
]
if _preset == "vision":
    cmd += ["--limit-mm-per-prompt", "image=4"]

print("Запускаю сервер. Первый раз качается модель (~5 ГБ) — это 2–6 минут. Жди...")
with open(VLLM_LOG, "w") as lf:
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

deadline = time.time() + 1200
ok = False
while time.time() < deadline:
    log = open(VLLM_LOG, errors="ignore").read()
    if ("Application startup complete" in log) or ("Uvicorn running on" in log):
        ok = True
        break
    if proc.poll() is not None:
        break
    time.sleep(5)

if ok:
    print("✅ LLM-сервер запущен локально на порту", PORT)
else:
    print("❌ Сервер не стартовал. Последние строки лога:")
    print(open(VLLM_LOG, errors="ignore").read()[-2500:])
    raise RuntimeError(
        "vLLM не запустился. Варианты: 1) выполни ячейку «АЛЬТЕРНАТИВА: Ollama» в конце; "
        "2) Среда выполнения → Перезапустить сессию → Выполнить все ещё раз."
    )
"""

LLM_CELL_TUNNEL = r"""
import subprocess, os, re, time, json, urllib.request

# 1) cloudflared — быстрый туннель к локальному порту (скачивается один раз)
if not os.path.exists("/content/cloudflared"):
    subprocess.run(
        "wget -q -O /content/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
        "&& chmod +x /content/cloudflared",
        shell=True, check=True,
    )

os.makedirs("/content/logs", exist_ok=True)
TUN_LOG = "/content/logs/cloudflared.log"
with open(TUN_LOG, "w") as lf:
    subprocess.Popen(
        ["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate"],
        stdout=lf, stderr=subprocess.STDOUT,
    )

url = None
for _ in range(60):
    mm = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(TUN_LOG, errors="ignore").read())
    if mm:
        url = mm.group(0)
        break
    time.sleep(2)

if not url:
    print(open(TUN_LOG, errors="ignore").read()[-1500:])
    raise RuntimeError("Туннель не поднялся — запусти эту ячейку ещё раз")

print("Публичный адрес:", url)
time.sleep(3)

# 2) самопроверка: список моделей + реальный чат через туннель
req = urllib.request.Request(url + "/v1/models", headers={"Authorization": "Bearer " + API_KEY})
models = json.load(urllib.request.urlopen(req, timeout=30))
print("Модели на сервере:", [x["id"] for x in models.get("data", [])])

payload = json.dumps({
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Ответь коротко: ты работаешь?"}],
    "max_tokens": 50,
}).encode()
req = urllib.request.Request(
    url + "/v1/chat/completions", data=payload,
    headers={"Authorization": "Bearer " + API_KEY, "Content-Type": "application/json"},
)
resp = json.load(urllib.request.urlopen(req, timeout=300))
print("Ответ модели:", resp["choices"][0]["message"]["content"])

print()
print("=" * 56)
print("СКОПИРУЙ ЭТО И ПРИШЛИ В ЧАТ HERMES")
print("URL: " + url)
print("KEY: " + API_KEY)
print("MODEL: " + MODEL_NAME)
print("=" * 56)
"""

LLM_MD_FALLBACK = r"""
---
## АЛЬТЕРНАТИВА: Ollama (если vLLM не завёлся)

Если предыдущие ячейки упали с ошибкой — выполни **только эту ячейку** (она самодостаточная: поставит Ollama, скачает модель 7B и поднимет туннель). Учти: перед этим лучше сделать **Среда выполнения → Перезапустить сессию**.
"""

LLM_CELL_OLLAMA = r"""
# ==== АЛЬТЕРНАТИВА: Ollama. Выполни ЭТУ ячейку, если vLLM не завёлся. ====
import subprocess, os, sys, time, re, json, secrets, urllib.request

API_KEY = "sk-colab-" + secrets.token_hex(12)
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
os.makedirs("/content/logs", exist_ok=True)

print("Ставлю Ollama...")
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

os.makedirs("/content/ollama_models", exist_ok=True)
env = os.environ.copy()
env["OLLAMA_MODELS"] = "/content/ollama_models"
env["OLLAMA_HOST"] = "127.0.0.1:%d" % PORT
with open("/content/logs/ollama.log", "w") as lf:
    subprocess.Popen(["ollama", "serve"], stdout=lf, stderr=subprocess.STDOUT, env=env)
time.sleep(5)

print("Качаю модель qwen2.5 7B (~4.7 ГБ, 2–4 минуты)...")
subprocess.run(["ollama", "pull", MODEL_NAME], check=True, env=env)

if not os.path.exists("/content/cloudflared"):
    subprocess.run(
        "wget -q -O /content/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
        "&& chmod +x /content/cloudflared",
        shell=True, check=True,
    )
with open("/content/logs/cloudflared.log", "w") as lf:
    subprocess.Popen(
        ["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate"],
        stdout=lf, stderr=subprocess.STDOUT,
    )

url = None
for _ in range(60):
    mm = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open("/content/logs/cloudflared.log", errors="ignore").read())
    if mm:
        url = mm.group(0)
        break
    time.sleep(2)
if not url:
    raise RuntimeError("Туннель не поднялся — запусти эту ячейку ещё раз")

time.sleep(3)
payload = json.dumps({
    "model": MODEL_NAME,
    "messages": [{"role": "user", "content": "Ответь коротко: ты работаешь?"}],
    "max_tokens": 50,
}).encode()
req = urllib.request.Request(url + "/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=300))
print("Ответ модели:", resp["choices"][0]["message"]["content"])

print()
print("=" * 56)
print("СКОПИРУЙ ЭТО И ПРИШЛИ В ЧАТ HERMES")
print("URL: " + url)
print("KEY: " + API_KEY)
print("MODEL: " + MODEL_NAME)
print("=" * 56)
"""

LLM_MD_FAQ = r"""
---
## Если что-то не так

- **«Нет GPU»** → Среда выполнения → Сменить тип среды выполнения → **T4 GPU** → Сохранить, и «Выполнить все» заново.
- **vLLM упал** → запасная ячейка «АЛЬТЕРНАТИВА: Ollama» (или Перезапустить сессию и ещё раз).
- **Colab пишет, что функция недоступна / не даёт GPU совсем** → включи VPN и перезапусти сессию.
- **Хочу другую модель** → поменяй `MODEL_PRESET` в ячейке 1 и «Выполнить все» (модель скачается заново).
- **Ссылка перестала работать** → сессия Colab умерла. Это норма: «Выполнить все» заново и пришли новый блок в Hermes.
"""


# =====================================================================
#  ComfyUI_Server.ipynb
# =====================================================================

COMFY_MD_INTRO = r"""
# 🎨 Colab → Hermes: генератор картинок (ComfyUI + SDXL)

Поднимает на видеокарте Google Colab **ComfyUI** — самый мощный открытый генератор картинок (модель SDXL класса Juggernaut), делает публичную ссылку, сам скачивает модель и **рисует тестовую картинку** для проверки. Потом этим генератором можно пользоваться прямо из Hermes: «сгенерируй картинку …».

**Как запустить:**
1. Меню **Среда выполнения → Сменить тип среды выполнения → аппаратный ускоритель: T4 GPU** → Сохранить.
2. Меню **Среда выполнения → Выполнить все** (Ctrl+F9). Предупреждение «ноутбук не от Google» → **Run anyway**.
3. Первый запуск: 6–12 минут (качается модель ~7 ГБ). В конце появится тестовая картинка и блок **«СКОПИРУЙ ЭТО…»** — пришли его в чат Hermes.

**Что важно:**
- Нужна видеокарта (T4 бесплатно). Без GPU не поедет.
- Сессия живёт до ~12 часов. Умерла — «Выполнить все» заново, ссылка новая.
- Одновременно один GPU-ноутбук (этот ИЛИ LLM-сервер). Оба сразу — только на платных GPU.

**Чекпоинты (меняются в ячейке 1):** `juggernaut` (универсальный фото-стиль, по умолчанию), `realvis` (фотореализм), `sdxl_base` (базовый SDXL).
"""

COMFY_CELL_SETTINGS = r"""
# ==== НАСТРОЙКИ ====
CHECKPOINT = "juggernaut"  # "juggernaut" | "realvis" | "sdxl_base"
PORT = 8188                # локальный порт ComfyUI
TEST_IMAGE = True          # сгенерировать тестовую картинку в конце (рекомендую оставить True)

CHECKPOINTS = {
    "juggernaut": ("RunDiffusion/Juggernaut-XL-v9", "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"),
    "realvis":    ("SG161222/RealVisXL_V4.0",       "RealVisXL_V4.0.safetensors"),
    "sdxl_base":  ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
}
UPSCALER = ("lokCX/4x-Ultrasharp", "4x-UltraSharp.pth")
print("Настройки загружены. Чекпоинт:", CHECKPOINT)
"""

COMFY_CELL_GPU = r"""
import subprocess, re

gpu_info = subprocess.run(
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
    shell=True, capture_output=True, text=True,
).stdout.strip()
print("GPU:", gpu_info if gpu_info else "НЕ НАЙДЕН")
if not gpu_info:
    raise RuntimeError(
        "Нет GPU. Сделай: Среда выполнения → Сменить тип среды выполнения → T4 GPU → Сохранить, "
        "затем: Среда выполнения → Выполнить все."
    )
"""

COMFY_CELL_INSTALL = r"""
import subprocess, sys, os

os.makedirs("/content/logs", exist_ok=True)
if not os.path.isdir("/content/ComfyUI"):
    print("Клонирую ComfyUI...")
    subprocess.run("git clone -q https://github.com/comfyanonymous/ComfyUI /content/ComfyUI", shell=True, check=True)

print("Ставлю зависимости (2–3 минуты)...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub[hf_transfer]"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "/content/ComfyUI/requirements.txt"], check=False)

if not os.path.isdir("/content/ComfyUI/custom_nodes/ComfyUI-Manager"):
    subprocess.run("git clone -q https://github.com/ltdrdata/ComfyUI-Manager /content/ComfyUI/custom_nodes/ComfyUI-Manager", shell=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    "/content/ComfyUI/custom_nodes/ComfyUI-Manager/requirements.txt"], check=False)

print("✅ ComfyUI готов")
"""

COMFY_CELL_MODELS = r"""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

repo, fname = CHECKPOINTS[CHECKPOINT]
print("Качаю чекпоинт %s (~7 ГБ, 2–4 минуты)..." % fname)
p = hf_hub_download(repo_id=repo, filename=fname, local_dir="/content/ComfyUI/models/checkpoints")
print("✅ %s — %.1f ГБ" % (p, os.path.getsize(p) / 2**30))
CKPT_FILE = fname

u_repo, u_file = UPSCALER
p2 = hf_hub_download(repo_id=u_repo, filename=u_file, local_dir="/content/ComfyUI/models/upscale_models")
print("✅ %s — %.0f МБ" % (p2, os.path.getsize(p2) / 2**20))
"""

COMFY_CELL_LAUNCH = r"""
import subprocess, sys, time, json, urllib.request

LOG = "/content/logs/comfyui.log"
with open(LOG, "w") as lf:
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", str(PORT), "--disable-auto-launch"],
        cwd="/content/ComfyUI", stdout=lf, stderr=subprocess.STDOUT,
    )

ok = False
for _ in range(120):
    try:
        json.load(urllib.request.urlopen("http://127.0.0.1:%d/system_stats" % PORT, timeout=3))
        ok = True
        break
    except Exception:
        if proc.poll() is not None:
            break
        time.sleep(2)

if ok:
    print("✅ ComfyUI запущен локально на порту", PORT)
else:
    print("❌ Не поднялся. Лог:")
    print(open(LOG, errors="ignore").read()[-2000:])
    raise RuntimeError("ComfyUI не запустился — Перезапусти сессию и Выполнить все заново.")
"""

COMFY_CELL_TUNNEL = r"""
import subprocess, os, re, time, json, uuid, urllib.request, urllib.parse

# туннель
if not os.path.exists("/content/cloudflared"):
    subprocess.run(
        "wget -q -O /content/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 "
        "&& chmod +x /content/cloudflared",
        shell=True, check=True,
    )
TUN_LOG = "/content/logs/cloudflared.log"
with open(TUN_LOG, "w") as lf:
    subprocess.Popen(
        ["/content/cloudflared", "tunnel", "--url", "http://127.0.0.1:%d" % PORT, "--no-autoupdate"],
        stdout=lf, stderr=subprocess.STDOUT,
    )
url = None
for _ in range(60):
    mm = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", open(TUN_LOG, errors="ignore").read())
    if mm:
        url = mm.group(0)
        break
    time.sleep(2)
if not url:
    raise RuntimeError("Туннель не поднялся — запусти эту ячейку ещё раз")

print("Публичный адрес:", url)
print("Проверяю ComfyUI через туннель...")
urllib.request.urlopen(url + "/system_stats", timeout=60).read()
print("✅ сервер отвечает через туннель")

if TEST_IMAGE:
    wf = {
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT_FILE}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "cinematic photo of a fluffy ginger cat barista serving coffee to a corgi in a cozy cafe, warm window light, shallow depth of field, high detail"}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "watermark, text, logo, blurry, low quality, deformed, extra limbs"}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 777, "steps": 28, "cfg": 6.0, "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "hermes_test", "images": ["8", 0]}},
    }
    req = urllib.request.Request(
        url + "/prompt",
        data=json.dumps({"prompt": wf, "client_id": uuid.uuid4().hex}).encode(),
        headers={"Content-Type": "application/json"},
    )
    pid = json.load(urllib.request.urlopen(req, timeout=120))["prompt_id"]
    print("Генерирую тестовую картинку (30–120 секунд)...")

    img = None
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            h = json.load(urllib.request.urlopen(url + "/history/" + pid, timeout=60))
        except Exception:
            h = {}
        if pid in h:
            for node_out in h[pid].get("outputs", {}).values():
                for im in node_out.get("images", []):
                    img = im
                    break
            break
        time.sleep(4)

    if img:
        src = (url + "/view?filename=" + urllib.parse.quote(img["filename"])
               + "&subfolder=" + urllib.parse.quote(img.get("subfolder", ""))
               + "&type=" + img.get("type", "output"))
        data = urllib.request.urlopen(src, timeout=300).read()
        os.makedirs("/content/outputs", exist_ok=True)
        with open("/content/outputs/test.png", "wb") as f:
            f.write(data)
        from IPython.display import Image, display
        display(Image(data=data))
        print("✅ Тестовая картинка выше: сгенерирована и скачана ЧЕРЕЗ ТУННЕЛЬ — всё работает.")
    else:
        print("⚠️ Тестовая картинка не вышла, но сервер и туннель живы — проверь из Hermes.")

print()
print("=" * 56)
print("СКОПИРУЙ ЭТО И ПРИШЛИ В ЧАТ HERMES")
print("URL: " + url)
print("=" * 56)
"""

COMFY_MD_USE = r"""
---
## Дальше

- Этот ноутбук оставь запущенным — ссылка живёт, пока живёт сессия.
- Скажи Hermes «сгенерируй картинку: …» — он отправит запрос сюда.
- Сессия умерла? «Выполнить все» заново и пришли новую ссылку.
- Больше моделей/нод можно добавить через встроенный **ComfyUI-Manager** (меню в самом ComfyUI).
"""


# =====================================================================

def build_llm():
    return notebook([
        md(LLM_MD_INTRO),
        code(LLM_CELL_SETTINGS),
        code(LLM_CELL_GPU),
        code(LLM_CELL_INSTALL),
        code(LLM_CELL_START),
        code(LLM_CELL_TUNNEL),
        md(LLM_MD_FALLBACK),
        code(LLM_CELL_OLLAMA),
        md(LLM_MD_FAQ),
    ])


def build_comfy():
    return notebook([
        md(COMFY_MD_INTRO),
        code(COMFY_CELL_SETTINGS),
        code(COMFY_CELL_GPU),
        code(COMFY_CELL_INSTALL),
        code(COMFY_CELL_MODELS),
        code(COMFY_CELL_LAUNCH),
        code(COMFY_CELL_TUNNEL),
        md(COMFY_MD_USE),
    ])


def validate(nb, name):
    ok = True
    assert nb["nbformat"] == 4
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            try:
                compile(src, "%s#cell%d" % (name, i), "exec")
            except SyntaxError as e:
                ok = False
                print("!! SYNTAX ERROR", name, "cell", i, ":", e)
    return ok


def main():
    llm = build_llm()
    comfy = build_comfy()
    good = validate(llm, "LLM_Server") and validate(comfy, "ComfyUI_Server")
    if not good:
        raise SystemExit("Валидация не прошла — ноутбуки не записаны")
    for nb, fname in [(llm, "LLM_Server.ipynb"), (comfy, "ComfyUI_Server.ipynb")]:
        path = ROOT / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print("записан:", path, "| ячеек:", len(nb["cells"]))


if __name__ == "__main__":
    main()
