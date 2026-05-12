# SimLens 推理伺服器部署指令（給 5090 機器的 Claude 執行）

> **對象**：在 5090 實驗室機器上執行的 Claude Code instance。
> **目標**：把 Whisper、LLaVA-NeXT、Llama-3 8B 三個模型部署成 HTTP 服務，供遠端 SimLens 後端呼叫。
> **使用者已確認**：硬體 RTX 5090 32GB、OS Windows 11 + WSL2 Ubuntu 24.04、CUDA driver 576.88 / CUDA 12.9、GPU passthrough 已驗證 OK。

---

## 0. 前置確認（先跑這些再開始）

請先確認環境，不要假設。如果有任何一項失敗，停下來回報使用者。

```bash
# 1. WSL2 內可看到 5090
nvidia-smi
# 預期：看到 RTX 5090，driver 576.88，CUDA Version 12.9

# 2. Python 版本
python3 --version
# 預期：Python 3.10 或 3.11（vLLM 對 3.12 支援還不穩）

# 3. 系統工具
which git ffmpeg curl
# 預期：三個都有；ffmpeg 是 LLaVA 抽幀必需

# 4. 磁碟空間
df -h ~
# 預期：至少 100GB 可用（模型權重很大：Whisper-large ~3GB / LLaVA-NeXT ~15GB / Llama-3 8B ~16GB）
```

如果 `ffmpeg` 沒裝：

```bash
sudo apt update && sudo apt install -y ffmpeg
```

---

## 1. 目錄結構

按照使用者既定規範（研究機器與工程機器分離），所有東西放在 `~/simlens-inference/`：

```bash
mkdir -p ~/simlens-inference/{servers,models,logs,scripts}
cd ~/simlens-inference
```

預期結構：

```
~/simlens-inference/
├── servers/         ← 三個 server 的 Python 程式碼
│   ├── whisper_server.py
│   └── llava_server.py
├── models/          ← HuggingFace cache（HF_HOME 指過來）
├── logs/            ← systemd / uvicorn log
├── scripts/         ← 啟動腳本
└── .venv/           ← 共用 Python 虛擬環境
```

---

## 2. Python 虛擬環境

**用一個共用 venv** 放 vLLM + faster-whisper + LLaVA 依賴。理由：vLLM 跟 transformers 版本綁很緊，分開裝容易衝突；統一管理比較單純。

```bash
cd ~/simlens-inference
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
```

> **重要**：之後所有 `pip install` 跟啟動 server 都要先 `source ~/simlens-inference/.venv/bin/activate`。

---

## 3. PyTorch（必須先裝對版本）

5090 是 sm_120 架構，**需要 PyTorch nightly 才有支援**。穩定版（2.4 / 2.5）會跑不起來或精度炸裂。

```bash
# CUDA 12.9 對應的 nightly
pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/nightly/cu124
```

裝完驗證：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 預期輸出類似：2.6.0.dev20250xxx+cu124 True NVIDIA GeForce RTX 5090
```

如果 `torch.cuda.is_available()` 是 False，**停下來回報**，不要繼續往下裝。

---

## 4. Whisper Server（faster-whisper + FastAPI）

### 4.1 裝套件

```bash
pip install faster-whisper fastapi uvicorn[standard] python-multipart
```

### 4.2 寫 server 程式

```bash
cat > ~/simlens-inference/servers/whisper_server.py <<'PYEOF'
"""
Whisper 轉錄服務
- POST /transcribe (multipart file) → {language, segments: [{start, end, text}]}
- GET  /health → {status: "ok"}
"""
import os, tempfile
from fastapi import FastAPI, UploadFile, HTTPException
from faster_whisper import WhisperModel

MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

app = FastAPI(title="SimLens Whisper Server")
model = None  # lazy load

def get_model():
    global model
    if model is None:
        model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return model

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_SIZE, "loaded": model is not None}

@app.post("/transcribe")
async def transcribe(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "filename required")
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        segments, info = get_model().transcribe(path, word_timestamps=False)
        return {
            "language": info.language,
            "duration": info.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segments
            ],
        }
    finally:
        os.unlink(path)
PYEOF
```

### 4.3 第一次啟動（手動測試）

```bash
cd ~/simlens-inference
source .venv/bin/activate
HF_HOME=~/simlens-inference/models uvicorn servers.whisper_server:app --host 0.0.0.0 --port 8001
```

第一次跑會下載 large-v3 權重（~3GB），等它跑完看到 `Uvicorn running on http://0.0.0.0:8001`。

**另一個 terminal 測試**：

```bash
curl http://localhost:8001/health
# 預期：{"status":"ok","model":"large-v3","loaded":false}

# 如果有測試影片：
curl -X POST -F "file=@/path/to/test.mp4" http://localhost:8001/transcribe
```

確認 OK 後 Ctrl+C 停掉，後面用 systemd 管。

---

## 5. LLaVA-NeXT Server

### 5.1 裝套件

```bash
pip install transformers accelerate pillow
```

### 5.2 寫 server 程式

```bash
cat > ~/simlens-inference/servers/llava_server.py <<'PYEOF'
"""
LLaVA-NeXT 視覺描述服務
- POST /describe {image_b64, prompt?} → {description}
- GET  /health → {status, loaded}
"""
import base64, io, os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

MODEL_ID = os.getenv("LLAVA_MODEL", "llava-hf/llava-v1.6-mistral-7b-hf")

app = FastAPI(title="SimLens LLaVA-NeXT Server")
processor = None
model = None

def load_model():
    global processor, model
    if model is None:
        processor = LlavaNextProcessor.from_pretrained(MODEL_ID)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="cuda",
            low_cpu_mem_usage=True,
        )
    return processor, model

class DescribeReq(BaseModel):
    image_b64: str
    prompt: str = "Describe this video frame in detail. Focus on what is happening, who is in the scene, and the overall mood."
    max_new_tokens: int = 300

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "loaded": model is not None}

@app.post("/describe")
def describe(req: DescribeReq):
    try:
        img_bytes = base64.b64decode(req.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"invalid image_b64: {e}")

    proc, mdl = load_model()
    conversation = [
        {"role": "user", "content": [
            {"type": "text", "text": req.prompt},
            {"type": "image"},
        ]},
    ]
    prompt_str = proc.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = proc(prompt_str, img, return_tensors="pt").to("cuda", torch.float16)
    with torch.inference_mode():
        out = mdl.generate(**inputs, max_new_tokens=req.max_new_tokens, do_sample=False)
    text = proc.decode(out[0], skip_special_tokens=True)
    # 去掉 prompt 部分，只回傳模型生成的描述
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:", 1)[1].strip()
    return {"description": text}
PYEOF
```

### 5.3 第一次啟動

```bash
cd ~/simlens-inference
source .venv/bin/activate
HF_HOME=~/simlens-inference/models uvicorn servers.llava_server:app --host 0.0.0.0 --port 8002
```

第一次會下載 ~15GB 權重，需要點時間。下載完看到 Uvicorn 起來後測試：

```bash
curl http://localhost:8002/health
# 預期：{"status":"ok","model":"llava-hf/llava-v1.6-mistral-7b-hf","loaded":false}
```

確認 OK 後 Ctrl+C。

---

## 6. Llama-3 8B Server（vLLM）

### 6.1 裝 vLLM

```bash
pip install vllm
```

> 如果 vLLM 對 PyTorch nightly 抱怨版本不符，**先試著跑跑看**——目前（2026-05）vLLM 對 5090 的支援可能還在追，如果起不來，回報使用者再決定 fallback（用 transformers 直接包 server）。

### 6.2 申請 HuggingFace 模型存取

Llama-3-8B-Instruct 是 gated model，需要先在 HuggingFace 同意條款。

```bash
# 如果使用者已經有 HF token：
huggingface-cli login
# 貼上 token

# 確認已 accept Meta 的條款：
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
```

如果使用者還沒申請，**停下來回報**，不要繼續。

### 6.3 啟動腳本

```bash
cat > ~/simlens-inference/scripts/start_llama.sh <<'SHEOF'
#!/bin/bash
set -e
cd ~/simlens-inference
source .venv/bin/activate
export HF_HOME=~/simlens-inference/models

vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 \
  --port 8003 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.55 \
  --dtype float16 \
  --served-model-name llama-3-8b
SHEOF
chmod +x ~/simlens-inference/scripts/start_llama.sh
```

> **`--gpu-memory-utilization 0.55` 的含義**：vLLM 會吃掉 5090 32GB 的 55%（~17.5GB），剩下給 LLaVA / Whisper。如果之後三個 server 同時跑出現 OOM，把這個值降到 0.45。

### 6.4 第一次啟動

```bash
~/simlens-inference/scripts/start_llama.sh
```

第一次下載 Llama-3 8B（~16GB）。看到 vLLM 起來後測試：

```bash
curl http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3-8b",
    "messages": [{"role": "user", "content": "say hi in one word"}],
    "max_tokens": 10
  }'
# 預期：回一個包含 "hi" 或類似的 chat completion JSON
```

確認 OK 後 Ctrl+C。

---

## 7. systemd 管理三個 service

讓三個 server 開機自動啟動、掛掉自動重啟。

### 7.1 建立 service 檔案

> **重要**：把下面三段裡的 `__USERNAME__` 換成 WSL2 的使用者名稱，可以用 `whoami` 取得。

先抓 username：

```bash
echo "Username: $(whoami)"
echo "Home: $HOME"
```

然後建立三個 service file：

```bash
sudo tee /etc/systemd/system/simlens-whisper.service > /dev/null <<EOF
[Unit]
Description=SimLens Whisper Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HOME/simlens-inference
Environment="HF_HOME=$HOME/simlens-inference/models"
ExecStart=$HOME/simlens-inference/.venv/bin/uvicorn servers.whisper_server:app --host 0.0.0.0 --port 8001
Restart=on-failure
RestartSec=10
StandardOutput=append:$HOME/simlens-inference/logs/whisper.log
StandardError=append:$HOME/simlens-inference/logs/whisper.err.log

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/simlens-llava.service > /dev/null <<EOF
[Unit]
Description=SimLens LLaVA-NeXT Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HOME/simlens-inference
Environment="HF_HOME=$HOME/simlens-inference/models"
ExecStart=$HOME/simlens-inference/.venv/bin/uvicorn servers.llava_server:app --host 0.0.0.0 --port 8002
Restart=on-failure
RestartSec=10
StandardOutput=append:$HOME/simlens-inference/logs/llava.log
StandardError=append:$HOME/simlens-inference/logs/llava.err.log

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/simlens-llama.service > /dev/null <<EOF
[Unit]
Description=SimLens Llama-3 8B vLLM Server
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HOME/simlens-inference
ExecStart=$HOME/simlens-inference/scripts/start_llama.sh
Restart=on-failure
RestartSec=15
StandardOutput=append:$HOME/simlens-inference/logs/llama.log
StandardError=append:$HOME/simlens-inference/logs/llama.err.log

[Install]
WantedBy=multi-user.target
EOF
```

### 7.2 啟動 + 設定開機自動

```bash
sudo systemctl daemon-reload

# 先一個一個啟動，確認沒問題再 enable
sudo systemctl start simlens-whisper
sleep 5
sudo systemctl status simlens-whisper --no-pager
curl http://localhost:8001/health

sudo systemctl start simlens-llava
sleep 5
sudo systemctl status simlens-llava --no-pager
curl http://localhost:8002/health

sudo systemctl start simlens-llama
sleep 30  # vLLM 啟動較慢
sudo systemctl status simlens-llama --no-pager
curl http://localhost:8003/v1/models

# 都 OK 才 enable 開機自動
sudo systemctl enable simlens-whisper simlens-llava simlens-llama
```

如果某個 service 起不來，看 log：

```bash
journalctl -u simlens-whisper -n 50 --no-pager
# 或
tail -f ~/simlens-inference/logs/whisper.err.log
```

---

## 8. WSL2 → Windows 端口轉發（如果後端不在 5090 機器）

**WSL2 有獨立的虛擬網卡**，預設外部裝置連不到 WSL2 內的 port。要在 Windows host 設 port forwarding：

在 Windows PowerShell（系統管理員）執行：

```powershell
# 先抓 WSL2 的 IP
$wslIp = (wsl hostname -I).Trim().Split()[0]
Write-Host "WSL2 IP: $wslIp"

# 三個 port 都轉發
netsh interface portproxy add v4tov4 listenport=8001 listenaddress=0.0.0.0 connectport=8001 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8002 listenaddress=0.0.0.0 connectport=8002 connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8003 listenaddress=0.0.0.0 connectport=8003 connectaddress=$wslIp

# 確認
netsh interface portproxy show all

# Windows 防火牆開洞
New-NetFirewallRule -DisplayName "SimLens Inference" -Direction Inbound -LocalPort 8001,8002,8003 -Protocol TCP -Action Allow
```

> **WSL2 IP 重啟後會變**。長期方案是設一個 Windows 排程，開機時自動更新 portproxy。如果使用者要這個，回報後再做。

---

## 9. Tailscale（推薦的跨網路連線方式）

如果 SimLens 後端不在同一個內網（例如後端在雲端、5090 在實驗室），用 Tailscale 把兩端串起來。

### 9.1 在 Windows host 裝 Tailscale

從 https://tailscale.com/download/windows 下載安裝。登入帳號後，這台機器會拿到一個 `100.x.x.x` 的固定 IP，跨網路可達。

### 9.2 確認後端機器也加入同一個 tailnet

後端那邊也裝 Tailscale 並登入同一個帳號。

### 9.3 後端連線位址

後端的 env 設：

```
LAB_INFERENCE_HOST=100.x.x.x   # 5090 機器的 Tailscale IP
WHISPER_URL=http://100.x.x.x:8001
LLAVA_URL=http://100.x.x.x:8002
LLAMA_URL=http://100.x.x.x:8003/v1
```

> **不要把 server port 直接暴露到公網**，沒有 auth、沒有 rate limit、會被掃。Tailscale 是內網疊加，安全。

---

## 10. 驗收清單

全部跑完後，確認以下都通過再回報「部署完成」：

```bash
# A. 三個 service 都 active
systemctl is-active simlens-whisper simlens-llava simlens-llama
# 預期：三個都 active

# B. 三個 health check 都 OK
curl -s http://localhost:8001/health
curl -s http://localhost:8002/health
curl -s http://localhost:8003/v1/models

# C. VRAM 使用 < 28GB（留 buffer 給推理峰值）
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
# 預期：~22-25GB used / 32760 MiB total（Llama 長駐 + Whisper/LLaVA lazy 還沒 load 的話會更少）

# D. 從外部機器（同 Tailnet 或同內網）測試
# 在後端機器跑：
curl http://<5090-ip>:8001/health
curl http://<5090-ip>:8003/v1/models
```

---

## 11. 回報模板

部署完成後，請回報以下資訊給使用者：

```
✅ SimLens 推理伺服器部署完成

服務：
- Whisper (large-v3)   http://<ip>:8001
- LLaVA-NeXT (mistral-7b)  http://<ip>:8002
- Llama-3 8B (vLLM)    http://<ip>:8003/v1

連線方式：[Tailscale IP / 內網 IP / 其他]

VRAM 占用（idle）：[XX] GB / 32 GB
模型權重總大小：[XX] GB

systemd 已 enable，開機自動啟動。
log 在 ~/simlens-inference/logs/。

[如果有任何步驟踩雷或 fallback，列在這裡]
```

---

## 12. 常見問題排錯

| 症狀 | 原因 | 解法 |
|---|---|---|
| `torch.cuda.is_available() == False` | PyTorch 不是 nightly / sm_120 不支援 | 重裝 §3 的 nightly |
| vLLM 啟動 OOM | `gpu-memory-utilization` 太高 | 降到 0.45 |
| LLaVA 推理慢 | FP16 + 大 image | 確認 `torch_dtype=torch.float16`，必要時降 max_new_tokens |
| Whisper 第一次跑很慢 | 模型還在下載 | `tail -f ~/simlens-inference/logs/whisper.log` 看進度 |
| 外部連不到 port | WSL2 沒做 portproxy | 跑 §8 的 PowerShell 指令 |
| systemd service 一直 restart | 看 journalctl | `journalctl -u <service> -n 100` |

---

## 13. 不要做的事

- ❌ 不要 `pip install` 在 system Python（會污染環境），一定要 `source .venv/bin/activate`
- ❌ 不要把三個 server 的 port 暴露到公網
- ❌ 不要在 Llama 跑著的時候同時 load LLaVA 13B 版本（會 OOM；7B 版才安全）
- ❌ 不要動 `~/simlens-inference/` 以外的目錄（與工程系統 docker SimLens 完全隔離）
- ❌ 不要假設 PyTorch 穩定版能跑 5090，**一定要 nightly**
