# SimLens Research Experiments

研究計畫的訓練 / 推理 / 評估腳本。在 3090 機器上執行，本地（VS Code Remote-SSH）編輯。

## 資料夾結構

```
experiments/
├── scripts/      # 可執行腳本（資料處理、訓練、評估）
├── configs/      # YAML 配置（persona schema、訓練超參數）
├── data/         # 影片素材、Whisper/LLaVA 輸出、Claude 蒸餾資料（不上 git）
├── models/       # 下載的預訓練權重（不上 git）
├── outputs/      # LoRA adapter、訓練 log、checkpoint（不上 git）
└── notebooks/    # 探索性分析、debug
```

## 執行環境

**3090 機器**（Ubuntu / RTX 3090 24GB）

兩個獨立 venv（避免依賴衝突）：

| venv | 用途 | 主要套件 |
|---|---|---|
| `simlens-perception` | Whisper + LLaVA-NeXT 推理 | `faster-whisper`, `transformers`, `torch` |
| `simlens-train` | Llama 訓練 + DPO | `LLaMA-Factory`, `trl`, `bitsandbytes`, `peft` |

加上 Ollama（系統層級，給 Qwen3-32B-Q4 reward judge 用）。

## Pipeline 概覽

對應研究計畫 v2.0：

```
Stage A: Whisper + LLaVA-NeXT → enriched_segments.json   (scripts/stage_a_*)
Stage B Phase 1: Claude 蒸餾資料生成 → SFT Llama-3B       (scripts/distill_*)
Stage B Phase 2: Qwen judge + DPO                         (scripts/dpo_*)
Stage C: 報告生成                                          (scripts/stage_c_*)
```

## Workflow

1. 本地用 VS Code Remote-SSH 連 3090
2. 在 3090 上 `cd ~/simlens-research/` 編輯/執行
3. 程式碼變更 commit + push 到 GitHub
4. 本地 `git pull` 同步回來

## 不要 commit 的東西

- 任何模型權重（看 `.gitignore`）
- 影片檔、轉錄檔、生成的反應矩陣
- API keys（用 `.env`）
- 訓練輸出
