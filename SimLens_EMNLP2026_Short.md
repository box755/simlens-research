# SimLens EMNLP 2026 Short Paper 切片計畫
## "One-Shot Sparse Temporal Commentary Prediction via Distillation: Aligning 3B Students with Real Viewer Mentions"

> 版本：v1.1 (2026-05-06)
> 投稿目標：EMNLP 2026 Short Paper（4 頁正文 + references）
> 截稿：2026-05-25
> 切片來源：SimLens_Research_Plan_v4.2.md (v4.2.3) Phase 1 切片

> **v1.1 vs v1.0 修正（2026-05-06）：**
> - 🔴 **Top-8 → Top-30 personas**：對齊 SimTube IUI 2025 範式；single LoRA 沒有 multi-LoRA 訓練數量約束，挑 top-30 增加 persona 多樣性 + 訓練資料量
> - 🔴 **100/30 → 450/50 split**：500 部 mention-heavy 影片，450 train + 50 hold-out test
> - 🔴 **訓練資料 800 → 13,500 條**：450 train × 30 personas，single LoRA 無拆分問題
> - 🔴 **預算 ~$10 → ~$100**：Claude 蒸餾 + baselines 量級提升
> - 既有 100 部 train Stage A 歸檔到 `data/_archive_v4.2.3_full/`，不用於 EMNLP

> **Short Paper 範圍 vs Full Paper（v4.2.3）的差異：**
> - ✅ **保留**：Stage A timeline pipeline（Whisper + LLaVA + Timeline Script）
> - ✅ **保留**：Phase 1 Claude 蒸餾 → 3B SFT
> - ✅ **保留**：Anchor A（HF 187K 文本錨）+ Anchor B（per-video Hotspot Recall on high-mention test）
> - ✅ **保留**：Group 0 SCR/TVR + Multi-judge spot-check
> - ⚙️ **簡化**：8-LoRA per-persona → **single LoRA + prompt-injected persona**（Phase 1 baseline 設計）
> - ⚙️ **擴大**：top-8 personas → top-30 personas（single LoRA 解開 multi-LoRA 數量約束）
> - ⚙️ **擴大**：100 train + 30 test → 450 train + 50 test（標準 9:1 ML hold-out）
> - ❌ **砍**：Phase 2 RLAIF / DPO（→ full paper）
> - ❌ **砍**：8-LoRA persona differentiation 評估 Group 2/3（→ full paper）
> - ❌ **砍**：Group 4 25-人 Upwork study（→ full paper）
> - ❌ **砍**：Iterative DPO K rounds ablation（→ full paper）

---

## 0. One-Page Summary (Short Paper Scope)

```
研究問題：
   能否用 3B 小模型，從 Claude Sonnet 4.5 蒸餾出對 1-3 分鐘 YouTube 影片
   生成時序對齊 (per-video Hotspot Recall) 真實觀眾反應的 sparse JSON 評論？

核心方法（Phase 1 only, single LoRA, v1.1）：
   Stage A：Timeline Script
     - Whisper-Large-v3 整段轉錄（含時間戳）
     - LLaVA-NeXT-13B (4-bit) 每 10 秒 visual caption
     - 對齊組裝為文字 timeline

   Phase 1 SFT：
     - 500 部 mention-heavy YouTube 影片（compilation / highlights / sports / music mix
       / reaction breakdown / tutorial chapters），split 450 train + 50 hold-out test
     - PersonaChat 8K → top-30 personas（cosine + MMR, λ=0.4），對齊 SimTube IUI 2025 範式
     - Claude Sonnet 4.5 對 450 train 影片 × 30 personas = 13,500 sparse JSON
       （persona 在 prompt 中注入，**不訓練 per-persona LoRA**）
     - Llama-3.2-3B-Instruct (4-bit GPTQ) + single LoRA
     - 13,500 (timeline_script + persona_yaml → sparse_json) pairs SFT

關鍵差異化（vs SimTube IUI 2025）：
   SimTube：影片 → 整體理解 → per persona 1 條評論【無 timestamp】
   SimLens：Timeline Script → per persona N 條評論 [(t1, c1), (t2, c2), ...]

預期貢獻：
   C1. 範式：one-shot text-level timeline → sparse JSON 適用 1-3min 視覺評論生成
   C2. 經驗：3B SFT 在 per-video Hotspot Recall (vs 真實 YouTube 觀眾 mentions)
            上接近 600B Claude teacher，且 deterministic anchor metrics 證明
            未陷入 LLM-as-Judge 封閉論證
   C3. 評估方法：HF 187K + per-video Hotspot Recall @ ±5s 雙錨點
            破除 Claude→Qwen→Qwen 訓練評估封閉迴路

明確不做（Future work，Full paper extension）：
   ✗ Per-persona multi-LoRA differentiation
   ✗ RLAIF Phase 2 DPO 訓練
   ✗ Persona Consistency / Linguistic Habits per-persona breakdown
   ✗ 25-人 Upwork crowd study
   ✗ Stage C 報告生成
```

---

## 1. System Architecture（精簡版）

```
┌──────────────────────────────────────────────────────┐
│        輸入：1–3 min YouTube 短影片                     │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage A：Timeline Script（同 v4.2.3）                  │
│   Whisper-Large-v3 + LLaVA-NeXT-13B (4-bit)            │
│   → "[00:00-00:10] Visual: ... | Audio: ..."          │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Phase 1 SFT：One-Shot Sparse JSON Generation           │
│                                                       │
│   Llama-3.2-3B-Instruct (4-bit GPTQ)                   │
│   + Single LoRA (rank 8, target q/k/v/o)               │
│                                                       │
│   推理 prompt（persona 注入）：                          │
│     "You are a YouTube viewer with persona: {YAML}    │
│      [Timeline Script ...]                            │
│      Output sparse JSON:                              │
│      [{"timestamp": "MM:SS", "comment": "..."}]"      │
│                                                       │
│   產出：每 (影片, persona) 對 1 條 sparse JSON list     │
└──────────────────────────────────────────────────────┘
```

### 跟 v4.2.3 Full paper 的架構差異

| 維度 | Full paper (v4.2.3) | Short paper (本文 v1.1) |
|---|---|---|
| Persona conditioning | per-persona multi-LoRA (Neeko-style) | single LoRA + prompt injection |
| Persona 數量 | 8 | **30** (single LoRA 解開 multi-LoRA 數量約束，對齊 SimTube top-30 範式) |
| 影片數量 | 100 train + 30 hold-out | **450 train + 50 hold-out**（標準 9:1 ML split）|
| 訓練資料 | 800 sparse lists 拆 8 組各訓 1 LoRA | **13,500 sparse lists 全合併訓 1 個 LoRA** |
| Phase 2 | RLAIF DPO | 不做 |

---

## 2. Persona Setup（v1.1: top-30，對齊 SimTube IUI 2025）

30 個 personas 來自 PersonaChat 8K (`AlekseyKorshuk/persona-chat`)：
- OpenAI text-embedding-3-small + 500-mention-heavy-video keyword query → cosine similarity
- MMR (λ=0.4) top-N=200 → top-30 個多樣性 personas
- Claude Sonnet 4.5 估 expected_comment_count_range（每 2-min 影片 0-6 條）
- 30 個 personas + display labels 在 Appendix 完整列出

**為什麼是 top-30**（v1.0 → v1.1 修正）：

v4.2.3 full paper 用 top-8 personas 是因為 per-persona multi-LoRA 訓練資料分配（30 個 LoRA × 100 樣本太稀疏）。EMNLP Short 改 single LoRA 後，這個約束**消失**：
1. **Single LoRA 學的是「conditional generation given persona description」這個通用 instruction-following 能力**，跟 persona 數量無關
2. **Top-30 直接對齊 SimTube IUI 2025 範式**（SimTube 用 PersonaChat top-30），論文 head-to-head 比較時 reviewer 不會質疑「為什麼你只用 8 個」
3. **Persona 多樣性提升**（30 vs 8 涵蓋更廣 demographic）→ Anchor B per-video Hotspot Recall 評估時，30 personas 的 timestamp 聯集更接近真實觀眾的時序分布
4. **訓練資料量提升 16×**（13,500 vs 800），small-LoRA 訓練更穩

→ Multi-LoRA 評估（per-persona Linguistic Habits / Persona Consistency）保留給 full paper（top-8 vs top-30 是兩個不同的 design point）。

---

## 3. Method

### 3.1 Stage A: Timeline Script

同 v4.2.3 §3.2 Step 1.2，對 500 部 mention-heavy YouTube 影片各跑：
- Whisper-Large-v3 整段轉錄（含時間戳）
- LLaVA-NeXT-13B (4-bit) 每 10 秒抽 4 frames panel + audio chunk → ~150 字 visual caption
- 對齊組裝為純文字 Timeline Script

→ 500 個 Timeline Script，後續 split 為 450 train + 50 hold-out test。

### 3.2 Phase 1 SFT: Single-LoRA Distillation (v1.1)

**Prompt 模板**（訓練 / 推理共用，persona 注入透過 instruction）：
```python
PROMPT_TEMPLATE = """You are a YouTube viewer with this persona:
{persona_description}

Expected comment frequency on a typical 2-minute video: {low}-{high} comments.

You just finished watching a short video. Below is the complete timeline of what happened:

{timeline_script}

Reflect on the entire video. List the moments where you would have left a comment, staying in character with the persona description above. Match your persona's expected comment count range above (scaled to actual video duration).

Output ONLY a valid JSON array in this exact format:
[
  {{"timestamp": "MM:SS", "comment": "your comment here"}},
  ...
]

If nothing struck you, output an empty array: []."""
```

**訓練資料生成**：
```python
# Phase 1 Claude 蒸餾：450 train videos × 30 personas = 13,500 sparse JSON
distill_data = []
for video in train_450:
    timeline = load_timeline(video.id)
    for persona_id in P1..P30:                    # ⭐ 30 個 personas
        persona = PERSONAS[persona_id]
        prompt = PROMPT_TEMPLATE.format(
            persona_description=persona["description"],
            low=persona["expected_comment_count_range"][0],
            high=persona["expected_comment_count_range"][1],
            timeline_script=timeline,
        )
        sparse_json = claude_sonnet_4_5.generate(prompt)
        distill_data.append({
            "instruction": prompt,
            "input": "",
            "output": json.dumps(sparse_json, ensure_ascii=False),
        })

# 共 13,500 條 SFT data，all 30 personas 在 instruction 文字中輪流出現
# Single LoRA 訓練「讀 persona description → 生成符合該 persona 的 sparse JSON」這個
# conditional generation 能力（PersonaChat ACL 2018 + Alpaca instruction tuning 範式）
```

**為什麼 single LoRA + persona-prompt injection 可行**：

我們採用 RoleLLM (Wang et al., ACL 2024 Findings) 的 **Role-Conditioned Instruction Tuning (RoCIT)** 範式：single LoRA-adapted base model，所有 persona 樣本在同一次訓練中交錯出現，persona 身分**僅透過 system instruction 傳遞**，不引入任何 per-persona 參數。具體背書：

- **RoleLLM (ACL 2024 Findings)** 在 RoleBench-en 上以同樣設定訓練 single LLaMA-7B + LoRA，混合 **95 個 role × 平均 ~1,770 樣本（總計 168,093 條）**；關鍵地，§4.2 "Role Generalization" 證明該模型**可泛化到 10 個訓練時完全未見的 held-out roles** — 模型確實學到「conditional generation given persona description」這個 mapping，而非把 role 烘焙進權重或 collapse 到平均風格
- **PersonaChat (Zhang et al., ACL 2018)** 原始範式即 persona 作為 dialogue prefix；Alpaca / Vicuna instruction tuning 為其延伸
- 我們的設定（30 personas × 450 影片 = 13,500 樣本，~450 條/persona）與 RoleLLM 同數量級且 conditioning 機制相同，因此可預期同等的 persona-conditional generation 能力
- 推理時改 prompt 中 persona 描述即可動態切換輸出風格，無需切換 LoRA 權重

**訓練資料分布**：
- 30 personas × 450 影片 = 13,500 條樣本
- 每個 persona 有 450 個 sample（足夠 single-LoRA 學該 persona 風格）
- 對比 v4.2.3 multi-LoRA 設計：原 8 個 LoRA 各 100 sample（資料稀疏）→ 現 1 個 LoRA 13,500 sample（資料豐富）

**SFT config**（single LoRA）：
```python
config = {
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "quantization": "4-bit GPTQ",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    "epochs": 3,
    "batch_size": 2,
    "gradient_accumulation": 8,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "lr_scheduler": "cosine",
    "max_seq_length": 4096,
}
```

→ **唯一一次訓練 run**（vs full paper 8 次 per-persona LoRA）。

### 3.3 Inference + Constrained Decoding

```python
# 對每部 test 影片 × 每個 persona：(50 × 30 = 1,500 calls)
for video in test_50:
    for persona in P1..P30:
        prompt = format_simlens_prompt(video.timeline, persona)
        sparse_json = simlens.generate(
            prompt,
            constrained_json_decoding=True,  # Outlines / XGrammar
        )
        save_record(video.id, persona.id, sparse_json)
```

**Constrained decoding**：用 Outlines 強制 JSON schema：
```python
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string", "pattern": "^\\d{2}:\\d{2}$"},
            "comment": {"type": "string", "minLength": 5, "maxLength": 300}
        },
        "required": ["timestamp", "comment"]
    }
}
```

---

## 4. Experiments

### 4.1 Dataset (v1.1)

**500 部 mention-heavy YouTube 影片**（用 mention-rich content type queries 撈，
不限既有 v4.2.3 的 5 cat 範圍），split 為 450 train + 50 hold-out test。

| Split | 來源 | 數量 | Filter |
|---|---|---|---|
| Train | YouTube Data API v3 | 450 部 | 1-3min, en, view ≥10K, mention-heavy queries |
| Test (high-mention) | 同上 500 部子集 | 50 部 | 同上 + ≥5 timestamp mentions in pre-fetched 200 comments |

**Query 池（mention-heavy 內容類型）：**
- "highlights compilation" / "best moments compilation" / "top 10 moments"
- "epic moments" / "funniest moments"
- "reaction breakdown" / "scene by scene analysis" / "trailer breakdown"
- "music video reaction" / "music mix tracklist"
- "sports highlights" / "best plays" / "incredible goals"
- "tutorial chapters" / "step by step tutorial"

**Train/Test 內容類型一致性**：500 部統一從 mention-heavy queries 撈，避免 train (vlog/食譜) 跟 test (compilation) covariate shift。

**Selection bias 透明聲明**：本研究 evaluation slice 是「1-3 min YT 影片中 event-rich 子集」（每部 ≥5 mentions），論文 Limitations 明寫 generic 1-3 min 短片上的 hotspot 評估屬 future work。

**Test set 為什麼篩 high-mention**：per-video Hotspot Recall 需要 ground truth mentions。1-3 min YouTube 影片普遍 mention 稀疏（pilot 實測 47% 影片 0 條），mention-rich 內容（compilation / highlight / reaction breakdown / sports / music mix）才能提供穩定 KDE peak detection。

> **Selection bias 透明聲明**：本 test set 是 1-3 min YouTube 短片中「event-rich」的子集（每部 ≥5 mentions）。論文 Limitations 明寫 SimLens 在 generic 1-3 min 短片上的 per-video Hotspot 評估屬 future work（需擴展 test set 或設計新指標）。

### 4.2 Baselines

| Model | 角色 | Persona Conditioning |
|---|---|---|
| Llama-3.2-3B (zero-shot, no LoRA) | Untrained floor | Prompt injection |
| GPT-4o-mini (zero-shot) | Independent strong LLM | Prompt injection |
| Claude Sonnet 4.5 (zero-shot, snapshot `claude-sonnet-4-5-20250929`) | Teacher itself | Prompt injection |
| **SimLens-SFT (ours)** | 3B + single LoRA + Phase 1 distillation | Prompt injection |

### 4.3 Evaluation Metrics

#### Group 0: Format Compliance（基礎前提）
- **SCR (Schema Compliance Rate)**: JSON schema 合規率
- **TVR (Timestamp Validity Rate)**: timestamp 落在影片範圍內 + 不重複 + 排序
- 預期：Llama-zs ~30-40% / Claude ~95%+ / SimLens-SFT ~95%+

#### Anchor A: Distributional Match（文本錨）
比較 SimLens 整體留言生態 vs HF `breadlicker45/youtube-comments-180k` (n=176,562 真實 YT 留言)：

框架背書：採用 **MAUVE (Pillutla et al., NeurIPS 2021, Outstanding Paper Award)** 作為 generated-text-vs-human-text 分布距離的 umbrella paradigm，並補充兩個 deterministic 邊際分布指標：

| 指標 | 公式 | Citation |
|---|---|---|
| **Length KS** ↓ | scipy.stats.ks_2samp | Kolmogorov 1933 |
| **Sentiment Wasserstein** ↓ | VADER (neg/neu/pos) → 1-Wasserstein on ordered support | VADER (Hutto & Gilbert, ICWSM 2014) |
| **MAUVE** ↑ | k-means quantize GPT-2 large terminal hidden states → KL divergence on cluster histograms (官方 `mauve-text` 套件預設) | Pillutla et al. (NeurIPS 2021) |

實作細節：MAUVE 用官方套件預設（GPT-2 large feature extractor），SimLens corpus 與 HF 187K 各抽 5,000 條（套件建議最小樣本量），cluster size = 0.1 × samples。Length KS / Sentiment Wasserstein 全 deterministic、無 LLM judge；MAUVE 計算雖需 GPT-2 forward pass，但無生成步驟、結果 deterministic。

#### Anchor B: Per-Video Hotspot Recall（時序錨）⭐ 強 claim

**Ground truth 採集**：對 50 high-mention test 影片，YouTube Data API v3 commentThreads.list 撈 max 500 comments，regex 抓 timestamp mentions（"MM:SS" / "@MM:SS"），filter clock-time false positives。預期每部 5-15 條 mentions。

**評估流程**：
```python
for video in test_50:
    real_mentions = load_mentions(video.id)
    hotspots = kde_peaks(
        real_mentions,
        video.duration_sec,
        bandwidth=3,
        min_separation=5,
        top_k=ceil(video.duration_sec / 30),  # 1-3 min → 2-6 hotspot
    )
    # SimLens 30 personas timestamp 聯集
    predicted_union = union(
        sparse_list[t] for persona in P1..P30 for t in get_timestamps(persona)
    )
    recall, precision = per_video_hotspot_recall_precision(
        hotspots, predicted_union, tolerance_sec=5
    )
# 50 部影片取 mean
```

**指標** [Refs 37, 38]:
- **Hotspot Recall @ ±5s** ↑ = `|hotspots ∩ predicted| / |hotspots|`
- **Hotspot Precision @ ±5s** ↑ = `|predicted ∩ hotspots_dilated| / |predicted|`

Citations:
- SoccerNet (Giancola et al., CVPR 2018 Workshops): tolerance window paradigm
- DanmakuTPPBench (Jiang et al., NeurIPS 2025): timestamp event analysis

#### Multi-Judge Spot-Check（v4.2.3 D2 移植）
- 從 1,500 test outputs (50 × 30) 隨機抽 250（17%）
- Qwen3-32B-Q4 + GPT-4o-mini 平行評 Persona Consistency / Linguistic Habits
- Cohen's κ ≥ 0.6 → 主結果 robust；< 0.6 → Limitations 註明 judge sensitivity
- 成本：~$3 USD

> 註：完整 Group 2 內容指標（Persona Consistency / Linguistic Habits / Coherence / Engagingness）僅在 spot-check 範圍出現，main result table 不展示（→ full paper）。

### 4.4 Expected Results

#### Table 1: Main Result (4 systems × 5 metrics)

```
                            Anchor A (vs HF 187K real comments)        Anchor B (per-video, 50 test)
Method                      Length KS↓ | Sentiment W↓ | MAUVE↑       | Hotspot R@5s↑ | Hotspot P@5s↑
─────────────────────────────────────────────────────────────────────────────────────────────────
Llama-3B zero-shot          0.42       | 0.38         | 0.45         | 0.20          | 0.25
GPT-4o-mini                 0.28       | 0.25         | 0.70         | 0.45          | 0.48
Claude Sonnet 4.5 (Teacher) 0.31       | 0.29         | 0.68         | 0.58          | 0.55
─────────────────────────────────────────────────────────────────────────────────────────────────
SimLens-SFT (ours, 3B) ⭐   0.20       | 0.15         | 0.80         | 0.55          | 0.50
                            ↓/↑ closer to real than Claude/GPT         ↑ approaches Teacher
```

**Key claims to support in paper**：
1. SimLens-SFT (3B) **比 Llama-3B zero-shot 大幅優於**所有指標 → distillation **有用**
2. SimLens-SFT **在 Anchor A 三個分布距離指標上比 Claude/GPT-4o-mini 更接近真實 YT 留言** → 蒸餾學會了真實留言生態的「平均風格」
3. SimLens-SFT **在 Anchor B per-video Hotspot Recall 上接近 Claude teacher** (~0.55 vs 0.58) → 3B 學會了 600B teacher 的時序判斷

#### Table 2: Format Compliance Rate (FCR)

```
Method                              | SCR    | TVR    | Composite FCR
─────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot              | 35%    | 28%    | 28%
Claude Sonnet 4.5 zero-shot         | 95%+   | 90%+   | 90%+
GPT-4o-mini zero-shot               | 92%+   | 88%+   | 88%+
SimLens-SFT (ours)                  | 96%+   | 92%+   | 92%+
SimLens-SFT + Outlines (inference)  | 99%+   | 96%+   | 96%+
```

#### Table 3: Multi-Judge Spot-Check (Cohen's κ on 80 samples)

```
Aspect                  | Qwen3-32B-Q4 vs GPT-4o-mini Cohen's κ
──────────────────────────────────────────────────────────────────
Persona Consistency     | 0.65+
Linguistic Habits       | 0.60+
```

→ κ ≥ 0.6 確認 single-judge robustness。

### 4.5 Ablations (Limited, ≤2 in Short paper)

| Ablation | 移除什麼 | 預期 |
|---|---|---|
| A1. SimLens-SFT (full)         | 完整方法 | (baseline) |
| A2. - w/o distillation         | 用 Llama-3B zero-shot 跑 | Hotspot Recall 從 0.55 → 0.20，證明蒸餾必要 |

→ Single-paper Short 不深入 ablation（Phase 2 / 8-LoRA / persona-injection-method ablation 全留 full paper）。

---

## 5. Discussion + Limitations

### 5.1 Discussion (0.25 頁)

- 3B + SFT 蒸餾**接近** Claude Teacher 在 per-video Hotspot Recall (0.55 vs 0.58)，且**超越** Claude 在 Anchor A 分布距離 → 蒸餾能學到 teacher 的時序判斷能力 + 真實留言生態風格
- Anchor B + Anchor A 兩個 deterministic 錨點（無 LLM judge）+ Multi-judge spot-check Cohen's κ 證明結果不依賴單一 judge 偏好

### 5.2 Limitations (0.25 頁)

明確列出以避開 reviewer 攻擊：

L1. **Single LoRA + persona-prompt injection vs Per-Character LoRA**：本 short paper 採用 RoleLLM (ACL 2024 Findings) 的 RoCIT 範式 — 30 個 personas 共用一個 LoRA，透過 system instruction 注入區分。**Neeko (Yu et al., EMNLP 2024)** 已證明 per-character LoRA 在多角色 role-playing 任務上可避免 inter-character interference 而優於 shared LoRA + prompt 配置；本工作不對此 trade-off 作直接比較，per-persona LoRA 與 dynamic LoRA gating 是否在 sparse temporal commentary 任務上同樣帶來增益，留作 full paper extension。

L2. **No Phase 2 RLAIF**：本 short paper 僅展示 Phase 1 SFT distillation 結果。RLAIF DPO 進一步 alignment 為 future work。

L3. **Test set selection bias**：50 部 test 經 ≥5 mention pre-filter（high-mention 子集）。Generic 1-3 min 短片上的 hotspot 評估需擴展 test set 或設計新指標 → future work。

L4. **Post-hoc framing**：SimLens 模擬「事後反思」評論而非即時即播反應。即時行為模擬需 streaming video LLM 範式 → future work。

L5. **English-only**：30 個 personas + 450 train + 50 test 皆英文。中文 / 跨語言擴展 → future work。

L6. **Distillation bias**：Phase 1 從 Claude 蒸餾，可能繼承 Claude 偏誤（過度禮貌 / 避開敏感話題）。Anchor A 的 sentiment 指標部分捕捉，但無法完全消除。

---

## 6. References (預計 ~20 篇，Short paper 上限)

精簡版引用，從 v4.2.3 References 中挑：

1. **SimTube** (Hung et al., IUI 2025) — direct prior work
2. **Chapter-Llama** (Ventura et al., CVPR 2025) — text-level video → timestamp output 範式
3. **Socratic Models** (Zeng et al., ICLR 2023) — all-modality-as-text
4. **UMaT** (Bi & Xu, 2025) — temporal alignment fixed-segment
5. **PersonaChat** (Zhang et al., ACL 2018) — persona dataset
6. **RoleLLM** (Wang et al., ACL 2024 Findings) — RoCIT 範式：single LoRA + prompt-injected role description 處理多角色，95 roles 訓練 + held-out role 泛化
7. **Neeko** (Yu et al., EMNLP 2024) — dynamic per-character LoRA 對照（Limitations L1）
8. **DPO** (Rafailov et al., NeurIPS 2023) — 留作 future work cite
9. **Whisper** (Radford et al., 2022)
10. **LLaVA-NeXT** (Liu et al., 2024)
11. **Llama 3.2** (Meta, 2024)
12. **LoRA** (Hu et al., ICLR 2022)
13. **Outlines** (Willard et al., 2023) — constrained decoding
14. **JSONSchemaBench** (2025) — JSON schema compliance
15. **SoccerNet** (Giancola et al., CVPR 2018) — tolerance window for Hotspot Recall
16. **DanmakuTPPBench** (Jiang et al., NeurIPS 2025) — timestamp event analysis
17. **MAUVE** (Pillutla et al., NeurIPS 2021, Outstanding Paper) — generated-text-vs-human-text 分布距離 (Anchor A)
18. **VADER** (Hutto & Gilbert, ICWSM 2014) — sentiment classifier
19. **MMR** (Carbonell & Goldstein, SIGIR 1998) — persona diversity sampling
20. **DistilBERT** (Sanh et al., 2019) — distillation precedent
21. **RLAIF** (Lee et al., 2023) — 留作 future work cite

---

## 7. Execution Timeline (5/6 → 5/25, v1.1)

```
Day 0 (5/6, today)          : EMNLP plan finalize + 資料目錄重組 (data/emnlp/) + git push
Day 1-2 (5/7-8)             : YouTube quota refresh (~PT midnight)
                              撈 500 mention-heavy 影片 (multiple queries × pre-filter)
                              下載 + Stage A pipeline 500 部
                                Whisper ~3 hr / LLaVA ~7-8 hr / Timeline 1 min
                              (5090 跑 GPU stage 約 12 hr，可背景跑過夜)

Day 3 (5/9)                 : Top-30 personas 抽樣
                                從 500 部 timeline scripts 抽 keyword
                                OpenAI embedding + MMR top-30
                                Claude Sonnet 4.5 估 30 personas activity range

Day 4 (5/10)                : 篩 50 high-mention test 影片
                                每部 pre-fetch 200 comments，留 ≥5 mentions
                                採 50 test 完整 timestamp mentions（max 500 留言/部）

Day 5 (5/11)                : 跑 3 baselines (Claude/GPT-mini/Llama) on 50 test
                                Claude Sonnet 4.5 batch + 1h cache, ~$15
                                GPT-4o-mini batch, ~$8
                                Llama-3B local 5090, $0
                                = 4,500 sparse JSON outputs
                              算 baseline 的 Group 0 / Anchor A / B 數字

Day 6-8 (5/12-14)           : Phase 1 蒸餾 13,500 calls
                                Claude Sonnet 4.5, batch + 1h cache
                                ~$60, 預估 8-12 hr batch processing
                              格式化 alpaca SFT dataset
                              準備 LLaMA-Factory single-LoRA YAML

Day 9-10 (5/15-16)          : SFT 訓練 single LoRA (3 epochs, ~10-12 hr 5090)

Day 11-12 (5/17-18)         : SimLens-SFT inference on 50 test (1,500 calls)
                              算 Group 0 + Anchor A + B + Multi-judge spot-check (250 sample)
                              對比所有 4 systems (Llama/GPT-mini/Claude/SimLens)
                              產生 Table 1/2/3 數字

Day 13-19 (5/19-25)         : 寫論文初稿 4 頁
                                Section 1-5 撰寫
                                繪 architecture figure (Stage A + Phase 1)
                                繪 result figure (per-video hotspot examples)
                                Format check (EMNLP template)
                              5/24-25 final polish + 提交
```

**Time buffer**：~2-3 天（Day 1-12 是執行週，13-19 寫作週，buffer 足夠）。

**Risk**: Day 6-8 Phase 1 蒸餾 13,500 calls batch 預期 1-2 hr，但 Anthropic batch 偶有 1-2 day SLA。如果 batch slip 到 Day 9-10，吃 buffer。

---

## 8. Budget (v1.1)

```
Phase 1 蒸餾 (Claude Sonnet 4.5, batch + 1h cache)
  13,500 calls × ~$0.0044/call ≈ $60

Baseline benchmarks (50 test × 30 personas = 1,500 calls per model):
  Claude     : ~$15 (batch + cache)
  GPT-4o-mini: ~$8 (batch)
  Llama-3B   : $0 (local 5090)

Multi-judge spot-check (250 samples × 2 judges):
  Qwen-32B  : $0 (local Ollama)
  GPT-4o-mini: ~$8

Stage A:
  YouTube API : $0 (within quota, may need 2-3 day refresh)
  Whisper / LLaVA: $0 (local 5090)

OpenAI embedding:
  text-embedding-3-small (persona top-30 抽樣 keyword query): ~$1
  Note: Anchor A 的 MAUVE 改用 GPT-2 large 本地計算（mauve-text 套件預設），無 API 成本

Total: ~$92 USD
```

對比 v4.2.3 full paper（Phase 1 only ~$25）：~3.7× 預算，但取得：
- Persona 多樣性 30 vs 8（提升 3.75×）
- 訓練資料 13,500 vs 800（提升 16×）
- Test 樣本量 50 vs 30（提升 1.67×，per-video metric 統計信度更高）
- 對齊 SimTube IUI 2025 範式（reviewer 攻擊面降低）

---

## 9. Mapping back to v4.2.3 (Full paper)

| v4.2.3 Section | Short Paper 對應 | 備註 |
|---|---|---|
| §0 一頁摘要 | §0 (此檔) | 砍 Phase 2 / Stage C / Group 4 |
| §1 系統架構 | §1 | 砍 Multi-LoRA, single LoRA + prompt injection |
| §1.4 Scope rationale | §1 一段 | 保留 1-3min YouTube scope 解釋 |
| §2 Persona | §2 | 同 v4.2.3 |
| §3 Phase 1 蒸餾 | §3 | v1.1: 13,500 calls (450 × 30) 訓練 1 個 LoRA |
| §4 Phase 2 RLAIF | ❌ 砍 | future work |
| §5.2.1 Group 0 (SCR/TVR) | §4.3 Group 0 | 完整保留 |
| §5.2.2 Group 2 內容指標 | ❌ 砍主表 | 移到 multi-judge spot-check 內含 |
| §5.2.3 Group 3 List-Level | ❌ 砍 | future work (Persona Distinctiveness) |
| §5.2.5 Group 1+ Anchor A | §4.3 Anchor A | 完整保留 |
| §5.2.5 Group 1+ Anchor B | §4.3 Anchor B | 完整保留（per-video Hotspot Recall）|
| §5.2.6 Group 4 25-人 Upwork | ❌ 砍 | future work |
| §5.2.7 Diagnostic D2 Multi-Judge | §4.3 spot-check | 保留簡化版 |
| §5.3 Ablation 8 組 | §4.5 1 組 (w/o distillation) | 大幅簡化 |
| §5.4 Tables 1a/1b/1c/2/3/4/5/6 | §4.4 Table 1/2/3 | 三張表夠 short paper |
| §6 Stage C 報告生成 | ❌ 砍 | future work |
| §7 Week 8 時程 | §7 19 天時程 | 完全重寫 |
| §11 Limitations | §5.2 6 條精簡 | 加 single-LoRA 限制 |

---

## 10. Risk Mitigation (Short Paper-specific)

| 風險 | 緩解 |
|---|---|
| **訓練不收斂** (single LoRA 樣本量 13,500) | 樣本量充足（>16× v1.0），DistilBERT-style 設定已驗證可行；如失敗，加 epochs to 5 |
| **High-mention test set 撈不夠 50 部** | YouTube quota 一天可撈 1000+ 候選，pre-filter ≥5 mentions 預期 5-10% 通過率；可用備用：放寬到 ≥3 mentions |
| **Anchor B Recall < Llama zero-shot** | 設計上極不可能；若發生，重檢 KDE peak 設定 + ground truth 採集 |
| **Single judge / Cohen's κ < 0.6** | Limitations 直接寫，main result 加 sensitivity 描述 |
| **5/25 截稿前訓練 / 評估 bug** | Day 1-8 是執行週，Day 9-19 寫論文週，如某天 slip 可吃 buffer |
| **Reviewer 質疑「為什麼不做 multi-LoRA / RLAIF」** | Limitations L1, L2 明說「out of scope, future work」+ 一句話：「single-LoRA + prompt injection 是 minimal viable persona conditioning baseline，建立 timestamp prediction 能力證明後再進 multi-LoRA differentiation」 |

---

## 11. Author / Submission Info

- 第一作者：傅聖祐 Sheng-You Fu (NCU, SAILY Lab)
- 指導教授：Prof. Chia-Yu Lin
- 投稿系統：EMNLP 2026 OpenReview / ARR
- 截稿：2026-05-25 23:59 AoE
- 預估字數：4 頁正文（references + appendix 不計）≈ 3500-4000 words

---

## 12. Future Work（保留 v4.2.3 完整研究計畫的延伸）

EMNLP Short paper 是 Phase 1 SFT 切片，full paper extension（v4.2.3 完整版）規劃：

- **F1. Multi-LoRA per-persona** (Neeko EMNLP 2024 範式)：8 個獨立 LoRA vs 1 LoRA，per-persona Linguistic Habits / Persona Consistency
- **F2. RLAIF Phase 2 DPO**：4-aspect reward → 2-aspect reward，K∈{1,2,3} ablation
- **F3. Stage C 報告生成**：跨受眾比較報告
- **F4. 25-人 Upwork crowd study**（SimTube IUI 2025 protocol）
- **F5. Generic 1-3 min test set**（非 high-mention 子集），需新評估指標
- **F6. Cross-platform anchor**（Bilibili 彈幕 distribution-level，已採集 36 部 archive 中）
- **F7. >3min 中長片 / Shorts (<60s) extension**
- **F8. 中文 / 跨語言 SimLens**
- **F9. Streaming video LLM (即時觀影模擬)**

→ 詳見 `SimLens_Research_Plan_v4.2.md` (v4.2.3) 對應段落。

---

**END OF EMNLP 2026 SHORT PAPER PLAN**
