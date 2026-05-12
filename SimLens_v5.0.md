# SimLens v5.0 — Persona-Conditioned Video Comment Generation via Timeline-Text Representation + Multi-LoRA

**版本**：5.0（全新方向，不繼承 v4.x）
**修訂歷程**：
- 2026-05-09 v5.0 初版
- 2026-05-10 修訂 1：reasoning 降級為 demo 章節，論文主體只剩 Stage 1
- 2026-05-10 修訂 2：使用者推論流程改為「從預設 persona 選一個」、Demo 加入 5 分鐘長片切 chunk 處理、明確學術誠實性聲明
- 2026-05-10 修訂 3：Demo 加入 Mode 2 (Custom Persona Fusion，Neeko-inspired)
- 2026-05-10 修訂 4 (重大改版)：**Pipeline 從 Video-LLaVA 直接餵 frames 改為 timeline-text representation**——影片用 LLaVA-NeXT (visual) + Whisper (audio) 預先轉成每 10 秒一段的 timeline text，後續用純文字 LLM (Llama-3.1-8B-Instruct) + LoRA。對標 PV-LLM 從 head-to-head 變為 alternative-approach，但獲得長片自然支援、訓練成本下降、v4.x 工具鏈復用三個好處。
- 2026-05-10 修訂 5：**新增 Phase 1.5 — Long Timeline Synthesis** 作為論文主貢獻 #3。透過 segment recombination + curriculum learning 解決 train-test length mismatch 問題（短 timeline 訓練、長 timeline 推論）。學術 grounding：Self-Instruct (ACL 2023)、LIFT (2024)、OpenCharacter (2025)、LongLoRA (ICLR 2024)。新增 RQ5。

**日期**：2026-05-10
**作者**：林冠均

---

## 0. TL;DR

本研究針對 **Personalized Video Comment Generation (PVCG)** 任務，提出 **timeline-text representation + persona-conditioned multi-LoRA** 方法：

1. **Phase -1 — Timeline 生成**：對 PerVidCom 影片以 10 秒為粒度切段，用 LLaVA-NeXT-13B 4-bit 生成 visual 描述、Whisper-Large-v3 生成 audio 字幕，組合成結構化 timeline text。
2. **Phase 0 — Persona 抽取**：將 PerVidCom 用戶依留言行為以 style embedding 分群為 K 個 cluster，每個 cluster 由 LLM 生成 persona 描述。
3. **Phase 1 — Multi-LoRA 訓練**：訓練資料中 user_id 被 persona ID 取代。輸入為 (persona prompt, timeline text)，輸出為 comment。Base model 為純文字 Llama-3.1-8B-Instruct + LoRA。
4. **Phase 2 — 應用推論**：使用者從 K 個預設 persona 中單選一個 + 上傳影片 → 系統跑 timeline 生成 → 載入對應 LoRA → 生成符合該 persona 風格的留言。

**論文核心貢獻**：
- 提出 **timeline-text representation** 作為 PVCG 的替代輸入路徑，相較 PV-LLM 的 frame-based 設計獲得長片擴展能力
- 提出 **persona-conditioned multi-LoRA**，補上 PV-LLM 缺乏顯式 user modeling 的空缺
- **Phase 1.5: Synthetic Long-Timeline Training**——應用 LIFT/OpenCharacter 方法論，以僅有短片訓練資料的情況下擴展至長片推論
- 雖然 input format 與 PV-LLM 不同（無法 head-to-head），仍在 PerVidCom 上以 FICL-Score 量化評估

**延伸應用（Stage 2，demo 用）**：在 Stage 1 訓好的 LoRA 上以純 prompt 方式套上時序生成，模擬 agent 邊看邊產生 reaction trace。**Demo 支援 5 分鐘長片為 native feature**（timeline-text 結構天然 scale），不需 hierarchical hack。

**學術誠實性宣告**：
- 論文評估範疇限定於 PerVidCom（~21 秒），但因 timeline-text 結構與長片完全一致，方法在架構層面支援長片
- 不主張「打贏 PV-LLM」（input format 不同），改主張「提供 timeline-text 路徑作為 alternative，且具備長片擴展優勢」
- Demo 上的 reaction trace 仍不主張任何擬真性質

---

## 1. Problem Statement

### 1.1 任務定義 (PVCG)

給定影片 v 與某個 persona p（由使用者選定），生成一則符合 persona p 留言風格的 comment。

與 Lin et al. (2024) PVCG 原始任務的差異：

| 維度 | Lin et al. PVCG | 本研究 |
|---|---|---|
| 推論輸入 | 特定 unseen user 的歷史留言 H_u | 使用者**從預設選單中選擇** persona ID |
| 任務本質 | user-specific personalization | **persona-conditioned simulation** |
| Ground truth | 該 user 對該影片的真實留言 | 同一 cluster 中真實用戶的留言（用作評估代理）|
| 應用情境 | 需有特定使用者的歷史資料 | 工具型應用：「我想看 X 類觀眾會怎麼留言」 |

### 1.2 既有方法的限制

**Lin et al. (2024) EMNLP — PV-LLM**：
- 從 Video-LLaVA-7B SFT，純粹靠 prompt 中塞 5 則歷史留言做 user conditioning
- **架構限制 1：frame-based input**——直接餵入 8 frames，對長片無法泛化（5 分鐘影片仍只抽 8 frames，視覺資訊嚴重稀疏）
- **架構限制 2：無顯式 user modeling**——無分群、無 user embedding、無 archetype-specific 參數
- **應用限制**：推論時必須有特定使用者的歷史留言，不適合工具型應用情境
- 結果：在 emotion / style 維度顯著輸給 Gemini-1.5-Flash（35.5 vs 39.6 / 51.8 vs 59.6）

**Wu et al. (2024) WWW — Personalized VideoIC**：
- 處理的是 Bilibili 彈幕（time-sync comment），不是一般留言
- 用戶風格被彈幕文化（迷因、跟風）稀釋，個人風格訊號弱
- 採用 seen-user 設定，較不擬真

**研究 gap**：
1. PVCG 任務缺乏對長片的擴展（PV-LLM frame-based 限制）
2. 沒有顯式 user/persona modeling 方法
3. 工具型應用情境下，使用者沒有特定 user 歷史，仍應能模擬「某類觀眾」

### 1.3 研究問題

- **RQ1**：以 timeline-text representation（LLaVA-NeXT visual + Whisper audio + 結構化分段）取代 frame-based input，能否在 PVCG 任務上達到合理的 FICL-Score？
- **RQ2**：persona-specific 權重（K 個 LoRA）vs persona-conditioned prompt（1 個 LoRA + persona token），哪個更能擬真？
- **RQ3**：在不同的群數 K 與 style encoder 選擇下，persona 的可解釋性與分群品質如何？
- **RQ4**：Timeline 段粒度（5s / 10s / 整支）對效能的影響？
- **RQ5**：合成長 timeline 訓練 (Phase 1.5) 是否能讓模型在長片推論上保持 persona 風格與品質？合成策略 (segment recombination vs LLM-generated) 哪個更有效？

---

## 2. Datasets

### 2.1 主資料集 — PerVidCom (Lin et al., 2024)

| 項目 | 數值 |
|---|---|
| 來源 | YouTube Shorts |
| 影片數 | 9,839 |
| 用戶數 | 16,702 |
| 留言數 | 344,441 |
| 平均影片長度 | ~21 秒 |
| 平均留言字數 | 10.85 |
| 切分 | Train 60% / Val 15% / Test 25%（**按 user 切，unseen 設定**）|
| Train | 10,021 users / 9,674 videos / 212,276 comments |
| Val | 2,505 users / 8,011 videos / 53,898 comments |
| Test | 4,176 users / 8,832 videos / 86,195 comments |
| 篩選 | 用戶留言 ≥11、影片 likes >400、Personalization Score >90% |

**待確認**：實際資料下載管道（YouTube Data API + 論文釋出的 video/user ID 列表）。

### 2.2 不採用 Personalized VideoIC 的理由

- 來源是 Bilibili 彈幕，個人風格被彈幕文化稀釋
- 中文 + seen-user 設定，與 unseen-user 的研究主軸不符

---

## 3. Method

### 3.1 整體架構

```
┌──────────── PHASE -1: Timeline 生成 (一次性, 預處理) ───┐
│  PerVidCom 影片                                          │
│        │                                                  │
│        ▼  切 10 秒 segment                               │
│        │                                                  │
│        ├─→ LLaVA-NeXT-13B 4-bit → visual 描述            │
│        └─→ Whisper-Large-v3 → audio 字幕                 │
│        │                                                  │
│        ▼  組合成結構化 timeline text                      │
│   每影片 → ~3 段 (21s / 10s) timeline                    │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────── PHASE 0: Persona 抽取 ────────────┐
│  PerVidCom train users (留言文字)             │
│        │                                       │
│        ▼  Style embedding (LUAR)               │
│        │                                       │
│        ▼  K-Means clustering (K personas)      │
│        │                                       │
│        ▼  LLM (Gemini) 生成 persona 描述       │
│   {persona_1, ..., persona_K} （固定文字）     │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────── PHASE 1: Multi-LoRA 訓練 ──────────┐
│  訓練樣本 = (persona_k, timeline_text,         │
│              comment)                          │
│  user_id 被 persona_k 取代                     │
│                                                │
│  變體 A (baseline):  Single LoRA + persona     │
│                      token in prompt           │
│  變體 B (main):      K independent LoRAs       │
│                      (one per persona)         │
│                                                │
│  Base model: Llama-3.1-8B-Instruct (純文字)    │
└────────────────────────────────────────────────┘
                      │
                      ▼
┌──────────── PHASE 2: 推論 (應用情境) ──────────┐
│  使用者從 K 個預設 persona 按鈕選一個 (單選)   │
│  + 上傳影片 v                                  │
│        │                                       │
│        ▼  跑 Phase -1 timeline 生成 pipeline   │
│           → timeline_text(v)                   │
│        │                                       │
│        ▼  載入 LoRA_k                          │
│        │                                       │
│        ▼  prompt = "You are persona-k.         │
│           {timeline_text}. Write a comment."   │
│        │                                       │
│        ▼  Generate comment                     │
│                                                │
│  註：本研究 Phase 2 採「單一 persona 選擇」    │
│  不做 Neeko 風格的多 LoRA 融合 / soft routing  │
│  自訂 persona 為 future work (§11)             │
└────────────────────────────────────────────────┘
```

### 3.2 Phase -1 — Timeline 生成（一次性，所有後續階段的前置）

對 PerVidCom 影片以 10 秒為粒度切段，每段獨立生成 visual 描述與 audio 字幕，組合成結構化 timeline text。此 timeline text 取代影片本身，作為後續所有訓練 / 推論的視聽輸入。

#### 3.2.1 工具鏈

| 工具 | 模型 | 用途 |
|---|---|---|
| 視覺描述 | **LLaVA-NeXT-13B 4-bit** | 對每 10 秒 segment 抽 4 frames，生成自然語言描述 |
| 語音字幕 | **Whisper-Large-v3** | 對每 10 秒 segment 跑 ASR，輸出文字字幕（無人聲時輸出空字串）|

**設計理由**：v4.x 計畫已驗證此工具鏈在 RTX 5090 32GB 4-bit 模式下穩定運行，每 segment 處理時間 ~16 秒。

#### 3.2.2 Segment 粒度設計

採用 **10 秒固定粒度**，每影片切 ceil(duration / 10) 段（PerVidCom ~21s → 平均 3 段 / 影片）。

**為何選 10 秒**：
- 5 秒：LLaVA-NeXT 對 5 秒 clip 描述過於簡短
- **10 秒：剛好（資訊密度足、未過度碎片化）**——v4.x 經驗顯示
- 整支：失去時序資訊，且難以對齊 Demo §8 的 reasoning UI

**長片擴展**：使用者上傳 5 分鐘長片時，同樣以 10 秒粒度處理 → 30 段 timeline。**訓練分佈與長片推論在「每段 10 秒」這個粒度上完全對齊**。

#### 3.2.3 LLaVA-NeXT 描述 Prompt（細粒度提示，避免簡短輸出）

```
For this {seg_length}-second video clip, describe in 30-60 words:
1. Who or what appears (people, objects, setting)
2. What action is happening
3. Visual mood/atmosphere (color, lighting, energy)

Output a single concise paragraph. Do not include disclaimers.
```

#### 3.2.4 Timeline Text 格式

每影片產出一段結構化 text，例如 21 秒籃球短片：

```
[0:00-0:10] visual: A basketball player in red jersey dribbles past a defender 
            and drives toward the basket with high energy and tension.
            audio: "Bro look at this move"
[0:10-0:20] visual: The player attempts a layup but the ball rolls off the rim. 
            Camera cuts to a slow-motion replay from another angle.
            audio: "Oh no he missed it!"
[0:20-0:21] visual: Crowd reaction shown.
            audio: ""
```

**儲存格式**：JSON Lines per video，欄位：
```json
{
  "video_id": "...",
  "duration": 21.3,
  "segments": [
    {"start": 0, "end": 10, "visual": "...", "audio": "..."},
    {"start": 10, "end": 20, "visual": "...", "audio": "..."},
    ...
  ]
}
```

#### 3.2.5 Pipeline 工程

複用 v4.x `experiments/scripts/` 已驗證的 LLaVA-NeXT + Whisper 處理腳本，僅改資料來源（v4.x 自爬影片 → PerVidCom）：
- 平行處理（多 process / 多 GPU）：~2-3 天
- 序列處理：~7 天
- Retry / checkpoint 機制：v4.x 已驗證對 1352 segments 達到 0 failed segments

### 3.3 Phase 0 — Persona 抽取（一次性，訓練前完成）

#### 3.3.1 用戶 Style Embedding

對每個 train set 用戶 u 蒐集其所有留言 {c_1, ..., c_n}，計算 user-level style vector：

$$
\mathbf{e}_u = \frac{1}{n} \sum_{i=1}^{n} \text{StyleEnc}(c_i)
$$

**StyleEnc 候選**（待實驗比較）：
- **LUAR** (Rivera-Soto et al., 2021)：authorship-style 對比學習，content-independent
- **Style-Embedding** (Wegmann et al., 2022)：social media 風格 embedding
- **all-MiniLM-L6-v2**：通用 sentence embedding，作為 baseline

預期 LUAR 或 Style-Embedding 表現較好（content-independent）。

#### 3.3.2 Clustering

對所有 train set 用戶的 e_u 進行 K-Means clustering。

**K 的選擇**：在 K ∈ {4, 8, 16, 32} 比較：
- Silhouette score（內部一致性）
- 各 cluster 的可解釋性（人工檢視 + LLM-as-judge）
- 下游生成任務的 FICL-Score（功能性指標）

#### 3.3.3 Persona 描述生成

每個 cluster 抽 100 個 representative users 的所有留言（或 cluster 全部留言取樣 500 則），餵給 Gemini-1.5-Pro 生成 persona 描述。

**Persona 描述格式**（中粒度，~150 字）：
- 興趣主題（topics they engage with）
- 語氣特性（tone: sarcastic / earnest / playful / analytical）
- 句式習慣（短句 / 長句、emoji 使用、口頭禪）
- 典型留言示例（3-5 則）

每個 cluster 產出**一段固定的 persona 描述**，整個訓練過程不變。

### 3.4 Phase 1 — Multi-LoRA 訓練

#### 3.4.1 兩個變體的訓練設計

本研究**同時訓練兩個變體**作為對照，回答 RQ2：

| 變體 | 描述 | 訓練成本 |
|---|---|---|
| **A. Single LoRA + persona token (baseline)** | 全部 train set 訓 1 個 LoRA，prompt 中含 persona token "type-{k}" | ~3 GPU hours |
| **B. Multi-LoRA, K personas (main method)** | 將 train set 依 persona 切成 K 份，每份訓練一個獨立 LoRA | ~K × 1.5 GPU hours |

**Base model**：**Llama-3.1-8B-Instruct**（純文字 LLM）
- 英文表現好（PerVidCom 為英文）
- 128k context window，5 分鐘長片 timeline (~3000 tokens) 完全容納
- HuggingFace + PEFT 生態完整
- 對標 PV-LLM 從 head-to-head 變為 alternative approach（input format 不同）

**LoRA 設定**（兩變體共用，保守配置）：
- rank = 16
- alpha = 32
- target modules = q_proj, k_proj, v_proj, o_proj
- learning rate = 5e-5
- epochs = 2
- batch size = 8 (gradient accumulation × 8)

#### 3.4.2 訓練 Prompt 模板

**訓練資料中不再出現 user_id 也不再出現原始影片**，分別被 persona ID 與 timeline text 取代。

**訓練 prompt（兩變體共用格式）**：

```
You are viewer type-{k}: {one-line persona summary}.

You watched this video:
{timeline_text}

Write a comment in your typical style.
```

範例（k=3，21 秒籃球短片）：

```
You are viewer type-3: a sports enthusiast with sarcastic tone, 
often makes age-related jokes and uses casual short sentences.

You watched this video:
[0:00-0:10] visual: A basketball player in red jersey dribbles past a defender 
            and drives toward the basket with high energy and tension.
            audio: "Bro look at this move"
[0:10-0:20] visual: The player attempts a layup but the ball rolls off the rim. 
            Camera cuts to a slow-motion replay from another angle.
            audio: "Oh no he missed it!"
[0:20-0:21] visual: Crowd reaction shown.
            audio: ""

Write a comment in your typical style.
```

**訓練目標**：給定上述 prompt（含 timeline text），最大化該 persona 真實用戶 ground truth comment 的 likelihood。

**設計理由**：
- Neeko-inspired minimal prompt：persona ID + 一句 summary 作為 explicit anchor
- Timeline text 取代 raw frames：模型不再依賴 vision encoder，純文字處理
- **不塞用戶歷史留言**——應用情境是「使用者選 persona」

#### 3.4.3 訓練樣本組成

每筆訓練樣本：
- **Timeline text**：來自 §3.2 預先生成的 JSON 檔
- **Prompt**：上述模板，{k} 為該 user 所屬 persona ID
- **Target**：該 user 對該影片的真實 comment

**樣本切法**：以 comment 為單位。
- 變體 A：全部 ~212k 樣本訓練 1 個 LoRA
- 變體 B：依 persona 切 K 份，平均每 persona ~212k/K ≈ 27k 樣本（K=8 時）

#### 3.4.4 推論流程（應用情境）

**輸入**：
- 使用者從 K 個預設 persona 按鈕**單選一個**（例如 persona-3）
- 使用者上傳影片 v

**流程**：
1. 對 v 跑 §3.2 timeline 生成 pipeline → timeline_text(v)
2. 變體 A：載入唯一 LoRA；變體 B：載入 LoRA_3
3. 組 prompt（與訓練格式 1:1 對齊，含 timeline_text）
4. LoRA forward 生成 comment

**訓練/推論完全對稱**，prompt 格式一致。

**長片自然支援**：5 分鐘影片 → 30 段 timeline → ~3000 tokens prompt → Llama-3.1-8B context (128k) 完全容納，**不需 hierarchical / chunk-based hack**。

**設計範圍說明**：
- ✅ 使用者單選一個預設 persona → 載入對應 LoRA
- ❌ **不支援自訂 persona**（為 Neeko-style soft routing，列為 §11 future work，但 Demo §8.5 提供 exploratory 實作）
- ❌ 不做 hard / soft routing 比較

### 3.5 Phase 1.5 — Long Timeline Synthesis（合成長 timeline 訓練資料）

**動機**：PerVidCom 全部是 ~21 秒短片，訓練分佈內 timeline 僅 ~3 段。直接讓模型在推論時面對 30 段長 timeline (5 分鐘長片) 會有 train-test length mismatch 問題（生成過短、attention 偏向結尾、風格漂移等）。

**做法**：在 Phase 1 主訓練資料外，**合成假長 timeline 訓練樣本**，讓模型同時學會處理長 input。

#### 3.5.1 學術 Grounding

此方法直接 follow 三條學術 lineage：

- **Self-Instruct 範式** (Wang et al., ACL 2023)：用 LLM 生成訓練資料訓練 LLM
- **LIFT framework** (2024, arXiv 2502.14644)：「使用精心設計的 LLM 生成合成 task 來增強長 context 理解」——本研究 Phase 1.5 的最直接先例
- **OpenCharacter** (Wang et al., 2025, arXiv 2501.15427)：persona + 合成資料 fine-tune LLaMA-3-8B 達到接近 GPT-4o 的性能
- **LongLoRA** (Chen et al., ICLR 2024)：證明 LoRA + 長 context 擴展是有效路徑

#### 3.5.2 合成策略：Segment Recombination

**核心想法**：從 PerVidCom 真實 timeline 段落「重組」成假長 timeline，target comment 仍對應原本的核心段落。

```python
def synthesize_long_sample(real_sample, target_n_segments=10):
    """
    real_sample: PerVidCom 一筆樣本 (3 段 timeline + comment)
    target_n_segments: 想合成多長（例如 10 / 20 / 30 段）
    """
    target_comment = real_sample.comment
    core_segments = real_sample.timeline  # 3 段
    
    # 從其他影片隨機抽段落作為「干擾」
    n_distractors = target_n_segments - len(core_segments)
    distractors = random.sample(other_videos.segments, n_distractors)
    
    # 隨機交錯（保留 core segments 的相對順序）
    fake_long_timeline = interleave_keep_order(core_segments, distractors)
    
    return {
        "timeline": fake_long_timeline,  # 10/20/30 段
        "persona_id": real_sample.persona_id,
        "comment": target_comment,  # 不變
        "synthesis_meta": {
            "core_indices": [i for i, seg in enumerate(fake_long_timeline) if seg in core_segments],
            "n_distractors": n_distractors
        }
    }
```

**為什麼這樣可行**：
- 模型學到「在長 timeline 中找到關鍵段落 → 生成 comment」
- target comment 不需要重新生成（用真實 ground truth）
- 不需要長片 ground truth 資料集，是合成出來的

#### 3.5.3 替代合成策略：LLM-Generated Comments for Synthetic Long Timelines

對少部分樣本，可進階用 Gemini-1.5-**Flash**（依「Smaller, Weaker, Yet Better」, Google DeepMind 2024 — Flash 比 Pro 在合成資料上更 compute-optimal）合成 comment：

```python
# 對少數真正屬於「長片風格」的合成樣本
def llm_synthesize_long_sample(persona_desc, fake_long_timeline):
    prompt = f"""You are a YouTube viewer with this persona: {persona_desc}.
    You watched this video:
    {format_timeline(fake_long_timeline)}
    
    Write a comment in your typical style."""
    
    return Gemini_Flash.generate(prompt)
```

僅用於少量補充（例如 10% 樣本），主力仍是 §3.5.2 的 segment recombination。

#### 3.5.4 訓練資料混合比例（Curriculum）

採用 curriculum learning（Bengio et al., 2009 風格）：

| Epoch | 真實 PerVidCom 樣本 | 合成短長片 (5-10段) | 合成中長片 (10-20段) | 合成長片 (20-30段) |
|---|---|---|---|---|
| 1 | 100% | 0% | 0% | 0% |
| 2 | 60% | 30% | 10% | 0% |
| 3 | 40% | 25% | 25% | 10% |

模型逐步適應更長 input，避免訓練不穩定。

#### 3.5.5 工程成本

- 合成樣本生成（segment recombination，純邏輯）：< 1 天
- LLM 補充合成（Gemini-Flash，~10k 樣本 × $0.001）：~$10、半天
- 訓練時間：原 ~12 hours × 1.5 (epoch 增加) ≈ **18 GPU hours**
- 總增加：**~1-2 週工程**（可在 Week 6-7 與 Phase 1c 並行）

#### 3.5.6 Ablation 預留位置

§4.4 ablation 將比較：
- 不做 Phase 1.5（純 PerVidCom 短片訓練）
- 做 Phase 1.5 但只用 segment recombination
- 做 Phase 1.5 含 Gemini-Flash 合成
- 不同 curriculum 配比

回答「合成長 timeline 訓練是否真的提升長片推論品質」這個 RQ。

### 3.6 Sanity Check（Stage 1 驗收條件，亦為 demo 預留）

訓完 LoRA 後測試是否仍能 follow 多種 prompt（保留 instruction-following）：

```python
# 測試 1：標準訓練分佈
"You are viewer type-3: {summary}. You watched: {timeline}. Write a comment."
# → 應產生符合 persona-3 風格的 comment ✓

# 測試 2：截斷 timeline (為 demo reasoning 預留)
"You are viewer type-3: {summary}. You watched: {timeline[:1段]}. 
What would you say at this moment?"
# → 應產生 persona-3 風格的短 reaction（不是 full comment）

# 測試 3：長片 timeline（為 demo 長片支援預留）
"You are viewer type-3: {summary}. You watched: {30 段 timeline 模擬 5 分鐘長片}. Write a comment."
# → 應產生合理長度的 comment（不應因 prompt 長度爆炸而崩潰）
```

測試 2 對應 Demo §8 reasoning UI，測試 3 對應 Demo 長片支援。**若任一測試失敗，降低 LoRA rank、減少 epochs、或調整 prompt 模板重訓**。

---

## 4. Evaluation（論文用，僅 Stage 1）

### 4.1 主指標 — FICL-Score (Lin et al., 2024)

使用 Gemini-1.5-Flash 做 5×3 few-shot in-context learning auto-rater，評三個維度：
- **E**motion similarity
- **S**tyle similarity
- **R**elevance to ground truth

評估在 PerVidCom test set 上進行，與 Lin et al. (2024) 對齊。

### 4.2 評估協定的設計考量（input format 不同的處理）

**重要**：本研究與 Lin et al. PVCG 的 input format 不同：
- PV-LLM input：影片 frames + 5 則 user history
- 本研究 input：timeline-text + persona ID

兩個方法**無法 head-to-head 直接對比**。論文採用以下評估設計避免 over-claim：

#### 4.2.1 主對比：相同 input format 下的 fair comparison

訓練一個 **PV-LLM-text baseline**：使用 timeline-text input + 5 則 history conditioning（仿 PV-LLM prompt 格式），但 base model 仍是 Llama-3.1-8B-Instruct。這確保：
- 同 base model
- 同 input format (timeline-text)
- 唯一差別：history-conditioning vs persona-conditioning

→ 這個對比能證明「在 timeline-text setup 下，persona-conditioning 是否比 history-conditioning 好」。

#### 4.2.2 跨 setup 參考：對比原始 PV-LLM

也報告原始 PV-LLM (Lin et al., video frames + history) 的數字，作為**參考**而非公平對比。論文明確標示：

> "We do not claim head-to-head superiority over the original PV-LLM, as our input format (timeline-text) differs from theirs (video frames). The comparison serves as a reference point to position our method within the PVCG literature."

#### 4.2.3 兩種推論模式（保留前版本設計）

| Mode | 推論輸入 | 對應對標 |
|---|---|---|
| **Mode A: history-augmented** | persona ID + 5 則 user history + timeline | 對標 PV-LLM-text |
| **Mode B: persona-only** | persona ID + timeline | 反映工具型應用情境 |

### 4.3 對比方法（Baselines）

| Method | Input format | 是否本研究跑 | 性質 |
|---|---|---|---|
| **PV-LLM (Lin et al. 2024)** | video frames + history | ❌ 引用論文數字 | 跨 setup 參考 |
| **PV-LLM-text** | **timeline-text** + history | ✅ 本研究訓練 | **公平對比基準** |
| Video-LLaVA / Video-ChatGPT (zero-shot) | video frames | ❌ 引用論文數字 | 弱 baseline 參考 |
| Gemini-1.5-Flash / Pro (zero-shot) | video frames + history | ❌ 引用論文數字 | 上限參考 |
| Llama-3.1-8B (zero-shot, no LoRA) | timeline-text | ✅ 本研究跑 | timeline-text 弱 baseline |
| **Ours-A: Single LoRA + persona token** | timeline-text + persona | ✅ 本研究跑 | 內部 ablation |
| **Ours-B: Multi-LoRA, K personas** | timeline-text + persona | ✅ 本研究跑 | **本研究主貢獻** |

### 4.4 Ablation Studies

| Ablation | 目的 | 表格行 |
|---|---|---|
| Llama-3.1-8B zero-shot, no persona | 完全無訓練、無 persona | baseline 1 |
| Single LoRA, no persona | 訓練但無 persona conditioning | baseline 2 |
| Single LoRA + persona token | 變體 A：純 prompt conditioning | baseline 3 |
| K LoRAs, no persona token | 純架構效益（純資料切分）| ablation 1 |
| K LoRAs + persona token | 變體 B：完整方法 | **main** |
| K = 4 / 8 / 16 / 32 | cluster 數對效能影響 | RQ3 |
| Style encoder: LUAR vs Style-Emb vs MiniLM | 分群依據 | RQ3 |
| **Timeline 粒度: 5s / 10s / 整支** | timeline 設計影響 | **RQ4** |
| Timeline 內容: visual-only / audio-only / both | 模態貢獻 | RQ4 補充 |
| **No Phase 1.5 vs Segment Recombination vs +LLM-generated** | 合成資料策略影響 | **RQ5** |
| **長 timeline 推論評估 (10s / 60s / 5min)** | 長片擴展能力 | **RQ5 補充** |

### 4.5 Cluster Interpretability 評估

- **內部指標**：Silhouette score, Davies-Bouldin index
- **外部指標**：每個 cluster 的 representative comment 由 3 位人類標註者判斷「這是否為一個可辨認的留言類型」（5-point Likert）
- **LLM-as-judge**：Gemini-1.5-Pro 對 persona 描述的合理性打分

### 4.6 Human Evaluation（小規模）

對 50 筆 test 樣本，3 位標註者就 emotion / style / relevance 給 0-1 分，與 FICL-Score 計算 NDCG 對齊度（複現 Lin et al. 的驗證流程）。

---

## 5. Expected Contributions（論文）

1. **Timeline-text representation for PVCG**：提出以 LLaVA-NeXT (visual) + Whisper (audio) 預處理影片成結構化 timeline text，作為 PV-LLM frame-based 的 alternative 輸入路徑。**架構優勢**：訓練成本下降、視聽資訊結構化、與純文字 LLM 生態整合。
2. **Persona-conditioned multi-LoRA**：在 unseen-user PVCG 上採用顯式 persona archetype + multi-LoRA，補上 PV-LLM 缺乏顯式 user modeling 的空缺。
3. **Synthetic Long-Timeline Training (Phase 1.5)**：應用 LIFT (2024) 與 OpenCharacter (2025) 的合成資料方法論，透過 segment recombination + curriculum learning 擴展模型至長片推論，**以僅有 21 秒短片訓練資料的情況下，在合成長 timeline 評估上保持 persona 風格穩定性**。
4. **架構效益實證**：透過 4-row ablation 區分 "persona-conditioned prompt" 與 "persona-specific weights" 的個別貢獻，回答 RQ2。
5. **資料分析**：對 PerVidCom 16,702 用戶的 persona 分析，產出可解釋的觀眾分類及描述。
6. **可重現性**：完整公開 timeline 生成 pipeline、合成資料 pipeline、clustering pipeline、LoRA 訓練 config、prompt templates。

---

## 6. Risks and Mitigations

| 風險 | 影響 | 緩解 |
|---|---|---|
| **PerVidCom 公開資料拿不到（已實現）** | 高 | 1. Email Xudong Lin 求資料；2. 後備：依論文 protocol 自爬類似資料；3. 退路：改用 Personalized VideoIC（中長片彈幕）|
| **Timeline 生成品質不足（LLaVA-NeXT 描述太粗）** | 高 | 預跑 pilot test：對 50 部影片驗證 timeline 資訊密度；若不足，加 prompt engineering 或換成 LLaVA-NeXT-34B |
| Cluster 之間風格差異不顯著 | 高 | 預先做 sanity check：cluster 間 perplexity / style distance；若不顯著退回 single LoRA |
| Multi-LoRA 沒贏 Single LoRA | 中 | 即使如此仍有 negative result 價值；先做 Single LoRA 跑通 pipeline 再升級 Multi-LoRA |
| Multi-LoRA 訓練時 K 個 cluster 樣本不均 | 中 | 對小 cluster 做 oversampling；最少 cluster size 設下限 |
| **Reviewer 質疑「為何不直接餵 video」** | 中 | 主動承認 input format 不同；提供 PV-LLM-text baseline 做相同 setup 對比；強調長片擴展優勢 |
| **Reviewer 質疑「LLaVA-NeXT 是否決定上限」** | 中 | 在 ablation 比較不同視覺描述工具（LLaVA-NeXT vs Gemini Vision API）的影響 |
| Timeline 預處理時間過長（>3 天）| 低 | RTX 5090 32GB + 4-bit 量化已驗證；多 process 平行化 |

---

## 7. Timeline (14 weeks)

| Week | 任務 | 產出 |
|---|---|---|
| 1 | PerVidCom 資料下載 + 探索 + Pilot test (10 部影片驗證 10s timeline 粒度) | 資料統計報告 + timeline 設計確認 |
| 2 | **Phase -1: 對全 PerVidCom 跑 timeline 生成 pipeline** (LLaVA-NeXT + Whisper, ~2-3 天 GPU) | Timeline JSON dataset |
| 3 | Phase 0: Style embedding 比較 + Clustering + Persona 描述生成 | K 個 persona 及描述 |
| 4 | **Phase 1a: Single LoRA (變體 A) 訓練** | Variant A baseline |
| 5 | **Phase 1b: PV-LLM-text baseline 訓練**（fair comparison）| PV-LLM-text checkpoint |
| 6 | **Phase 1c: Multi-LoRA (變體 B) 訓練** (K=8) | K 個 trained adapter |
| 7 | **Phase 1.5: 合成長 timeline 資料 + 重訓 K LoRAs (curriculum)** | Long-extended K LoRAs |
| 8 | Sanity check + 推論 pipeline | 通過 §3.6 測試 |
| 9 | Stage 1 完整評估 (Mode A + Mode B + 對 PV-LLM-text)| 主結果表格 |
| 10 | Ablation studies (4-row + K + StyleEnc + timeline 粒度 + synthetic data) | RQ2-RQ5 結果 |
| 11 | Cluster interpretability + human eval + 長 timeline 推論評估 | RQ3 + RQ5 結果 |
| 12 | **專題 Demo 開發**：Mode 1 + Mode 2 (Custom Fusion) + 任意長度影片支援 | Demo working |
| 13 | 論文初稿撰寫 | Draft v1 |
| 14 | 論文修訂 + 投稿 | Submit |

---

## 8. Stage 2 — 專題 Demo 應用（不寫入論文主體）

> **此章節描述專題 demo 的設計，純粹作為產品 showcase。所有 demo 行為不寫入論文主體，不需要學術評估或 grounding 證明。**

### 8.1 Demo 目標與 Persona 選擇模式

展示 SimLens 作為「觀眾留言模擬工具」的應用樣貌：使用者選擇 persona + 上傳影片，看到 agent **邊看邊產生 reaction、最後寫出留言**。

**Demo 提供兩種 persona 選擇模式**：

| Mode | 使用者操作 | 後端行為 | 對應章節 |
|---|---|---|---|
| **Mode 1: Pre-defined** | 從 K 個預設 persona 按鈕**單選一個** | 載入對應 LoRA_k | §8.3 / §8.4 |
| **Mode 2: Custom Fusion** | 用自然語言描述「我想看哪種觀眾」或上傳幾則範例留言 | 計算 query embedding → 找 top-N 相似 persona → output-層加權融合 | §8.5 |

**設計範圍**：
- ✅ Mode 1 Mode 2 都實作於 Demo
- ❌ **論文僅評估 Mode 1**（單選預設 persona），Mode 2 不報告任何指標
- ❌ Mode 2 為 Neeko-inspired exploratory feature，不主張學術有效性

### 8.2 影片長度政策

| 影片長度 | Demo 行為 | UI 標示 |
|---|---|---|
| ≤ 30 秒 | 統一 §8.3 timeline pipeline | 無特別標示 |
| 30 秒 – 5 分鐘 | 統一 §8.3 timeline pipeline（更多 reaction 段）| 無特別標示（native support）|
| > 5 分鐘 | 不支援 | UI 拒絕並提示 |

**設計範圍**：碩論 Demo 支援任意長度 ≤ 5 分鐘。**不再區分 short / long mode**——timeline-text 結構讓兩者走同一 pipeline，只是 timeline 段數不同。

### 8.3 統一推論流程（任何長度，timeline-based）

得益於 timeline-text representation，**不再需要區分 short-mode 和 long-mode 的核心 pipeline**。短片（21 秒）和長片（5 分鐘）走同一條 pipeline，只是 timeline 段數不同。

```python
def demo_inference(video, persona_k):
    # Step 1: 跑 Phase -1 timeline 生成（與訓練時相同 pipeline）
    timeline = generate_timeline(video, segment_length=10)
    # 21s 影片 → 3 段 timeline；5 分鐘影片 → 30 段 timeline
    
    # Step 2: Reaction trace（per-segment progressive prompt）
    load(LoRA_k)
    reactions = []
    for i, seg in enumerate(timeline):
        partial_timeline = timeline[:i+1]  # 累積到當前段
        prompt = f"""You are viewer type-{k}: {summary_k}.
You're watching a video. So far you've seen:
{format_timeline(partial_timeline)}
Earlier reactions: {reactions[-3:]}
What would you say at this moment? (1-2 short sentences)"""
        r = LoRA_k.generate(prompt)
        reactions.append((seg.end, r))
    
    # Step 3: Final comment（完整 timeline + 所有 reactions）
    final_prompt = f"""You are viewer type-{k}: {summary_k}.
You watched this video:
{format_timeline(timeline)}

Your reactions during viewing:
{format_reactions(reactions)}

Now write a public comment for this video."""
    final_comment = LoRA_k.generate(final_prompt)
    
    return reactions, final_comment
```

**長片自然支援的關鍵**：
- Timeline 是文字 → 5 分鐘的 30 段 timeline = ~3000 tokens
- Llama-3.1-8B-Instruct context 128k → 完全容納
- **沒有 chunk-based hack、沒有 hierarchical hack、沒有 OOD claim**
- 訓練分佈和長片推論在「每段 10 秒 timeline」這個粒度上完全對齊

### 8.4 影片長度與 reaction 觸發頻率

| 影片長度 | Timeline 段數 | Reaction 觸發點 | UI 表現 |
|---|---|---|---|
| ≤ 30 秒 | 1-3 段 | 每段一個 reaction | 簡潔 panel |
| 30 秒 – 5 分鐘 | 3-30 段 | 每段一個 reaction，UI scrollable | 帶 timestamp |
| > 5 分鐘 | 不支援 | UI 拒絕 | 提示訊息 |

**長片 reaction 顯示策略**：
- 每段 10 秒一個 reaction → 5 分鐘 30 個 reaction → UI scrollable
- 隨影片進度逐段 reveal（前端動畫配合，後台一次跑完）
- 對長片，可選 "summary mode"：只顯示重點 N 個 reaction（依 LLM 判斷）

### 8.5 Custom Persona Fusion Mode（Mode 2，**exploratory**）

**動機**：使用者可能有不在預設 K 個 persona 中的需求（例如「想看 K-pop 飯圈會怎麼留言」）。Mode 2 允許使用者用自然語言描述或範例留言，系統路由到最相似的 N 個 persona 並融合 LoRA。

**參考**：Neeko (Yu et al., EMNLP 2024) §3.2 的 gating + role embedding 設計。本研究採用簡化版（output-layer weighted ensembling），不訓練 gating network。

#### 8.5.1 推論流程

```python
# Pre-computed (訓練後固定)
# - K 個 persona 描述 {desc_1, ..., desc_K}（Phase 0 LLM 生成的 persona summary）
# - 對應 K 個 LoRA: {LoRA_1, ..., LoRA_K}
# - K 個 persona embedding: e_k = SentenceEncoder(desc_k)

def custom_fusion_inference(user_query, video, top_n=3, temperature=0.5):
    # Step 1: 將使用者輸入轉成 query embedding
    # 使用者可輸入: (a) 自然語言描述, (b) 上傳的範例留言（concat 後 encode）
    query_emb = SentenceEncoder(user_query)
    
    # Step 2: 計算和所有 persona 的 cosine similarity
    sims = [cos(query_emb, e_k) for k in range(K)]
    
    # Step 3: 取 top-N 並 softmax weighting
    top_idx = topk(sims, n=top_n)
    raw_weights = [sims[i] for i in top_idx]
    weights = softmax(raw_weights / temperature)
    
    # Step 4: Output-layer weighted ensemble (避免 LoRA 權重平均的 noise)
    # 對 N 個 LoRA 各自跑一次 forward，再加權平均 logits
    prompt = build_fusion_prompt(top_idx, weights, ...)  # prompt 中標明融合來源
    
    fused_logits = sum(
        w * load_lora(LoRA_i).forward(prompt, frames=video).logits
        for w, i in zip(weights, top_idx)
    )
    
    comment = decode(fused_logits)
    return comment, top_idx, weights  # 回傳 transparency info 給 UI
```

#### 8.5.2 設計選擇

| 設計 | 採用 | 理由 |
|---|---|---|
| **融合層級** | Output-layer (logits 加權) | 比 LoRA 權重平均穩定，避免低秩矩陣加權後的 rotational ambiguity |
| **N（融合幾個）** | 3 | 太少 = 退化成 hard routing；太多 = noise 累積 + 推論成本爆炸 |
| **Temperature** | 0.5（可調）| 控制 weights 集中度；低溫接近 hard routing、高溫接近 uniform |
| **Encoder** | sentence-transformers/all-MiniLM-L6-v2 或 LUAR | 短描述用 MiniLM、範例留言用 LUAR |
| **Sim 門檻** | 設下限（如 sim < 0.3 警告）| 使用者輸入太離群時 fallback 提示 |

#### 8.5.3 推論成本

- 每次生成需跑 N 次 LoRA forward（N=3 → 推論時間 ×3）
- Demo 可接受（背景批次跑、配合 UI 動畫）
- 不影響論文（Mode 2 不評估）

#### 8.5.4 Mode 2 學術界線

- **不寫入論文主體**，僅 Demo 功能
- 不報告 Mode 2 的任何指標
- 為 Neeko-style soft routing 的簡化實作，未做學術驗證
- 論文 §11.1.2 中提到「Neeko-style soft routing 為 future work（含完整評估）」，本 Demo 的實作不算「已完成的 future work」

### 8.6 Demo UI

**Persona 選擇介面（Mode 1 + Mode 2）**：
```
┌─────────────────────────────────────────────┐
│ Choose persona:                             │
│                                             │
│ ● Pre-defined personas (Mode 1)             │
│   ◆ Sports Fan       ◆ Casual Emoji User    │
│   ◆ Detail Analyst   ◆ ...                  │
│                                             │
│ ○ Custom (Mode 2, exploratory)              │
│   ┌─────────────────────────────────────┐  │
│   │ Describe your target viewer or       │  │
│   │ paste 3-5 example comments           │  │
│   └─────────────────────────────────────┘  │
│   [Find similar personas →]                 │
│                                             │
│ Top-3 matched personas (Mode 2 only):       │
│   1. Sports Fan       weight = 0.62         │
│   2. Casual Emoji     weight = 0.25         │
│   3. Detail Analyst   weight = 0.13         │
└─────────────────────────────────────────────┘
```

**Short-mode UI**（≤ 30 秒）：
```
┌─────────────────────────┐  ┌────────────────────────┐
│                         │  │ 💬 Live reactions       │
│      [Video player]     │  │ Persona: Sports Fan     │
│  ▶━━━━━╋━━━━━━━ 8s   │  │ [5s] "Bro that move..." │
│                         │  │ [10s] (waiting...)      │
│                         │  │ [15s] (waiting...)      │
│                         │  │ [20s] (waiting...)      │
└─────────────────────────┘  └────────────────────────┘
                              ┌────────────────────────┐
                              │ 📝 Final comment        │
                              └────────────────────────┘
```

**Long video UI**（≥ 30 秒，scrollable reaction panel；無 disclaimer，因為長片是 native support）：
```
┌─────────────────────────┐  ┌────────────────────────┐
│                         │  │ 💬 Live reactions       │
│      [Video player]     │  │ Persona: Sports Fan     │
│  ▶━━━━╋━━━━━━ 1:23   │  │ [0:00-0:10] "..."       │
│                         │  │ [0:10-0:20] "..."       │
│                         │  │ [0:20-0:30] (...)       │
│                         │  │ ↓ scrollable            │
│                         │  └────────────────────────┘
└─────────────────────────┘  ┌────────────────────────┐
                              │ 📝 Final comment        │
                              └────────────────────────┘
```

**Mode 2 UI 額外要素**：
- Top-N 結果展示（含權重）：transparency，使用者看得到融合來源
- ⚠️ Exploratory mode banner
- 若最高 sim 低於門檻 → 顯示「Your description doesn't strongly match any persona. Try rephrasing or use Mode 1.」

### 8.7 Demo 設計約束

為了讓 Stage 2 純 prompt 即可運作（不重訓），Stage 1 訓練必須做對：
- ✅ 從 instruction-tuned base model fine-tune (Llama-3.1-8B-Instruct)
- ✅ 使用自然語言 prompt（不用死格式 token）
- ✅ LoRA 配置溫和（rank=16, lr=5e-5, 2 epochs）
- ✅ 通過 §3.5 sanity check（特別是測試 2、3）
- ✅ Timeline 預處理一致（demo 推論時用同一 LLaVA-NeXT + Whisper pipeline）

### 8.8 學術誠實性聲明（**核心**）

**論文（§1-§7）與 Demo（§8）嚴格分離，學術主張不混淆：**

| 範疇 | 論文 | Demo |
|---|---|---|
| 影片長度 | PerVidCom ~21 秒（評估範圍）| 任何長度 ≤ 5 分鐘 |
| Persona 選擇 | 預設選單單選（Mode 1）| Mode 1 + Mode 2（Custom Fusion）|
| Reaction / Reasoning | **完全不提** | 統一 timeline-based pipeline |
| 是否報告指標 | ✅ FICL-Score 對 PV-LLM-text baseline（僅 Mode 1）| ❌ 不報告任何指標 |
| 長片宣稱 | 「architecture supports long videos via timeline length scaling」（架構層面論述）| Demo 實作 |
| 是否宣稱 Mode 2 融合有效 | ❌ **明確不宣稱** | Mode 2 UI 標明 exploratory |
| 學術主張 | timeline-text representation + persona-conditioned multi-LoRA | 無 |

**對 Demo 中產生的「reactions」不主張任何擬真性質**：
- 不是真實人類的觀影思考過程
- 不是 distilled from any reasoning corpus
- 是 LoRA 在訓練分佈內的延伸生成（每段 timeline 都是訓練分佈內的粒度）

**長片支援的學術立場（修訂後較強）**：
- 訓練分佈是 PerVidCom timeline (~3 段)，長片是同 pipeline 處理出更長 timeline
- **每段 10 秒粒度在訓練 / 推論完全一致**，不是 OOD
- 但**論文不報告長片 FICL-Score**（沒有長片 ground truth dataset）
- 論文可寫「**architecture-level** support for long videos through timeline length scaling, evaluated qualitatively in demo (§8)」

**對 Mode 2 (Custom Persona Fusion) 不主張學術有效性**：
- 簡化版 Neeko-style soft routing，未訓練 gating network
- 沒有跑 Mode 2 vs Mode 1 的對比評估
- 融合策略（output-layer ensembling, N=3, temperature=0.5）為工程啟發，非經實證選擇
- 完整評估之 soft routing 為 §11.1.2 future work

**論文中如何提到 SimLens Demo**：可在 §11 future work 或附錄寫一段「Application Showcase」，含長片 qualitative example、Mode 2 qualitative example，**不放任何長片 / Mode 2 的量化數字**。寫法範例：

> "To illustrate practical applicability, we deploy our model in the SimLens demo system. The timeline-text representation naturally extends to longer videos by scaling the timeline length, requiring no architectural changes. We additionally provide an exploratory custom-persona feature via output-layer LoRA ensembling (Neeko-inspired). We do not claim quantitative validity for these extensions; rigorous evaluation on long-form content and learned soft routing are left as future work (§11)."

---

## 9. Resource Requirements

- **GPU**：RTX 5090 32GB（已有）
- **Phase -1 Timeline 生成成本（一次性，可平行化）**：
  - LLaVA-NeXT-13B 4-bit：每 segment ~14s，~9,839 影片 × 平均 3 segments = 29,517 segments
  - Whisper-Large-v3：每 segment ~2s
  - 序列：~131 hours = **5.5 天**
  - 平行（多 process）：**2-3 天**
- **訓練成本**：
  - PV-LLM-text baseline：~3 GPU hours
  - 變體 A (Single LoRA)：~3 GPU hours
  - 變體 B (Multi-LoRA, K=8)：~12 GPU hours
  - **Phase 1.5 重訓 (含合成長 timeline + curriculum)**：~18 GPU hours
  - 總計：**~36 GPU hours**（純文字 LLM 比 video LLM 仍便宜）
- **API 預算**：
  - Persona 描述生成（Gemini-1.5-Pro）：~$5
  - **Phase 1.5 LLM 補充合成（Gemini-1.5-Flash, ~10k 樣本）**：~$10
  - FICL-Score 評估（Gemini-1.5-Flash）：~$30
  - 總計：~$45 USD
- **資料儲存**：
  - PerVidCom 影片：~50 GB
  - Timeline JSON（產出）：~500 MB（純文字）
  - LoRA checkpoints：~50 MB × 9 個 = ~450 MB

---

## 10. Related Work（核心參考文獻）

### 10.1 PVCG 任務與資料集
- **Lin, X. et al. (2024)**. *Personalized Video Comment Generation*. EMNLP Findings. — PerVidCom 資料集主要對標；PV-LLM 為 frame-based baseline
- **Wu, Y. et al. (2024)**. *Understanding Human Preferences: Towards More Personalized Video to Text Generation*. WWW. — Personalized VideoIC，方向類似但用彈幕資料
- **Salemi, A. et al. (2024)**. *LaMP: When Large Language Models Meet Personalization*. — Personalization benchmark

### 10.2 Multi-LoRA 與 Persona Modeling
- **Yu, X. et al. (2024)**. *Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing*. EMNLP. — Multi-LoRA per persona 啟發；本研究 prompt 設計參考其 minimal meta-prompt
- **Hu, E. et al. (2022)**. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR. — LoRA 基礎
- **Wang, X. et al. (2025)**. *OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas*. arXiv:2501.15427. — **persona + 合成資料 + LLaMA-3-8B 達到接近 GPT-4o**；本研究 Phase 1.5 直接最近鄰

### 10.3 合成資料與長 Context Fine-Tuning（Phase 1.5 核心 grounding）
- **Wang, Y. et al. (2023)**. *Self-Instruct: Aligning LMs with Self-Generated Instructions*. ACL. — **合成資料訓練的根本方法論**
- **Taori, R. et al. (2023)**. *Stanford Alpaca: An Instruction-following LLaMA model*. — Self-Instruct 的工程實踐
- **Hsieh, C-Y. et al. (2023)**. *Distilling Step-by-Step!*. arXiv:2305.02301. — 大模型生 rationale + answer 蒸餾小模型
- **Chen, Y. et al. (2024)**. *LongLoRA: Efficient Fine-tuning of Long-Context LLMs*. ICLR. — **LoRA + 長 context 擴展**
- **Yi, K. et al. (2024)**. *LIFT: Long Input Fine-Tuning*. arXiv:2502.14644. — **本研究 Phase 1.5 的最直接先例**：用 LLM 生成合成 task 增強長 context 理解
- **Faria, A. et al. (2024)**. *Smaller, Weaker, Yet Better: Training LLM Reasoners*. arXiv:2408.16737. — **證明用 weak LLM (Flash 而非 Pro) 生資料 compute-optimal**
- **Bai, Y. et al. (2024)**. *LongSkywork: Training Recipe for Extending Context Length*. arXiv:2406.00605. — 長 context 訓練 recipe

### 10.4 Style Embedding
- **Rivera-Soto, R. et al. (2021)**. *LUAR: Learning Universal Authorship Representations*. EMNLP.
- **Wegmann, A. et al. (2022)**. *Style-Embedding for social media writing style*.

### 10.5 預處理工具與 Base Model
- **Liu, H. et al. (2024)**. *LLaVA-NeXT*. — 視覺描述工具（Phase -1 timeline 生成）
- **Radford, A. et al. (2023)**. *Whisper-Large-v3*. — 語音字幕工具（Phase -1 timeline 生成）
- **Dubey, A. et al. (2024)**. *The Llama 3 Herd of Models*. — Base model (Llama-3.1-8B-Instruct)

---

## 11. Future Work / 開放問題

### 11.1 已被排除於本研究範疇的擴展（明確列出）

以下方向**有研究價值但不在 v5.0 範圍**，列為碩論後或期刊版工作：

#### 11.1.1 真實長片資料集上的 PVCG 量化評估
- **動機**：v5.0 Phase 1.5 提供合成長 timeline 訓練 + 評估，但仍缺真實長片 user comment ground truth 對比
- **挑戰**：(1) 需要真實長片 + user history 資料集（已驗證 YT-30M 不可用 — videoID hash、用戶留言密度過低）；(2) 需要重跑 PV-LLM 等 baseline 在長片上的對比
- **狀態**：v5.0 Phase 1.5 提供合成資料驗證、Demo 提供 qualitative 展示；真實長片量化驗證列為**期刊延伸版主題**

#### 11.1.2 Learned Soft Routing（完整 Neeko gating）
- **動機**：Demo §8.5 提供簡化版 output-layer ensembling，但未訓練 gating network
- **方法**：實作 Neeko gating network（global role embedding + Softmax weighting，可訓練）
- **預估工程量**：1-2 週訓練 + 評估
- **狀態**：碩論不做，未來可加入

#### 11.1.3 視覺描述工具的影響
- **動機**：LLaVA-NeXT-13B 4-bit 可能限制 timeline 品質
- **方向**：比較不同視覺描述工具（Gemini Vision API、GPT-4V、LLaVA-NeXT-34B）對下游 FICL-Score 的影響
- **狀態**：v5.0 ablation 中可選做；正式評估留作 future work

### 11.2 開放討論問題

1. K 的選擇是否應該依下游任務動態決定（end-to-end），而非預先 K-Means？
2. 同一用戶在不同類型影片下是否會展現不同 persona？（per-context vs per-user clustering）
3. Persona 描述應該手寫、LLM 生成、還是 learnable embedding？
4. Timeline 是否應該包含 high-level 摘要 + low-level 細節（hierarchical timeline）？
5. Reasoning trace（Demo §8）能否在未來成為合法學術主張？需要什麼形式的 grounding？

### 11.3 學術誠實性紅線（一次性整理）

本計畫嚴格遵守以下界線，避免 over-claim：

| 不會做的事 | 理由 |
|---|---|
| 在論文宣稱「在長影片上 SOTA / 打贏 X 方法」 | 沒有長片 ground truth 量化評估 |
| 在論文 head-to-head 對比原始 PV-LLM | Input format 不同（timeline-text vs frames）—— 改與 PV-LLM-text baseline 對比 |
| 在論文宣稱 Demo 的 reactions 反映真實人類思考 | 沒有 grounding |
| 在論文宣稱 Mode 2 (Custom Fusion) 是 SOTA soft routing | 簡化版實作，未訓練 gating network |
| 把 LLaVA-NeXT 描述當「ground truth visual content」 | 是 model 的描述，有偏差和遺漏 |

**論文評估範疇 = PerVidCom + timeline-text representation + persona-conditioned multi-LoRA + FICL-Score**，這個範疇內的所有主張都有實驗證據；範疇外的事一律放 future work 或 demo qualitative showcase，不放任何數字。
