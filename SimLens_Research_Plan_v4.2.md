# SimLens 完整研究計畫 v4.2.3
## "Event-Driven Persona-Conditioned Video **Commentary Generation** via Sparse Temporal Prediction and RLAIF, Anchored on Real-World YouTube Signals"

> 版本：v4.2.3
> 投稿目標：ACM MM 2026 BNI / UIST 2026 Posters / Demos / 智慧創新大賞 2026

> **v4.2.3 vs v4.1 重點差異**（詳見 §13 Changelog）：
> - **video-level train/test split**：100 train + 30 hold-out test，所有評估只在 test set 做
> - **Real-World Anchoring (2 個獨立外部訊號)**：
>   - Anchor A 文本錨：HuggingFace 187K YouTube comments dataset，3 個 deterministic 分布指標 (Length KS / Sentiment Wasserstein / Embedding Frechet)
>   - Anchor B 時序錨：**per-video Hotspot Recall / Precision @ ±5s** on 30 高-mention test 影片（每部 ≥5 條真實觀眾 timestamp mentions）。各模型對 30 test 跑完整 SimLens pipeline 預測 sparse JSON，比對輸出 timestamp 是否對齊真實觀眾留言中的 mention timestamp。學術背書：[Ref 37] SoccerNet CVPR 2018 tolerance window paradigm + [Ref 38] DanmakuTPP NeurIPS 2025
> - **TAS 降級**：Teacher Alignment Score 從主指標降為 Appendix 診斷指標
> - **Multi-judge spot-check**：80-sample Qwen3-32B-Q4 + GPT-4o-mini 平行評分，Cohen's κ 驗證
> - **2-aspect reward**：R_timing (50%) + R_content_quality (50%)
> - **K iteration ablation**：Iterative DPO rounds K∈{1,2,3} 為 ablation 超參

> **v4.2.3 vs v4.2.2 修正**（簡述）：
> - 砍掉 v4.2.2 的 Anchor B-2（Bilibili popular 跨平台分布級錨）— 原因：B-1 (YT mentions, back-loaded 51%) 和 B-2 (Bilibili 彈幕, front-loaded 43%) baseline 方向相反，推翻「universal reaction-time pattern」假設，兩個 anchor 互相矛盾無法同時對齊
> - 砍掉 v4.2.2 distribution-level KS（Mention Depth KS / Inter-gap KS）→ 改回 v4.2 原本想要的 per-video Hotspot Recall 強 claim
> - Test set 換成「高 mention」篩選版（每部 ≥5 mentions），確保 per-video KDE peak 統計信度
> - 既有低 mention 30 test set 歸檔到 `data/_archive_2026-05-06/manifest.test_v1_lowmention.jsonl`

---

## 0. 一頁摘要（The One-Page Summary）

```
研究問題：
   能否用 3B 小模型在「缺乏真實 persona 觀影行為資料」的場景下，
   訓練出能對短 YouTube 影片生成 persona-specific、時序定位精確的多受眾評論
   生成系統？
   （Scope: 1–3 minute YouTube videos，刻意排除 ≤60s Shorts —— 詳見 §1.4 scope rationale）

關鍵 framing：
   ✓ 定位為「事後反思型 AI 評論生成」（post-hoc commentary generation）
   ✓ 此定位與 SimTube、YouTube 真實留言情境一致
       —— 觀眾本來就是「看完整片才寫評論」
   ✗ 不 claim「模擬即時觀影體驗」

核心方法：
   Stage A：UMaT-inspired temporal alignment
     - Whisper（含時間戳）+ LLaVA-NeXT（每 10 秒分段）→ 結構化 Timeline Script
     
   Stage B：兩階段事件驅動訓練（One-Shot 全片輸入）
     - Phase 1（蒸餾）：一次餵全片 Timeline Script → Claude 輸出 Sparse JSON
                       [{timestamp, comment}, ...] → SFT Llama-3B
     - Phase 2（RLAIF）：Qwen3-32B 用 2-aspect reward → DPO

   Stage C：報告生成（同一個 Llama-3B base）
     - 從 (timestamp, comment) 列表整合出跨受眾比較與改善建議

關鍵差異化（vs SimTube）：
   SimTube：影片 → 整體理解 → per persona 1 條【整片評論，無 timestamp】
            （內部用 timestamp 對齊輸入模態，但摘要成 video summary 後丟掉時間軸）
   SimLens：影片 timeline (含 timestamp) → per persona N 條【sparse comments + timestamp】
            → 輸出 [(t1, c1), (t2, c2), ...]，N 由 persona 活躍度自決
   
   核心差異：SimTube 的 LLM 輸出捨棄時間軸；SimLens 的 LLM 輸出保留時間軸。
   對創作者實用價值差一個量級 —— 從「整體分數」變成「段落級回饋」。
   架構簡潔（per video × persona 1 次推理）、且符合真實 YouTube 留言行為情境。

預期貢獻：
   C1. System：首個 segment-localized persona-conditioned 影片評論生成系統
   C2. Method：one-shot timeline-to-sparse-JSON 蒸餾 + 2-aspect reward + DPO
              (None 反應自然編碼為「該 persona 不在此 timestamp 出現」)
   C3. Empirical：3B student 在時序定位 + persona 區分上接近 Claude，且 on-device
   C4. ⭐ v4.2 新增：Real-World Anchoring 範式
       採集 YouTube top comments + Bilibili 彈幕 timestamp 作為兩種互補外部錨：
         - 文本錨（同語言、英文 corpus）：4 個 deterministic distributional 指標
         - 時序錨（跨語言、彈幕 timestamp 密度）：2 個 crowd hotspot 指標
       打破 Claude 蒸餾 → Qwen reward → Qwen evaluation 的封閉迴路，
       提供獨立於兩個 LLM 訊號的真實世界對照。

明確不做（誠實 framing）：
   ✗ 留存曲線預測
   ✗ 跳出率預測
   ✗ 任何宣稱「即時行為模擬」的 claim
   ✗ Future-leakage 規避（我們大方承認看完全片才生成，符合 post-hoc 定位）
   ✗ 任何需要對應「真實段層級觀眾行為」的指標
   ✗ 對應「個別真實人類 ↔ 個別 SimLens persona」（Bilibili 彈幕僅取群體 timestamp 密度）
```

---

## 1. 系統架構（Architecture Overview）

```
┌──────────────────────────────────────────────────────┐
│        輸入：短 YouTube 影片 (1–3 min, 刻意排除 ≤60s Shorts) │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage A：UMaT-inspired 時序對齊 → 全局 Timeline Script  │
│                                                       │
│   Whisper-Large-v3                                    │
│     → 帶時間戳的整段轉錄                                 │
│     → [(t=0.5s, "Hi"), (t=2.1s, "today..."), ...]    │
│                                                       │
│   LLaVA-NeXT-13B                                      │
│     → 每 10 秒抽 4 frames + audio context              │
│     → N 個段描述（每段 ~150 字視覺敘述）                  │
│       N = ⌈video_duration / 10s⌉, typically 6–18      │
│       （1–3min 範圍對應 6–18 段，分段邊際價值高）         │
│                                                       │
│   時間軸對齊 → 全局 Timeline Script                      │
│     [00:00-00:10] Visual: ... | Audio: "..."          │
│     [00:10-00:20] Visual: ... | Audio: "..."          │
│     ...                                               │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage B：One-Shot Sparse JSON 評論生成（核心架構）       │
│                                                       │
│   Llama-3.2-3B-Instruct（4-bit GPTQ 量化）              │
│   + 8 個 LoRA Adapter（每個 persona 一個）              │
│                                                       │
│   對每個 (影片, persona_p)：                             │
│     Input: 全局 Timeline Script + persona_p YAML        │
│     Prompt: "You just finished watching this video.     │
│              Reflect and list moments you would have    │
│              left a comment, in JSON format."           │
│     Output (Sparse JSON):                              │
│       [{"timestamp": "00:15", "comment": "..."},       │
│        {"timestamp": "01:42", "comment": "..."}]        │
│                                                       │
│   產出：每影片 8 條 sparse list（不固定長度，由 persona  │
│        活躍度自然決定，「沒反應」= 空陣列 []）             │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage C：報告整合（同一個 Llama-3.2-3B，無 adapter）      │
│                                                       │
│   Input: 8 條 (timestamp, comment) 列表 + Timeline      │
│   Output: 結構化報告                                    │
│     - 跨受眾比較（哪些時段哪些 persona 留言）             │
│     - Persona 共鳴熱區（每 persona 集中發言的時段）       │
│     - 改善建議（沒人留言的時段、爭議時段）                 │
└──────────────────────────────────────────────────────┘
```

### 1.1 架構特徵摘要

| 維度 | SimLens 設計 | 說明 |
|---|---|---|
| 輸入方式 | 一次餵入全片 Timeline Script | UMaT-style structured text |
| 推理單位 | (video, persona) → 1 條 sparse list | 每 persona 對整片 1 次推理 |
| 輸出格式 | 稀疏 JSON `[{timestamp, comment}, ...]` | Chapter-Llama 範式（text-level timestamp output） |
| 「沒反應」編碼 | 不出現該 timestamp / 空陣列 | 自然嵌入輸出格式 |
| 模擬定位 | **事後反思評論**（post-hoc commentary）| 與 YouTube 真實留言情境一致 |

### 1.2 架構設計依據

| 設計元素 | 來源文獻 | 為什麼這樣選 |
|---------|---------|-------------|
| Whisper-Large-v3 + LLaVA-NeXT | SimTube (Hung et al., 2024) | 直接借鏡 SimTube multimodal pipeline |
| 10 秒等長分段策略 | **UMaT (Bi & Xu, 2025, arXiv 2503.09081)** | structured segmentation 細節依據（fixed-length avoid fragmentation）|
| **All-modality-as-text 哲學** | **Socratic Models (Zeng et al., ICLR 2023, arXiv 2204.00598)** | 將視覺 / 音訊全部降維為文字（"language-based world-state history"），讓 LLM 純在文字域推理 |
| **One-shot timeline → text-output timestamp** | **Chapter-Llama (Ventura et al., CVPR 2025, arXiv 2504.00072)** | LoRA-tuned LLM 從 ASR + caption + timestamp 文字 timeline 輸出 timestamp + 內容的最直接前例 |
| **Event-driven sparse prediction** | **MMDuet (arXiv 2411.17991), MM-When2Speak (arXiv 2505.14654)** | "VideoLLM Knows When to Speak"——本研究在 post-hoc 場景的延伸 |
| **事後反思 framing** | **YouTube comment 真實情境 + post-hoc commentary 定位** | 觀眾本來就是看完才留言，不需偽裝即時性 |
| Llama-3.2-3B 為 student | Meta Llama 3.2 release notes (2024) | 1B 太弱、8B 太大，3B 是甜蜜點 |
| 4-bit GPTQ 量化 | LLaMA-Factory 官方推薦工作流 | 在 24GB VRAM 跑得動 |
| Multi-LoRA per persona | **Neeko (EMNLP 2024)** | 已驗證 per-character LoRA 優於 single LoRA + prompt |
| Same model for generation + report | Tülu 3 multi-task post-training | 一模多用節省部署成本 |
| **JSON 結構化輸出** | **JSONSchemaBench (arXiv 2501.10868)、IFEval (arXiv 2311.07911)** | 結構化輸出評估的學術標準 |

### 1.3 三個關鍵設計決策的學術背書

#### 決策 A：為什麼分離「感知（Whisper + LLaVA-NeXT）」與「推理（Llama）」？

**潛在質疑**：「為什麼不把整部短 YouTube 影片直接餵給 Gemini 2.5 Flash 或 GPT-4o 一次到位？」

**學術背書 1：感知與推理分離有可解釋性與可除錯性優勢**
- **Socratic Models (Zeng et al., ICLR 2023, arXiv 2204.00598)** 提出「all-modality-as-text」哲學：將視覺/音訊全部降維為「language-based world-state history」，讓 LLM 純在文字域推理 —— 提供 interpretability 與 modular composition 能力。
- **VideoMultiAgents (Kugo et al., arXiv 2504.20091)** 在 Intent-QA 達 79.0%（+6.2% over SOTA），證明專門代理人 + 獨立文字報告能避免單一巨型模型的黑箱干擾。

**學術背書 2：原生多模態模型的時序理解仍有缺陷**
- **VBenchComp / Time Blindness 系列**：頂尖原生多模態模型（GPT-4o、Gemini）對影片存在「shuffling invariance」——影格打亂順序，輸出仍幾乎不變，顯示依賴語言先驗而非真實時序推理。

> **註**：本段提到的 GPT-4o / Gemini 是論述「即便最強原生多模態大模型也有時序盲區」的論證對象，引自 VBenchComp 文獻；本研究**實驗本身的 OpenAI baseline 一律使用 GPT-4o-mini**（預算決策，詳見 §5.1 / §8.2）。

#### 決策 B：為什麼將影片分段（每 10 秒）？

**學術背書 1：等長分段策略的細節依據**
- **UMaT (Bi & Xu, arXiv 2503.09081)** 明確指出，要在影片任務中維持語義與時間一致性，必須將視覺描述與 ASR 轉錄「依時間戳切分為結構化片段」；且採用 **fixed-length** 分段可避免 fragmentation。SimLens 採用 10s 等長分段直接借鏡此策略。
- 註：**Chapter-Llama (CVPR 2025)** 採用 ASR-guided 動態 frame selection（非等長），SimLens 為簡化實作與保證每段都有文字 + 視覺輸入，選擇等長分段路線。

**學術背書 2：分段能規避視覺模型的記憶體與品質下降問題**
- **QMAVIS (Lin et al., arXiv 2601.06573)** 證明 chunking + late fusion 在 VideoMME 長影片基準上比端到端原生多模態模型**準確率高 38.75%**。

#### 決策 C ⭐：為什麼採用「一次餵入 + 事後反思」？

**潛在質疑**：「為什麼不模擬即時觀影體驗，逐段累積決定反應？」

**學術背書 1：Post-hoc commentary 是更誠實的學術定位**
- 真實 YouTube 留言**本就是看完整片才寫**——觀眾並非邊看邊即時打字。事後反思 framing 與真實留言情境一致。
- SimTube (Hung et al., 2024) 本身也是事後對整片給評論，學界接受此 framing。
- 大方承認 post-hoc，scope 乾淨、無 future leakage 破口。

**學術背書 2：Text-level video-to-timestamp 範式有 CVPR 2025 直接前例**

SimLens 採用 **text-level path（影片轉文字 timeline → LLM 推理輸出 timestamp）**，
而非 **token-level path（修改 VLM 在視覺 token 加入 timestamp）**。
這兩條路線的代表工作對比如下：

| 路線 | 代表工作 | 機制 | SimLens 採用？ |
|---|---|---|---|
| **A. Token-level**：修改 VLM 視覺 encoder | VTG-LLM (AAAI 2025) | 影格 → 加 absolute-time token → VLM 內建時間理解 | ✗ |
| **B. Text-level**：影片轉文字 → LLM 推理 | **Chapter-Llama (CVPR 2025)**、Socratic Models (ICLR 2023)、UMaT (2025)、SimLens | 影片 → ASR + caption + timestamp 文字 → LLM 輸出 timestamp + 內容 | ✓ |

**為什麼 SimLens 選路線 B**：
- **Chapter-Llama (CVPR 2025, arXiv 2504.00072)** 是與 SimLens **機制最一致**的直接前例：
  Llama-3.1-8B + LoRA rank=8，輸入 `ASR [HH:MM:SS]: ...` + `Caption [HH:MM:SS]: ...`
  按 timestamp 排序的純文字 timeline，輸出 timestamp + 章節標題。
  在 VidChapters-7M 上 F1 從 26.7（VTG-style baseline）大幅提升至 **45.3**。
  **SimLens 採用同樣 pipeline，差別只在輸出內容（chapter title → persona commentary）**
  與 base model size（8B → 3B + per-persona LoRA）。
- **Socratic Models (Zeng et al., ICLR 2023, arXiv 2204.00598)** 提供哲學基礎：
  將所有模態（視覺 / 音訊）降維為「**language-based world-state history**」，
  讓 LLM 純在文字域推理 —— 是 SimLens Stage A Timeline Script 設計的祖師爺工作。
- **MMDuet (arXiv 2411.17991)** / **MM-When2Speak (arXiv 2505.14654)**：
  支援「event-driven sparse prediction」概念，但實作偏 streaming / token-level，
  作為 sparse output 概念的補充背書。
- **路線 A（VTG-LLM 等）需要訓練 / 修改 VLM 本身**，成本高、需 8B+ multimodal 模型，
  不適合 SimLens 的 24GB 消費級 GPU 部署目標。

**學術背書 3：工程效率**
- One-shot 設計：每 (video, persona) 1 次 Claude API call ≈ ~$0.015/call
- 100 影片 × 8 persona = 800 calls，總 ~$12 USD —— 在學生研究預算內。
- Chapter-Llama 在 1 hr 影片做到 single forward pass，SimLens 應用到 1–3 min 短影片是合理範圍縮減。

**對 SimLens 的意義**：post-hoc + one-shot + text-level sparse JSON 是建立在
Chapter-Llama (CVPR 2025) 與 Socratic Models (ICLR 2023) 等成熟工作上的範式延伸，
不偽裝即時性、不需修改 VLM 架構、scope 乾淨、reviewer 攻擊面最小。

### 1.4 Scope Rationale：為什麼是 1–3 分鐘 YouTube 影片，而不是 Shorts？

**SimLens v4.1 scope 嚴格鎖定：1–3 分鐘 YouTube 短影片，刻意排除 ≤60s Shorts。**

潛在質疑：「TikTok / Reels / YouTube Shorts 才是 short-form 主流，為什麼不做 Shorts？」

#### 排除 Shorts (≤60s) 的四個理由

**理由 1：分段架構在 Shorts 上邊際價值低**
- ≤60s 影片只有 1–6 段，segment-level 分析能展現的「persona 對不同段反應差異」訊息量受限
- LLaVA-NeXT 對 60s 整段直接處理仍在能力範圍內，**UMaT/QMAVIS 引用的「長影片需分段」motivation 站不穩**
- 60s 內 sparse JSON 平均只有 1–3 個 timestamp，sparse 與 dense 預測差別不顯著

**理由 2：Persona 區分能力需要足夠 sparse list 長度**
- 8 個 persona 的差異化主要體現在「**反應內容風格**」（口頭禪、情緒、視角）
- 60s 內 sparse list 平均只有 1–3 條評論，內容樣本太少難以展現 persona 風格特徵
- 1–3min 對應每 persona 平均 0–7 條 sparse list，足以累積 persona-specific 內容差異
- Persona Content Distinctiveness (Group 3) 在足夠樣本下才能穩定區分 8 個 persona

**理由 3：與 SimTube baseline 對標基準明確**
- SimTube (Hung et al., 2024) 實驗用的影片長度雖未嚴格限定，但實際分布偏向 1–10min 中短片
- SimLens 鎖定 1–3min 與 SimTube 主流區間有清楚交集，Table 1 baseline 數字可比
- 純 Shorts 場景下 SimTube 沒有對應實驗，跨域比較會失準

**理由 4：訓練資料與 prompt 約束連動匹配**
- Prompt-level expected_comment_count 以 2min 影片為 baseline（高活躍 persona 約 3-6 條 / 2min）
- 此活躍度提示需要「足夠長的時間軸來展開」才能讓 persona 風格差異化
- 30s 影片若強要高活躍 persona 留 3 條評論 = 每 10s 一條，**違反真實留言行為分布**

#### 為什麼不做 > 3min？

- **Token 預算**：3min × 6 段/min = 18 段 ≈ 2700 token timeline，加 persona ≈ 3000 token，還在 Llama-3.2-3B 的 4096 max_seq_length 內
- **Claude 蒸餾品質**：> 3min 影片 timeline 過長，Claude 對「該在哪些 timestamp 留言」的判斷品質下降（觀察 SimTube 也未處理 > 5min）
- **訓練資料異質性**：> 3min 影片內容類型分布過廣（教學、vlog 長集等），會稀釋 8 persona 的訓練訊號

#### 三個生態系比較表

| Scope | 影片數 | N 段 | UMaT 引用 | Persona 區分 | SimTube 對標 | 推薦度 |
|---|---|---|---|---|---|---|
| ≤60s Shorts | 6 段內 | 弱 | ✗ | 弱 | ✗ | ✗ 排除 |
| **1–3min YouTube ⭐** | **6–18 段** | **✓** | **✓** | **✓** | **✓** | **本研究採用** |
| > 3min 中長片 | > 18 段 | ✓ | △ | ✓ | △ | future work F6 |

#### 對 reviewer 的直接回應

> 「我們刻意不做 Shorts (≤60s)，因為 SimLens 的核心貢獻是 segment-level persona differentiation 與 sparse temporal localization。這兩個能力都需要足夠長的時間軸（6 段以上）才能展現。我們的 scope 與 SimTube 主流區間對齊，是有意的方法論選擇，並非資料蒐集限制。Shorts 與 > 3min 中長片皆列為 future work（F5、F6）。」

---

## 2. Persona 設計（從 PersonaChat 8K 取樣，不手動設計）

### 2.1 設計哲學：不重新造輪子

**SimLens 完全照搬 SimTube (IUI 2025) 的 PersonaChat-based persona selection 範式，**
**唯一延伸是「top-30 → top-8」與「per-video query → dataset-aggregated query」。**

為什麼採用：
- **PersonaChat (Zhang et al., ACL 2018)** 提供 8000+ 人工撰寫 persona descriptions，學界引用 4000+ 次，是 persona-based dialogue 的事實標準資料集
- **SimTube (Hung et al., IUI 2025)** 已驗證「PersonaChat + cosine similarity 取樣」能產生高品質、與影片內容相關的 personas
- **我們不發明新的 persona 集合**，避免「為什麼是這 8 個？」的 reviewer 質疑
- **完全可重現**：給定影片資料集 + embedding 模型 + random seed，任何人都能複製出同樣 8 個 personas

### 2.2 為什麼選 8 個 persona

```
數量決策：
- SimTube 用 top-30，但他們不訓練模型（純 prompting）
- SimLens 要訓 per-persona LoRA：30 個 LoRA × 100 樣本 = 訓練資料過稀
- PersonaGym (EMNLP 2025) 評估實驗每任務只用 5 個 personas，多樣性可能不足
  （註：PersonaGym 自帶 200 personas，但 SimLens 不使用其 personas，
   只借用其評估 rubric — 詳見 §4.3 / §5.2）
- 8 個是甜蜜點：每 persona 100 個 sparse list 樣本（足夠 LoRA 訓練）
              + 8 維度涵蓋主流受眾類型
              + Neeko (EMNLP 2024) 多角色實驗範圍 (3-10) 內
```

### 2.3 Persona 取樣流程（PersonaChat → top-8）

```python
# Step 1: 載入 PersonaChat
from datasets import load_dataset
personachat = load_dataset("bavard/personachat_truecased")
all_personas = personachat["train"]["personality"]   # 8K+ persona descriptions
                                                      # 每個 persona = 5 句以上自我描述

# Step 2: 對 100 部影片提取 keywords + 聚合
import openai
video_keywords = []
for video in videos_100:
    # 從 video 的 LLaVA caption + Whisper transcript 用 GPT-4o-mini 摘出 5-10 個 keywords
    kws = extract_keywords(video.timeline_script)
    video_keywords.extend(kws)
dataset_query = " ".join(set(video_keywords))   # aggregated query 涵蓋整批訓練影片

# Step 3: 用 OpenAI text-embedding-3-small 嵌入（同 SimTube 設定）
client = openai.OpenAI()
def embed(text):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

query_emb = embed(dataset_query)
persona_embs = [embed(p) for p in all_personas]

# Step 4: cosine similarity → top-K with diversity filter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
sims = cosine_similarity([query_emb], persona_embs)[0]

# top-N 候選（取 N=80），再以 MMR 過濾出 8 個多樣 personas
top_n_idx = np.argsort(sims)[-80:][::-1]
selected = mmr_diversity_filter(top_n_idx, persona_embs, k=8, lambda_=0.6)
                                             # MMR: Maximal Marginal Relevance
                                             # lambda_=0.6 平衡相關性 vs 多樣性

PERSONAS = [all_personas[i] for i in selected]
                                             # 8 個 PersonaChat-format personas

# Step 5: 為每個 persona 補上 expected_comment_count（唯一手動補欄位）
# 由 Claude zero-shot 估計（基於 persona 描述推斷活躍度）
# ⚠ v4.2 用法說明：此欄位僅作為 prompt-level constraint（提示模型留言條數區間），
#                  不再進入 RLAIF reward 訊號（避免「Claude 估的活躍度」用來訓練學自 Claude 的 Student 形成循環）
for p in PERSONAS:
    p["expected_comment_count_range"] = claude_estimate_activity(p)
                                             # e.g., (0, 2) for introvert
                                             #       (3, 6) for extrovert
```

### 2.4 與 SimTube 範式的兩個延伸（SimLens-specific）

| 維度 | SimTube (IUI 2025) | SimLens (本研究) | 為什麼延伸 |
|---|---|---|---|
| **取樣數量** | top-30 personas | **top-8 personas + MMR** | per-persona LoRA 需要每 persona 充足訓練資料 |
| **Query 範圍** | 單部影片的 keywords | **整批 100 部影片 aggregated keywords** | 全資料集需共用同 8 個 personas，不能每片切換 |
| **Persona 內容** | PersonaChat 原文不動 | **PersonaChat 原文 + expected_comment_count（prompt-level）** | 提示模型留言條數區間（非 reward 訊號）|

### 2.5 Persona 範例（從 PersonaChat 8K 抽樣後預期長相）

> 實際 8 個 personas 在 Week 1 跑完取樣腳本後產生。以下為 PersonaChat 原始格式範例（非手選結果）：

```
範例 Persona（PersonaChat 原始格式）：
  "i am a youtuber. i make videos about makeup.
   i have a pet cat named whiskers.
   i love drinking iced coffee in the morning.
   i go to the gym three times a week."

  + (SimLens 補) expected_comment_count_range: (3, 5)
```

每個 persona 都是 PersonaChat 中真實人撰寫的 5 句以上自我描述，**內容由資料集決定，不是我們編的**。

### 2.6 Persona 設計學術依據

| 設計元素 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| **PersonaChat 8K 為來源** | **PersonaChat (Zhang et al., ACL 2018)** | 8000+ 標準 personas, 學界事實標準 |
| **Cosine similarity 取樣** | **SimTube (Hung et al., IUI 2025)** | 直接照搬其 PersonaChat → top-K 範式 |
| **OpenAI text-embedding-3-small** | **SimTube (Hung et al., IUI 2025)** | 同款 embedding 模型，可重現 |
| **MMR 多樣性過濾** | Carbonell & Goldstein (SIGIR 1998) | 經典 diversity-aware selection 演算法 |
| **8 personas (vs SimTube 30)** | Neeko (EMNLP 2024) | per-character LoRA 場景 3-10 個是合理範圍 |
| **expected_comment_count（prompt-level）** | **本研究新增** | 提示模型 persona 活躍度區間，避免顯式 None 標籤；**僅在 prompt 中使用，不作為 RLAIF reward 訊號** |

### 2.7 Reviewer 預期質疑與回應

**Q: 「為什麼是這 8 個 personas？」**
A: 不是我們選的，是 PersonaChat top-8 by cosine similarity + MMR 程式抽樣。給定資料集 + random seed，任何人都能完全重現。

**Q: 「PersonaChat 是 2018 年資料，會不會過時？」**
A: SimTube (IUI 2025) 仍採用此資料集並達 SOTA。Persona descriptions 中的人格特質（外向 / 內向 / 興趣）與年代無關。

**Q: 「8 個夠涵蓋多元受眾嗎？」**
A: MMR 過濾保證 8 個 persona 在嵌入空間最大化多樣性。若 reviewer 要求，可在 ablation 加跑 top-16 對比。


---

## 3. Phase 1：蒸餾（Knowledge Distillation）

### 3.1 目標

讓 Llama-3.2-3B 繼承 **Claude Sonnet 4.5**（snapshot `claude-sonnet-4-5-20250929`）的「**事後反思型 sparse JSON 評論生成能力**」。

> **Teacher 版本說明**：原 v4.1 草稿寫「Claude-3.5 Sonnet」，但 Anthropic 已於 2025 Q4 將 3.5 Sonnet 標記 deprecated；2026-05-05 實際呼叫 API 時 `claude-3-5-sonnet-latest` 已不可用。改採 **Claude Sonnet 4.5**（snapshot 版避免 auto-upgrade，可重現性高）。價格與 3.5 Sonnet 相同（$3 / 1M input + $15 / 1M output），預算 ~$24.6 USD 不變。

### 3.2 影片資料準備

```
═══════════════════════════════════════════════════════
Step 1.1a: 訓練集影片（100 部）
═══════════════════════════════════════════════════════
   來源：YouTube Data API v3
   數量：100 部短 YouTube 影片 (1–3min, avg ~2min)
   用途：Stage A → Timeline Script → Claude 蒸餾 → SimLens 訓練

   類型分佈：5 categories × 20 部
     - Vlog/Lifestyle / Tech Review / Food/Cooking / Education/How-to / Entertainment/Comedy

   篩選條件：
     - 英文 + 有 captions（videoCaption=closedCaption）
     - 公開且 view > 10K，無 18+ 標籤
     - 長度嚴格 60–180s（videoDuration=short + ISO 8601 二次過濾）
     - 排除 #shorts tag 與 vertical 9:16 影片

═══════════════════════════════════════════════════════
Step 1.1b: 測試集影片（30 部，video-level hold-out）
═══════════════════════════════════════════════════════
   設計目的：消除 video-level data leakage（同一部影片不能同時當訓練資料 + 評估資料）

   數量：30 部短 YouTube 影片，5 categories × 6 部
   篩選條件：與 Step 1.1a 相同
   額外條件：exclude_manifest = 訓練集 manifest（避免重複）
   用途：
     - Anchor A supplementary per-video distributional check
     - Anchor B-1 同源時序錨（YouTube timestamp mentions）
     - Group 2 / 3 內容指標的最終評估
     - Group 4 25 人 human eval 影片從 30 test 中抽 8 部

   採集成本：YouTube API quota 內 ~5 min；Stage A pipeline ~2 hr

═══════════════════════════════════════════════════════
Step 1.2: UMaT-inspired 時序對齊 → 全局 Timeline Script
═══════════════════════════════════════════════════════
   對每部影片：
   
   (a) Whisper-Large-v3 整段轉錄（含時間戳）
       輸出：[(0.5s, "Hi everyone"), (2.1s, "today..."), ...]
   
   (b) LLaVA-NeXT 段描述（每 10 秒一段，共 N = ⌈duration/10s⌉ 段）
       對每段：抽 4 frames（在段內 t=0%, 33%, 66%, 100%）
              拼接成 panel image
              連同段內 transcript 餵給 LLaVA-NeXT
              生成段視覺描述（~150 字）
   
   (c) 時序對齊 → 全局 Timeline Script（v4.1 關鍵改變）
       直接產生整片可一次餵入的 Timeline Script：
       
       === Timeline Script ===
       [00:00-00:10] Visual: <LLaVA 段描述>
                     Audio: <該段 Whisper 文字>
       [00:10-00:20] Visual: ...
                     Audio: ...
       ...
       [02:50-03:00] Visual: ...
                     Audio: ...
       === End ===
       
   產出：100 個影片 × 1 個 Timeline Script per video

═══════════════════════════════════════════════════════
Step 1.3: Claude 蒸餾資料生成（核心 v4.1 改變）
═══════════════════════════════════════════════════════
   
   對每個 (影片 V, persona P)：
     一次餵入 V 的 Timeline Script + persona P
     讓 Claude 輸出 sparse JSON list
     
   Prompt template:
   ┌────────────────────────────────────────────────────────────┐
   │ You are a YouTube viewer with this persona:                 │
   │ {persona_yaml}                                              │
   │                                                             │
   │ You just finished watching a short video. Below is the      │
   │ complete timeline of what happened:                         │
   │                                                             │
   │ {timeline_script}                                           │
   │                                                             │
   │ Reflect on the entire video. List the moments where you     │
   │ would have left a comment, staying in character with the    │
   │ persona description above. Match your persona's expected    │
   │ comment count: {persona.expected_comment_count_range}       │
   │ (e.g., "low" personas may comment 0-2 times, "high" 3-6).   │
   │                                                             │
   │ Output ONLY a valid JSON array in this exact format:        │
   │ [                                                           │
   │   {"timestamp": "MM:SS", "comment": "your comment here"},   │
   │   ...                                                       │
   │ ]                                                           │
   │                                                             │
   │ If nothing struck you, output an empty array: []            │
   │                                                             │
   │ Important:                                                  │
   │ - Choose timestamps that fall within actual segment ranges  │
   │ - Match your persona's expected comment count               │
   │ - Each comment should be 10-50 words                        │
   │ - This is post-hoc reflection (you finished watching)       │
   └────────────────────────────────────────────────────────────┘
   
   產出量：
     100 影片 × 8 persona = 800 sparse JSON lists
     平均每 list 包含 0-7 條評論（依 persona 活躍度）
     預估總評論數：~3,200 條（avg ~4 條 per (video, persona) pair）
     
   為什麼採用此設計：
     (1) "無反應" 自然編碼為「不出現該 timestamp」，無需 None 標籤
     (2) Persona 的活躍度直接由 list 長度體現，更貼近真實行為
     (3) Text-level timestamp 輸出已被 Chapter-Llama (CVPR 2025) 在 1 hr 影片驗證可行
     (4) 成本可控：800 calls × $0.015 ≈ $12 USD
     (5) Post-hoc 定位無 future leakage 破口
   
   成本估算：
     Timeline Script avg ~2200 tokens (in) + persona ~200 tokens (in)
     Output avg ~500 tokens
     Per call: ~$0.015
     Total: 800 calls × $0.015 = $12 USD

═══════════════════════════════════════════════════════
Step 1.4: Real-World Anchor 資料採集（v4.2.3）
═══════════════════════════════════════════════════════

   設計目的：打破 Claude 蒸餾 → Qwen reward → Qwen evaluation 的封閉迴路。
   採集 2 個獨立外部訊號，於 §5.2.5 Group 1+ 評估使用。

   v4.2.3 設計（2 個獨立訊號）：
     A. 文本錨：HF 187K YouTube comments → 3 個 deterministic 分布指標
     B. 時序錨：30 高-mention test 部 YouTube 觀眾 timestamp mentions
                 → per-video Hotspot Recall / Precision @ ±5s

   ─── Anchor A：YouTube 留言生態（文本錨，分布級對齊）───

   主來源：HuggingFace `breadlicker45/youtube-comments-180k`
     - 187,482 條真實英文 YouTube 留言（單欄 text，HF Hub 直接下載）
     - 過濾 length < 10 / > 1000 outliers 後 → 176,562 條可用
     - 樣本量大、無 video-level leakage、$0 採集成本

   Supplementary：30 test 部對應的 top comments
     - YouTube Data API v3 commentThreads.list, order=time
     - 過濾 outliers + random sample 30 條/部
     - 用於 per-video distributional check 補強

   ─── Anchor B：YouTube Timestamp Mentions（per-video Hotspot Recall）⭐ 強 claim ───

   學術背書：
     - SoccerNet (CVPR 2018) [Ref 37]：tolerance window paradigm
       — 候選 timestamp 落在 ground truth ±θ 秒內算 true positive
     - DanmakuTPPBench (NeurIPS 2025) [Ref 38]：觀眾標記時點 = TPP events 範式

   設計演進（v4.2 → v4.2.3 的學習過程）：
     v4.2 原版：Bilibili 跨平台 per-video → 跨平台配對失敗（22/49 全錯）
     v4.2.1：YT mentions per-video → random test set mention 太稀疏（47% 0 條）
     v4.2.2：YT + Bilibili 雙 distribution-level → 兩 baseline 方向相反矛盾
     v4.2.3：保留 per-video Hotspot Recall 強 claim + 高 mention 篩選 test set
            破解原因：mention 稀疏是「test set 內容類型」問題，撈 mention-rich
            content type（compilation / highlight / reaction breakdown / mix tracklist）
            即可解決

   採集對象：30 部 high-mention 1-3 分鐘 YouTube 影片
     篩選 query：highlights compilation / best moments / top 10 / reaction
                 breakdown / sports highlights / music mix tracklist /
                 tutorial chapters
     pre-filter：對每部 pre-fetch 200 條 top-level comments，
                 regex 抓 "MM:SS" / "@MM:SS" mentions，
                 排除 clock-time false positives，
                 留下 mentions ≥ 5 的影片進 test set
     完全 hold out：與 100 train 部不重疊（exclude_manifest 排除）

   採集邏輯：
     階段 1 — Test set 篩選：
       collect_high_mention_videos.py 跑 mention-heavy queries，
       每部 pre-fetch 200 留言計 mention 數，留 ≥ 5 mentions 的，撈到 30 部
     階段 2 — Stage A pipeline：
       下載 mp4 → Whisper transcribe → LLaVA 段視覺 caption → Timeline Script
     階段 3 — 完整 mention 採集（per-video ground truth）：
       對每部 test 影片再撈所有 top-level comments（max 500），
       regex 抓所有 mentions，落在影片 duration 內，排重 → ground truth list

   產出格式：
     data/anchor_b/timestamp_mentions/{yt_id}.json
       {"yt_id": ..., "duration_sec": ..., "n_mentions": N,
        "mentions": [{"comment_id":..., "timestamp_sec":..., "comment_text":...}]}

   評估流程：
     對每個模型（Claude / GPT-4o-mini / Llama / SimLens）：
       對 30 test × 8 personas 跑 SimLens pipeline 預測 sparse JSON
       抽出 predicted timestamps（8 personas 聯集，per video）
     對每部影片計算：
       (a) KDE peak detection（bandwidth=3, min_separation=5s）→
           top-K hotspot（K = ⌈duration / 30s⌉）
       (b) Hotspot Recall @ ±5s = 對齊 hotspot 預測 / |hotspots|
       (c) Hotspot Precision @ ±5s = 對齊 hotspot 預測 / |predicted_union|
       (d) 30 部影片取平均 → main result table 數字

   核心 claim（強）：
     「在 30 部從未進訓練的 high-mention test 影片上，SimLens 預測的 timestamp
       對齊真實 YouTube 觀眾在留言中主動標記的 timestamp，per-video Recall=X /
       Precision=Y within ±5s tolerance，超越 Claude / GPT-4o-mini / Llama 三個
       baseline。」

   合規：純 YouTube 公開資料；無 IRB / 法律風險
   成本：$0（YouTube API quota 內）

   ─── Anchor B-2：Bilibili Popular 短片彈幕（跨平台時序錨, distribution-level）───

   學術背書：
     - DanmakuTPPBench (Jiang et al., NeurIPS 2025, arXiv 2505.18411) [Ref 38]：
       將彈幕視為 timestamp-rich 多模態 temporal point process events 的 NeurIPS 級 dataset，
       為「彈幕 timestamp = 觀眾在時間軸上自發標記反應時點」提供標準學術範式
     - Plot-Aligned Danmaku & Viewer Engagement (ScienceDirect 2025) [Ref 39]：
       實驗顯示劇情對齊彈幕顯著影響 viewer engagement，
       間接證明彈幕分布集中處 = 真實 reaction hotspots

   採集對象：50 部 Bilibili popular 1-3 分鐘短片，**完全獨立、不對應 YouTube test set**

   採集邏輯：
     (a) 撈 popular 候選：
         GET https://api.bilibili.com/x/web-interface/popular?ps=50&pn=N
         多頁取直到 50 部 duration 60-180s 的影片，不限類型 / 不限語言
     (b) 對每部 BVID 取 cid，再抓彈幕 XML：
         GET https://comment.bilibili.com/{cid}.xml
         解析每條 <d p="time,..."> 的 time 欄位
         **僅儲存秒級 timestamp，不儲存彈幕文本**（隱私 / 法律合規）
     (c) 儲存格式：
         {"bvid": "...", "cid": ..., "duration_sec": ...,
          "n_danmaku": N, "timestamps": [12.3, 18.7, ...]}

   產出量：50 部 × 平均 ~800 彈幕 = ~40,000 個秒級 timestamps

   設計重點：
     - 不要求對應 YouTube 影片（規避跨平台配對失敗問題）
     - distribution-level KS（同 B-1 指標，比整體分布性質）
     - 跟 B-1 互補：B-1 同源同 hold-out / B-2 跨平台獨立樣本
     - 兩錨同時對齊 → SimLens captures universal human reaction-time pattern

   合規：純秒級 timestamp，不儲存彈幕文本內容；Bilibili 公開 API 免費；
        學術用途、不發佈原始資料
   成本：$0

   ─── 採集時程 ───

   Week 1：100 train 部 + 30 test 部 Stage A pipeline + Anchor A/B-1 採集
   Week 2 Day 1（並行）：Bilibili 50 部 popular + 彈幕 timestamp 採集（~30 min）
   總額外工作量：~3 小時，可平行於 Stage A 跑時做
```

### 3.3 SFT 訓練設定（蒸餾階段）

```python
# 訓練 hyperparameters
config = {
    "base_model": "meta-llama/Llama-3.2-3B-Instruct",
    "quantization": "4-bit GPTQ",
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lora_dropout": 0.05,
    
    "training": {
        "epochs": 3,                  # 樣本數 800 較少，需多 epoch 收斂
        "batch_size": 2,              # 序列變長（整片 timeline）
        "gradient_accumulation": 8,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "max_seq_length": 4096        # 整片 Timeline Script 加 persona 約需 3000+ tokens
    },
    
    "data": {
        "samples_per_persona": 100,   # 100 部影片 × 1 list per persona
        "total_samples": 800,
        "split": {"train": 0.85, "val": 0.10, "test": 0.05}
    },
    
    "output_format": "constrained_json",  # 用 Outlines/XGrammar 強制 JSON 合規
    "json_schema": {
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
}
```

### 3.4 Per-Persona LoRA Adapter 訓練策略

```python
# 8 個獨立 LoRA Adapter
for persona_id in [P1, P2, ..., P8]:
    # 該 persona 的 100 個 (video, sparse_json_list) 樣本
    persona_data = data[persona_id]
    
    # 訓練該 persona 的 LoRA
    lora_p = train_lora(
        base_model=Llama-3.2-3B,
        train_data=persona_data,
        lora_rank=8,
        epochs=3,
        constrained_decoding=True   # 確保輸出合法 JSON
    )
    save_adapter(lora_p, f"./adapters/{persona_id}")

# 推理時
def generate_sparse_comments(video, persona_id):
    """對整片影片生成 persona_id 的稀疏評論列表"""
    base = load(Llama-3.2-3B)
    adapter = load(f"./adapters/{persona_id}")
    model = merge(base, adapter)
    
    prompt = format_sparse_prompt(
        timeline_script=video.timeline_script,
        persona=PERSONAS[persona_id]
    )
    
    output = model.generate(prompt, constrained_json_decoding=True)
    
    # 直接 parse 為 List[{timestamp, comment}]
    sparse_list = json.loads(output)
    return sparse_list  # 可能是 [] 也可能是 [{...}, {...}]
```

### 3.5 Phase 1 學術依據

| 設計選擇 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| Knowledge distillation 為起點 | DistilBERT (Sanh et al., 2019)、Tülu 3 (Lambert et al., 2024) | SFT on synthetic data |
| Claude as teacher | OpenCharacter (arXiv 2501.15427) | 大模型蒸餾 role-playing 行為 |
| Synthetic persona data | PersonaLLM (Jiang et al., 2024) | 合成 persona dialog 訓練 |
| **10 秒等長分段策略** | **UMaT (Bi & Xu, arXiv 2503.09081)** | fixed-length structured segmentation 細節依據 |
| **All-modality-as-text 哲學** | **Socratic Models (Zeng et al., ICLR 2023, arXiv 2204.00598)** | 將視覺/音訊統一降維為 language-based world-state history，是 SimLens Stage A Timeline Script 的祖師爺工作 |
| **One-shot timeline → text-output timestamp** | **Chapter-Llama (Ventura et al., CVPR 2025, arXiv 2504.00072)** | Llama-3.1-8B + LoRA rank=8，輸入「ASR [HH:MM:SS] / Caption [HH:MM:SS]」按時間排序的純文字 timeline，輸出 timestamp + 內容；機制與 SimLens 99% 一致 |
| **Sparse temporal prediction（補充背書）** | **MMDuet (arXiv 2411.17991)、MM-When2Speak (arXiv 2505.14654)** | 「VideoLLM Knows When to Speak」事件驅動稀疏輸出概念背書 |
| LoRA per persona | **Neeko (EMNLP 2024, arXiv 2402.13717)** | per-character LoRA 已被證明優於 single LoRA + prompt |
| 4-bit GPTQ + LoRA rank 8 | LLaMA-Factory 官方文件、Thakkar et al. (ACL 2024) | 標準 PEFT 工作流 |
| **LoRA SFT 後續可接 DPO** | **Thakkar et al. (ACL 2024, arXiv 2406.04879)** | 300+ 實驗證明 LoRA-SFT → LoRA-DPO 範式可行 |
| **兩階段 (SFT + DPO) on LoRA** | **Multi-MLLM Distillation (Gu et al., arXiv 2505.22517)** | 直接前例 |
| **Constrained JSON decoding** | **JSONSchemaBench (arXiv 2501.10868)** | 確保 sparse list 輸出格式合規 |
| **「沒反應」隱式編碼為空陣列** | 本研究新增（受 Chapter-Llama 啟發）| 避免顯式 None 標籤需求；Chapter-Llama 也用此方式：無 chapter boundary 時 LLM 不輸出對應 timestamp |

---

## 4. Phase 2：RLAIF（Reinforcement Learning from AI Feedback）

### 4.1 目標

讓 Phase 1 蒸餾後的 Llama-3.2-3B 在**時序定位精度 + 內容品質**兩個維度上接近或超越 Claude（teacher）。

> 學術背書：<br>
> *"In some settings, e.g., harmless dialogue generation, RLAIF even surpasses RLHF due to more consistent label definition."* (RLAIF survey, Lee et al., 2023)

### 4.2 完整 DPO 訓練流程

```
═══════════════════════════════════════════════════════
Step 2.1: 候選 Sparse List 生成
═══════════════════════════════════════════════════════
   對每個 (影片, persona):
     用 Phase 1 蒸餾後的 Llama-3B + LoRA_P 生成 N=4 個候選 sparse JSON list
     - temperature=0.9, top_p=0.95
   
   注意：候選的「沒反應」=空陣列，自然出現，不需特殊處理
   
   產出：800 (video, persona) 對 × 4 candidates = 3,200 candidate lists

═══════════════════════════════════════════════════════
Step 2.2: 2-aspect Multi-Reward 評分（v4.2 簡化）
═══════════════════════════════════════════════════════
   對每個候選 sparse list 評分（注意：是對整個 list 評分，不是單條評論）
   
   R_total = 0.50 × R_timing            (時機合理性，對應 timestamp 集合)
           + 0.50 × R_content_quality   (每條評論的 persona-aligned 內容品質)
   
   ⚠ 注意：因為輸出單位是「sparse list」（不是單條評論），
          所有 reward 設計為 list-level（對整條 list 評分）。
   
   ⭐ v4.2 設計原則：每個 reward 必須對應一個「外部可驗證」或
                    「與 Phase 1 蒸餾不重複」的訊號維度：
     - R_timing：對應「事件層級顯著性」，可由 §5.2.5 Crowd Hotspot 外錨間接驗證
     - R_content_quality：對應「persona-aligned 內容品質」，與 Phase 1 蒸餾的
                          「整體格式 / 語氣」訊號正交
   
   list 長度由 prompt 中的 expected_comment_count 區間軟引導，不再做為 reward 訊號
   （expected_comment_count 由 Claude 估計，若用作 reward 會形成「Claude 估的活躍度
   訓練學自 Claude 的 Student」循環，因此降為 prompt-level constraint）。
   
   timestamp 分布的疏密由 R_timing 隱式調節（事件本就稀疏 → 預測自然不擠堆），
   不再額外設置「均勻分布」reward（與 sparse event-driven motivation 矛盾）。

═══════════════════════════════════════════════════════
Step 2.3: Preference Pair 構造
═══════════════════════════════════════════════════════
   對每個 (video, persona)：
     取 4 候選中 R 最高 → chosen
     取 4 候選中 R 最低 → rejected
   
   產出：800 個 (video, persona, chosen, rejected) preference pairs

═══════════════════════════════════════════════════════
Step 2.4: DPO Update
═══════════════════════════════════════════════════════
   用 trl library 的 DPOTrainer
   每個 LoRA adapter 獨立訓練

═══════════════════════════════════════════════════════
Step 2.5: 迭代 DPO 輪數（可調超參）
═══════════════════════════════════════════════════════
   重複 Step 2.1-2.4 共 K 輪，K 為實驗超參（不固定 2 輪）
   
   ⭐ v4.2 標記：K 為 ablation 超參
       - 主實驗預設 K = 1（最低成本完整跑通 pipeline）
       - Ablation A9：跑 K ∈ {1, 2, 3} 三組對比，於 §5.3 與 Appendix C 報告差異曲線
       - 啟動條件：若 Group 1+ Real-World Anchor 在 K = 1 已飽和（Hotspot Recall
                  與 Length KS 不再隨 K 提升），則維持 K = 1，K = 2/3 視為負面結果報告
       - 報告：Appendix C 列出 K = 1 / 2 / 3 在 Group 1+ 與 Group 2 的差異曲線
   
   依據與保留：Bootstrapping with Implicit Rewards (ICLR 2025)
        但該文獻用於更大模型 + 更多 preference pairs，
        SimLens 在 3B + 800 pairs 規模下，多輪是否帶來顯著額外訊號需經實驗驗證。
   
   重要區別：SimLens 不採用 Self-Rewarding LM 的 self-judge 機制
     - actor (Llama-3.2-3B) 與 judge (Qwen3-32B-Q4) 為不同模型
     - 「強者評弱者」設計，避開 3B 自評的 self-bias 與能力不足
```

### 4.3 2 個 Reward 完整定義

#### Reward A: Timing（時機合理性）— 50%

```python
def reward_timing(sparse_list, timeline_script):
    """
    對 list 中每個 timestamp 判斷：該時點是否真的有「值得反應的事件」
    
    來源：SoccerNet (CVPR 2018) action spotting saliency 概念
    """
    if not sparse_list:
        # 空陣列：判斷「整片是否真的沒高潮」
        peak_count = count_high_engagement_segments(timeline_script)
        return 1.0 if peak_count == 0 else max(0, 0.5 - 0.1 * peak_count)
    
    scores = []
    for entry in sparse_list:
        ts = parse_timestamp(entry["timestamp"])
        local_segment = get_segment_at(timeline_script, ts, window=5)
        
        prompt = f"""
You are evaluating whether a YouTube viewer would naturally comment at this moment.

Local context (±5s around {entry['timestamp']}):
{local_segment}

On a 1-5 scale, rate whether this moment contains a noteworthy event 
(highlight, twist, climax, surprise, emotional peak):
1: completely uneventful
2: minor event
3: moderate event
4: clear highlight
5: strong climax/twist

Output ONLY the integer.
"""
        score = ollama_call("qwen3:32b-q4_K_M", prompt)
        scores.append(int(score) / 5.0)
    
    return sum(scores) / len(scores)
```

#### Reward B: Content Quality（每條評論的 persona-aligned 內容品質）— 50%

```python
def reward_content_quality(sparse_list, persona_yaml, timeline_script):
    """
    對 list 中每條評論評分（內容品質 + persona 一致性 + 局部相關性）
    
    來源：PersonaGym (EMNLP 2025)、Score Before You Speak (2025)
    """
    if not sparse_list:
        return 0.5  # 空陣列不適用，給中性分（list 是否該空由 R_timing 隱式判斷）
    
    scores = []
    for entry in sparse_list:
        ts = parse_timestamp(entry["timestamp"])
        local_context = get_segment_at(timeline_script, ts, window=5)
        
        prompt = f"""
You are evaluating a YouTube comment on three dimensions.

Persona description:
{persona_yaml}

Local video context (±5s around {entry['timestamp']}):
{local_context}

Generated comment:
"{entry['comment']}"

Rate 1-5 on each dimension, then output the AVERAGE as a single integer:

A. Persona Consistency (PersonaGym rubric):
   1: contradicts persona; 5: strongly reflects persona
B. Linguistic Habits:
   1: tone/word/emoji mismatch; 5: perfectly matches persona style
C. Local Relevance:
   1: comment refers to events not in this segment;
   5: directly responds to events in this 5s window

Output ONLY the average integer (1-5).
"""
        score = ollama_call("qwen3:32b-q4_K_M", prompt)
        scores.append(int(score) / 5.0)
    
    return sum(scores) / len(scores)
```

### 4.4 LLM-as-Judge 設定

```python
def llm_judge(sparse_list, persona, aspect):
    """
    使用單一 Qwen3-32B-Q4 作為 frozen judge。

    依據：RLAIF (Lee et al., 2023) d-RLAIF — frozen off-the-shelf LLM
         直接給 reward 比訓練 reward model 更穩，避開 RM staleness。
    """
    return call_judge("qwen3:32b-q4_K_M", sparse_list, persona, aspect)
```

### 4.5 DPO 訓練設定

```python
dpo_config = {
    "loaded_from_phase_1": True,
    "ref_model_frozen": "phase_1_checkpoint",
    
    "training": {
        "epochs": 1,  # DPO 容易 overfit
        "batch_size": 1,    # 序列長
        "gradient_accumulation": 16,
        "learning_rate": 5e-7,
        "beta": 0.1,
        "max_length": 4096,
        "max_prompt_length": 3500
    },
    
    "preference_data": {
        "total_pairs": 800,
        "iterative_rounds": 1   # v4.2 預設 K=1；K ∈ {1,2,3} 為 ablation A9 超參
    }
}
```

### 4.6 Phase 2 學術依據

> **方法定位**：SimLens Phase 2 = **RLAIF + DPO**（feedback 來源是 AI，訓練演算法是 DPO）。

| 設計選擇 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| RLAIF 用於 alignment | RLAIF (Lee et al., 2023, Google DeepMind) | AI feedback 達 RLHF 同等效果 |
| **不訓練 reward model（用 frozen Qwen-32B）** | **RLAIF (Lee et al., 2023) d-RLAIF** | frozen LLM 直接給 reward 比訓練 RM 更穩 |
| **DPO 取代 REINFORCE / PPO** | DPO (Rafailov et al., NeurIPS 2023) | 對 LoRA 微調更友善、24GB 單卡跑得動 |
| **DPO on LoRA adapter** | **Thakkar et al. (ACL 2024)、Multi-MLLM Distillation (arXiv 2505.22517)** | 「LoRA SFT → 同 LoRA DPO」直接前例 |
| Multi-aspect reward | MORLAIF (Williams, arXiv 2406.07496) | 多目標 reward 比單一更穩 |
| **List-level reward (2 aspects, v4.2 簡化)** | **本研究新增** | 配合 sparse 輸出格式，對整條 list 評分而非單條評論；v4.2 從 4-aspect 簡化為 2-aspect（保留 R_timing + R_content_quality） |
| **R_timing (event saliency)** | **SoccerNet (CVPR 2018) action spotting paradigm** | 借鏡「tolerance window 內算 true positive」的時點顯著性概念 |
| 本地 LLM-as-Judge | "Replacing the Judge" (SambaNova, 2024) | Llama-3.1 70B ≈ GPT-4 Turbo |
| **AI 訊號驅動的 preference data** | **Multi-MLLM Distillation (Gu et al., 2025/05)** | teacher 不一致即作為 preference signal |
| **Iterative DPO（K 為超參，v4.2 預設 K=1）** | Bootstrapping with Implicit Rewards (ICLR 2025) | 多輪迭代提升 alignment；v4.2 將 K ∈ {1,2,3} 列為 ablation A9 |

---

## 5. 評估方案（Evaluation Protocol）

### 5.1 Benchmark 對標模型

| Baseline | 角色 | 為何選它 |
|---------|------|---------|
| **SimTube** (Claude+GPT-4, whole-video, 1 comment) | 直接競爭 SOTA | 同類最相關工作 |
| **Claude Sonnet 4.5 zero-shot** (one-shot sparse JSON) | Teacher 本身 | 證明 student 能否超越 |
| **GPT-4o-mini zero-shot** (one-shot sparse JSON) | 獨立大模型對照 baseline | 預算實際選擇 — Claude 之外的第二個 LLM 家族，避免結果是 Claude 特殊偏好；mini 而非 4o 是預算決策（v4.1 全研究 OpenAI 預算規劃為 GPT-4o-mini） |
| **Llama-3.2-3B zero-shot** (one-shot sparse JSON) | 未訓練起點 | 證明訓練有效性 |
| **Llama-3.2-3B + Phase 1 only (SFT)** | 蒸餾 ablation | 證明 RLAIF 必要 |
| **Llama-3.2-3B + Phase 2 only (DPO from zero-shot)** | RLAIF ablation | 證明蒸餾必要 |
| **Llama-3.2-3B + SimLens (full)** | **本研究方法** | 完整 SimLens |

### 5.2 評估指標（四層架構）

#### Group 0: Format Compliance（基礎前提）

```python
# 因為輸出是結構化 JSON 含時序約束，必須先驗證合規率
# 設計依據：
#   - IFEval (Zhou et al., 2023) 的 verifiable instruction 哲學
#   - JSONSchemaBench (Geng et al., 2025) 的 compliance rate 評估範式
# 設計原則：所有指標皆為 deterministic（程式可驗證），不需 LLM judge

format_metrics = {
    "SCR (Schema Compliance Rate)":
        # 通用 JSON 合規維度（對標 JSONSchemaBench compliance rate）
        # 兼含：能被 json.loads() 解析 + 符合 {timestamp: 'MM:SS', comment: str} schema
        "輸出能被解析、且符合預期 JSON schema 的比例（預期 >= 95%）",
    
    "TVR (Timestamp Validity Rate)":
        # SimLens 任務特化維度（對應 sparse temporal prediction 的時序約束）
        # 對通過 SCR 的輸出再驗證 timestamp 合理性：
        #   (a) 'MM:SS' 格式正確
        #   (b) 落在影片實際長度內
        #   (c) 不重複且按時序排列
        "timestamps 同時滿足格式 + 範圍內 + 不重複 + 排序的比例"
}

# Composite FCR = min(SCR, TVR)（兩者都通過才算合規）

# 預期效益：
# - Llama-3B zero-shot: ~30-40% FCR
# - SimLens Phase 1 (SFT): ~95%+ FCR
# - 證明蒸餾賦予 3B 模型嚴格的格式控制能力
```

**指標設計說明**：SCR 與 TVR 兩個指標分別對應「通用 JSON 合規」與「SimLens 任務特化合規」兩個正交維度。這個切分讓我們在 Table 2 能診斷出失敗模式：若 SCR 高但 TVR 低，代表「格式對但 timestamp 不合理」（例如 Claude 給出 03:45 但影片只有 02:30），這是 sparse temporal prediction 任務獨有的失敗模式。

#### Group 2: Tier 2 — Content Quality（內容指標）

```python
content_metrics = {
    # 對每條評論獨立評估
    "Local Context Relevance (BERTScore)":
        "評論 vs 該 timestamp ±5s 局部劇本的 BERTScore",
    "Persona Consistency (PersonaGym rubric, Qwen judge)":
        "評論是否符合 persona 背景與信念",
    "Linguistic Habits (PersonaGym rubric, Qwen judge)":
        "tone / 用詞 / emoji 是否符合 persona",
    "Coherence (Score Before You Speak rubric)":
        "評論本身是否通順、合理",
    "Engagingness (UniEval)":
        "評論的吸引力與互動潛力",
    "Distinct-1 / Distinct-2":
        "lexical diversity"
}
```

**評估粒度與聚合方式**（每個維度算法不同，Table 1b 每個 cell 對應計算層級）：

| 維度 | 評估單位 | 該 cell（如 P1 / Persona Cons.）的計算 |
|------|---------|----------------------------------------|
| Persona Consistency | per **list** | mean over 100 影片：PersonaGym(P1_desc, P1's full sparse list for video_i) |
| Linguistic Habits | per **persona-corpus** | concat P1 在 100 影片的所有 comments → Qwen judge 對照 PersonaChat P1 dialogue ground truth |
| Local Relevance (BERTScore) | per **comment** | 雙層平均：mean over P1 在該影片的所有 comments → mean over 100 影片 |
| Coherence | per **list** | mean over 100 影片：coherence_score(P1's full sparse list for video_i) |
| Engagingness (UniEval) | per **comment** | 雙層平均：mean over P1's all comments per video → mean over 100 影片 |
| Distinct-1 / Distinct-2 | per **persona-corpus** | concat P1 在 100 影片的所有 comments → 計算整 corpus n-gram diversity |

**Std-Dev 行（Table 1b 底部）的算法**：
跨 8 個 persona 的標準差，量「8 個 LoRA 訓練是否均勻」 — 不是每個 persona 內部跨影片的 std。
```python
std_dev_persona_cons = std([P1.cell, P2.cell, ..., P8.cell])  # n=8
```
若 std-dev > 0.10 → 顯示某個 persona 拖垮整體（例：P5 very low activity 訓練資料太少）。

#### Group 3: Tier 3 — List-Level（列表層級指標）⭐ v4.1 新增

```python
# 設計原則：
#   1. 避免 reward-evaluation contamination
#      （v4.2 reward 已簡化為 R_timing + R_content_quality 兩個維度，
#        此處保留歷史脈絡：原 v4.1 還有 R_frequency_match / R_coverage_diversity，
#        評估端因此排除對應維度，避免封閉論證；v4.2 砍掉這兩個 reward 後此風險自動消失）
#   2. 量「persona 內容差異」而非「persona timestamp 差異」
#      ✗ Persona Differentiation (Jaccard timestamps) — 邏輯有破口
#         共同爆點時 8 personas 自然會在同 timestamp 反應，
#         Jaccard timestamps 不能反映「反應內容」的 persona-specific 差異
#      ✓ 改用 Persona Content Distinctiveness (Embedding-based)

list_metrics = {
    "Persona Content Distinctiveness (Embedding-based)":
        "對每個 persona，把該 persona 對 100 部影片的所有 comments concat 起來，"
        "用 OpenAI text-embedding-3-small 嵌入成一個 persona-level embedding。"
        "8 個 persona 兩兩算 cosine distance，取 28 對的平均（0-1）。"
        "獨立於 reward — 量 8 個 LoRA 學到的「persona-specific 內容風格」差異。"
        "高分 → 8 個 persona 真的寫出風格不同的評論；"
        "低分 → 8 個 LoRA 只是換口頭禪，內容本質相同。",

    # ⚠ v4.2 變更：Teacher Alignment Score (TAS) 降為 Appendix 診斷指標
    # 不再列為主結果指標，避免「越像 Claude 越高分」的封閉論證問題。
    # 詳見 §5.2.5 Group 1+ 與 §5.2.7。
}
```

**為什麼不用 Jaccard timestamps？**
共同爆點時 8 personas 都在同 timestamp 反應是合理的（例：影片 0:42 主持人摔手機，
所有 persona 都會留言，只是 P1 心疼、P2 講螢幕、P5 說教、P8 玩梗）。
Jaccard timestamps 會把這個合理行為錯判為「LoRA 失敗」。
正確做法是量「**反應內容**」的差異，不是「**反應時間**」的差異。

#### Group 1+: Real-World Anchoring（外部錨點，打破封閉迴路）⭐ v4.2 新增（P0）

```python
# 設計目的：
#   v4.1 整個訓練 + 評估 pipeline 中，訊號鏈為：
#     Claude（蒸餾 teacher） → Qwen3-32B（reward） → Qwen3-32B（evaluation judge）
#   除了 Group 4 的 25 人 human evaluation 外，所有訊號都來自 LLM。
#   這形成「Student 越像 Claude/Qwen 共識 → 分數越高」的封閉論證風險。
#
# Group 1+ 透過 2 個獨立的「真實世界」訊號打破此迴路（v4.2.3）：
#   A. 文本錨：HuggingFace 187K YouTube comments + 30 test 部對應留言
#   B. 時序錨：30 高-mention test 部 YouTube 觀眾 timestamp mentions
#              → per-video Hotspot Recall / Precision @ ±5s
#
# 核心設計原則：
#   1. 全 deterministic 指標，完全不用 LLM judge
#   2. Anchor A 量「群體層級寫作風格分布對齊」（影片無關）
#   3. Anchor B 量「per-video 真實觀眾標記時點 vs 模型預測時點對齊度」
#      強 claim：模型對影片 X 預測的 timestamp 對齊影片 X 真實觀眾
#                自發在留言中標的 timestamp（±5s tolerance）
#
# v4.2.3 修正：砍掉 v4.2.2 的 distribution-level Anchor B-1/B-2
#   - B-1 (YT random test) mention 太稀疏（47% 影片 0 條）
#   - B-2 (Bilibili) baseline 與 B-1 方向相反，universal pattern 假設不成立
#   - 改用「高 mention 篩選 test set + per-video Hotspot Recall」回到強 claim

real_world_anchors = {
    # ─── Anchor A: Distributional Match（影片無關文本錨, 3 個指標）───
    # 主來源：HF breadlicker45/youtube-comments-180k (n=176,562 after filter)
    # 補強：100 train + 30 test 部對應 ~3,250 條真實留言（per-video sup.）
    # 比較對象：SimLens 8 LoRA 對 30 test 部所有生成留言 concat 為 corpus

    "Length KS-statistic (↓)":
        "scipy.stats.ks_2samp 比較 SimLens corpus 留言長度分布 vs HF 187K corpus。"
        "Kolmogorov-Smirnov 兩樣本檢驗（statistical-textbook standard）。"
        "預期 (target)：Llama zero-shot ~0.42 → Phase 1 SFT ~0.20 → Full ~0.13。",

    "Sentiment Wasserstein (↓)":
        "用 VADER (Hutto & Gilbert, ICWSM 2014) 對兩個 corpus 各算 neg/neu/pos 比例，"
        "計算 1-Wasserstein distance（discrete CDF L1 distance）。"
        "捕捉「SimLens 是否過於正面/禮貌」（distillation bias 常見症狀）。"
        "預期 (target)：Llama zero-shot ~0.38 → Phase 1 SFT ~0.15 → Full ~0.09。",

    "Embedding Frechet Distance (↓)  [Ref 42]":
        "Heusel et al. (NeurIPS 2017) FID 範式跨模態變體。"
        "對 SimLens corpus 與 HF 187K corpus 各 sample 5,000 條，"
        "用 OpenAI text-embedding-3-small 嵌入後計算 Frechet distance："
        "FD = ||μ_S − μ_R||² + Tr(Σ_S + Σ_R − 2(Σ_S Σ_R)^½)"
        "（圖像領域 FID 的文本變體；Frechet BERT/CLIP/Audio Distance 已是文獻常見作法）。"
        "整體層級分布錨，不依賴個別樣本對應。"
        "預期 (target)：Llama zero-shot ~0.52 → Phase 1 SFT ~0.21 → Full ~0.13。",

    # ─── Anchor B: Per-Video Hotspot Recall on High-Mention Test Set ⭐ v4.2.3 強 claim ───
    # 資料來源：30 部 high-mention test YouTube 影片
    #          (pre-fetch 200 留言篩 ≥5 mentions，預期每部 5-15 條 mentions)
    # 比較對象：每個模型對影片 X 預測的 timestamps 集合 vs 影片 X 真實 mentions
    # 學術背書：[Ref 37] SoccerNet (CVPR 2018) tolerance window 範式
    #          [Ref 38] DanmakuTPPBench (NeurIPS 2025) TPP events 分析

    "Hotspot Recall @ ±5s (↑)  [Ref 37, 38]":
        "對每部 test 影片，從真實 mentions 用 KDE peak detection（bandwidth=3s, "
        "min_separation=5s）抽 top-K hotspot（K = ⌈duration / 30s⌉）。"
        "計算各模型 8 personas 預測 timestamps 聯集對 hotspots 的覆蓋率（±5s tolerance）："
        "Recall = |hotspots ∩ predicted_union| / |hotspots|"
        "對 30 部影片取平均 → main result number。"
        "預期 (target)：Llama zero-shot ~0.20 → Phase 1 SFT ~0.55 → Full ~0.68。",

    "Hotspot Precision @ ±5s (↑)  [Ref 37]":
        "Precision = |predicted_union ∩ hotspots_dilated| / |predicted_union|"
        "（hotspots_dilated = 每個 hotspot 擴張 ±5s 的 union）。"
        "防止「亂撒 timestamp 換 recall」的 reward hacking 變體。"
        "預期 (target)：Llama zero-shot ~0.25 → Phase 1 SFT ~0.50 → Full ~0.62。",

    # ─── Anchor C: Persona Plausibility（弱錨點, supplementary）───
    # 用 8 個訓練後的 LoRA 對 HF 187K 隨機抽 3000 條真實留言計算 perplexity

    "Real Comment LoRA Coverage (Avg PPL ↓ + Coverage % ↑)":
        "對每條真實留言，計算它在 8 個 LoRA 下的最低 perplexity。"
        "Avg PPL = mean(min_PPL across 3000 comments)"
        "Coverage % = |min_PPL < 100| / 3000，表示真實留言能被 SimLens persona 集合"
        "以合理 perplexity 解釋的比例。"
        "高 Coverage + 低 Avg PPL → 8 個 LoRA 集合 cover 真實留言多樣性。"
        "弱錨點，僅作 supplementary（perplexity 比較跨模型不嚴謹）。"
}
```

**Group 1+ 為什麼能打破封閉迴路？**

| 訊號鏈 | v4.1 | v4.2.2（加入 Group 1+）|
|---|---|---|
| 訓練 teacher | Claude Sonnet 4.5 | Claude Sonnet 4.5（不變）|
| 訓練 reward | Qwen3-32B-Q4 | Qwen3-32B-Q4（不變）|
| Group 0 / 2 / 3 評估 | 基於 LLM judge / Teacher 對齊 | 基於 LLM judge / Teacher 對齊 |
| **Group 1+ 評估** | **無** | **HF 187K YT comments + 30 test 部 YT mentions + 36 部 Bilibili popular 彈幕（三個獨立外部訊號源）** |
| Group 4 評估 | 25 人 Upwork（外部）| 25 人 Upwork（不變）|
| **獨立外部訊號數量** | **1（人類）** | **4（HF 大池 + YT 同源 mentions + Bilibili 跨平台彈幕 + 25 人 Upwork）** |
| **Video-level data leakage** | 有（同 100 部訓練+評估）| **無**（v4.2.1: 100 train + 30 hold-out test）|

**設計檢驗清單**：

- ✅ 不依賴 Claude（資料源來自 YT + Bilibili real users）
- ✅ 不依賴 Qwen（指標全 deterministic：KS / Wasserstein / Frechet）
- ✅ 6 個指標全部有學術背書：KS = scipy/Kolmogorov 1933, Wasserstein = Kantorovich 1942 + VADER (ICWSM 2014), Frechet = [Ref 42] Heusel NeurIPS 2017, B-1/B-2 KS = [Ref 37] SoccerNet CVPR 2018 + [Ref 38] DanmakuTPPBench NeurIPS 2025 + [Ref 39] Plot-Aligned Danmaku 2025
- ✅ Video-level hold-out（30 test 部完全不進訓練）
- ✅ B-1 + B-2 兩錨提供結構互補的時序 prior（sparse delayed vs dense real-time）
- ✅ 與既有 Group 0/2/3 互補：Group 0/2/3 量「內部相對表現」，Group 1+ 量「外部世界對齊度」

#### Group 4: Human Evaluation（25 人 Crowd Study, 照 SimTube IUI 2025 規格）

> **設計原則**：完全對齊 SimTube IUI 2025 Section 6.1 protocol（招募平台 / 樣本數 /
> 品管流程 / 統計方法），確保 Table 5 數字可與 SimTube paper 直接比對。在此基礎上
> **新增 Timing Naturalness 維度**對應 SimLens 核心貢獻（sparse temporal localization），
> 該維度 SimTube 因無 timestamp 輸出而標 N/A。

```python
human_eval = {
    # ─── 招募（照 SimTube Section 6.1）───
    "participants":      25,
    "platform":          "Upwork（與 SimTube IUI 2025 一致，非 Prolific）",
    "recruitment":       "隨機 crowd workers，不做 demographic-matching",
    "recruitment_rationale": (
        "與 SimTube / PersonaGym / PersonaLLM 學界範式一致 — persona simulation "
        "領域慣例量第三方 believability 而非 self-reported alignment；且 PersonaChat "
        "persona 過於 niche，Upwork screener 無法精確匹配。"
    ),
    "compensation":      "$15 USD/人 × 25 ≈ $375 + 5% buffer ≈ $400",

    # ─── 任務分配 ───
    "video_count":               8,
    "video_categories":          ["vlog", "教學", "遊戲", "生活", "旅遊",
                                  "動畫", "音樂", "美食"],   # YouTube 主流 genre
    "videos_per_participant":    8,        # 每人評全 8 部，與 SimTube 一致
    "comments_per_video":        30,       # 與 SimTube Section 6.1 完全一致
    "total_ratings":             "25 人 × 8 影片 × 30 條 × 4 維度 = 24,000 個 Likert",

    # ─── 每部影片 30 條評論池組成（雙盲、隨機順序）───
    "comment_pool_per_video": {
        "SimLens_Full":           8,    # 8 personas × 1 條代表評論
        "SimLens_SFT_only":       8,    # ablation：證明 Phase 2 RLAIF 有效
        "Claude_zero_shot":       8,    # Teacher baseline
        "SimTube_baseline":       1,    # whole-video baseline (無 timestamp)
        "Llama_3B_zero_shot":     5,    # floor baseline (5 personas 隨機抽)
        # 合計：30 條
    },

    # ─── 評分維度（照 SimTube 3 維 + Timing 1 維）───
    "rating_dimensions": [
        "Relevance        — 與影片該時間點內容相關度",        # 照 SimTube
        "Believability    — 像真實 YouTube 用戶會留的評論",   # 照 SimTube
        "Helpfulness      — 對創作者了解觀眾反應有幫助",     # 照 SimTube
        "Timing Naturalness — 在這個 timestamp 留此評論的時機合理",  # SimLens 新增
        # ※ SimTube baseline 因無 timestamp，Timing 維度標 N/A
    ],
    "rating_scale": "7-point Likert (1=完全不同意, 7=完全同意)",

    # ─── 額外：Forced-choice A/B 對打（5 題）───
    "forced_choice_questions": {
        "count":         5,
        "format":        "對於 P_n（給定 persona 描述），下面哪一條更像他會留的？",
        "options":       "(A) SimLens 出 vs (B) SimTube/Claude 出，順序隨機、來源不告知",
        "purpose":       "提供 head-to-head 勝率，比絕對 Likert 更具說服力",
        "analysis":      "Binomial test (SimLens 勝率 vs 50%)",
    },

    # ─── 5 階段 form 流程（照 SimTube）───
    "form_stages": [
        "1. Watch Video       — 強制觀看完整影片",
        "2. Video Quiz        — 4-5 題選擇題，必須 ≥80% 正確才能繼續",
        "3. Write Summary     — ≥50 字影片摘要（作為 reference 與二次過濾）",
        "4. Rate Comments     — 30 條評論 × 4 維度 + 5 題 forced-choice",
        "5. Form Feedback     — 開放回饋（流程是否清楚）",
    ],
    "quality_control": {
        "must_watch_video":      True,
        "must_pass_video_quiz":  "≥80% accuracy",
        "must_write_summary":    "≥50 字",
        "attention_checks":      "Form 中嵌入 2 題 attention check（如「請選 4」）",
    },

    # ─── 統計分析（照 SimTube Section 6.1）───
    "statistical_analysis": {
        "primary_test":      "Wilcoxon Signed-Rank Test（paired comparison）",
        "multiple_comparison": "Bonferroni Correction",
        "significance_level": "α = 0.05",
        "comparisons": [
            "SimLens Full vs SimTube              → 證明 SimLens 整體更優",
            "SimLens Full vs SimLens SFT-only     → 證明 Phase 2 RLAIF 有效",
            "SimLens Full vs Claude zero-shot     → 證明蒸餾達 Teacher 水準",
            "SimLens Full vs Llama-3B zero-shot   → 證明訓練流程有效",
        ],
        "forced_choice_test": "Binomial test (one-sided, p_null=0.5)",
    },

    # ─── 報告呈現 ───
    "reporting": {
        "main_table":      "Table 5（5 systems × 4 dimensions, mean ± SD + p-values）",
        "qualitative":     "Appendix A：3-5 個 case study figure（同部影片不同系統輸出對比）",
        "ablation_subset": "Appendix B：per-persona Likert breakdown（8 personas × 4 維度）",
    },

    # ─── 預算與時程 ───
    "estimated_cost":  "$400 USD（25 × $15 + 5% buffer）",
    "estimated_time":  "Week 7 完成招募 + 收集 + 統計分析（7 天）",
    "ethics":          "通過所屬機構 IRB review；參與者匿名、可隨時退出",
}
```

#### 5.2.7 Diagnostic Metrics（診斷指標，不放主結果表）⭐ v4.2 新增

```
診斷指標 = 用來判斷「訓練流程是否災難性失敗」，不是用來「展示 SimLens 多強」。
這類指標放 Appendix，不進主結果表，避免封閉論證。

─── Diagnostic D1: Teacher Alignment Score (TAS) ───
（v4.1 原為 Table 3 主指標，v4.2 降級為診斷指標）

  定義：Student 預測的 timestamps vs Teacher (Claude) 預測的 timestamps 的對齊度
        用 ±5s tolerance window 做 greedy matching，計算 temporal F1 @ ±5s
        借 SoccerNet (CVPR 2018) action spotting 的 tolerance window 範式

  為什麼降級：
    TAS 越高 = Student 越像 Teacher。在「Claude 蒸餾 Student」設定下，
    把「越像 Claude」當主指標等於封閉論證 —— Student 學會 Claude 偏誤
    與 Student 真正能力提升在 TAS 上無法分離。

  v4.2 用法：
    - 報告位置：Appendix B，僅 Phase 1 / Full 兩個 row
    - 健康範圍：0.5 < TAS < 0.85
        TAS < 0.5 → 蒸餾失敗（Student 沒學到 Teacher 的時序判斷）
        TAS > 0.85 → 過度模仿（Student 是 Claude 的「複印機」，沒有 RLAIF 校正效果）
    - 不發出「TAS 高 = SimLens 強」的 claim，Phase 1 本身已是合格的能力證明工具

─── Diagnostic D2: Multi-Judge Spot-Check（Qwen vs GPT-4o-mini 一致性）───
⭐ v4.2 新增

  動機：
    v4.1 整個 Phase 2 reward + Group 2 / 3 評估全部用單一 Qwen3-32B-Q4 作為 judge
    L4 已承認 Qwen vs GPT-4 一致性約 85-90%，但未實測。
    若 Qwen 評分本身有系統性偏誤，Phase 2 訓練與 Group 2 評估會同步受污染。

  做法：
    1. 從 800 個 (video, persona) 對中隨機抽 80 個（10% sample）
    2. 對 2 個 reward（R_timing / R_content_quality）：
       同時用 Qwen3-32B-Q4 + GPT-4o-mini 各跑一次評分
    3. 計算每個 reward 的：
       - Pearson correlation r（連續分數一致性）
       - Cohen's κ（離散排序一致性，binning 為 5 級）
    4. 報告綜合一致性指標
    5. 對 Group 2 內容指標（Persona Cons / Linguistic / Coherence / Engagingness）
       同時抽 80 條 comments 做相同 spot-check

  判讀準則：
    - Cohen's κ ≥ 0.6（substantial agreement）→ 主結果視為穩定
        論文正文：可發出「SimLens > Teacher」的強 claim
    - 0.4 ≤ Cohen's κ < 0.6（moderate agreement）→ 主結果註明 judge sensitivity
        論文正文：弱化為「SimLens 接近 Teacher（judge dependent）」
    - Cohen's κ < 0.4（fair / poor agreement）→ 必須加入 GPT-4o-mini 全量重跑
        Phase 2 訓練若已用 Qwen reward 完成，補做 sensitivity analysis（GPT-4o-mini judge 下的相對排序是否一致）

  成本：
    Reward spot-check：80 cases × 2 rewards × $0.005 ≈ $0.8
    Group 2 spot-check：80 comments × 4 dimensions × $0.005 ≈ $1.6
    總計：$2.4 USD（v4.2 新增預算項）

  論文價值：
    堵住 reviewer 「你的所有評分都來自 Qwen，如果 Qwen 是錯的呢？」的質疑。
    把 LLM-as-Judge 的 sensitivity 從「未驗證假設」變成「實測 + 透明報告」。

─── Diagnostic D3: Pre-flight Teacher Sanity Check ───
（沿用 v4.1 §11 L12 緩解策略，補完正式描述）

  Week 3 Day 1 跑：
    - 5 部影片 × 8 personas = 40 個 sparse list
    - 人工檢查 ~50 個 timestamp 對應關係
    - 通過標準：≥ 80% timestamp 與其聲稱的事件對齊

  失敗 fallback：
    通過率 60-80% → 啟動二級緩解（重寫 prompt + 加 3 個 few-shot）
    通過率 < 60% → fallback 到 GPT-4o-mini 當 teacher
    通過率 < 60% 且 GPT-4o-mini 也失敗 → 升級到 GPT-4o（+$25 預算）

  將通過率寫進論文 Section 4.1（Experimental Setup 透明度）。
```

### 5.3 Ablation Study 設計

```
Ablation 的目的：證明 SimLens 每個關鍵設計決策都是必要的
（不是所有可能的變體都跑——只跑「移除某設計後性能下降」的對照）

必跑 ablations（v4.2 共 7 組，較 v4.1 簡化）：

═══ 訓練流程 ablation ═══
A1. SimLens (full)                              ← 完整方法（baseline）
A2. - w/o Phase 2 (RLAIF) → SFT only            ← 證明 RLAIF 必要
A3. - w/o Phase 1 (Distillation)                ← 證明蒸餾必要
    → DPO from Llama-3B zero-shot
A4. - w/o Multi-LoRA                            ← 證明多 LoRA 必要
    → single LoRA + persona injected via prompt

═══ Reward 設計 ablation（v4.2 簡化）═══
A5. - w/o Multi-aspect (only R_content_quality) ← 證明 2 reward 比 1 reward 好
A6. - w/o R_timing                              ← 證明時序顯著性 reward 必要
    （v4.1 的 A7 / A8 已隨 R_frequency_match / R_coverage_diversity 砍除而移除）

═══ 超參敏感度 ablation（v4.2 新增）═══
A9. Iterative DPO rounds K ∈ {1, 2, 3}          ← K 為實驗超參，非固定值
    判斷：若 K=1 在 Group 1+ 已飽和 → 維持 K=1 為主結果，K=2/3 列為負面結果

註：每個 ablation 獨立訓練、獨立評估，每組產生「全 6 指標 × 8 persona + Group 1+ 外錨」的數字
    為避免結果表過度膨脹，§5.4 主結果表只列關鍵 baseline，完整 ablation 數字
    放 appendix（或 supplementary materials）
```

### 5.4 預期結果表（論文 main results）

> **重要 Framing：Table 1 不是 SOTA 排行榜，是「能力對齊驗證」**
>
> 影片評論生成（**video commentary generation**）這個任務目前**沒有公認的學術 SOTA**：
> - SimTube (Hung et al., IUI 2025) 是領域內**唯一**直接前身，發表至今仍是 niche
> - GPT-4o-mini / Claude Sonnet 4.5 **不是專門評論影片的模型**，只是強通用 LLM
> - Llama-3.2-3B zero-shot 完全不是 SOTA，只是「未訓練起點」
>
> 因此 Table 1 的設計目的是 **3 件事**，而非 leaderboard 競賽：
> 1. **師徒對齊驗證**：Student (3B) 在 5 維度上是否接近 / 超越 Teacher (Claude)
> 2. **跨大模型家族對照**：GPT-4o-mini 作為**獨立 LLM 家族 baseline**，避免結果是 Claude 特殊偏好
>    （選 mini 而非 4o 是預算決策；reviewer 若關心 4o vs mini 落差，可在 rebuttal 補跑 4o subset，~$5）
> 3. **vs SimTube 直接競爭**：唯一同類學術前身的比較
>
> **Per-persona breakdown 設計說明**：
> 因為 SimLens 訓練後是 8 個獨立 LoRA adapter（per persona），每次推理都要指定
> 一個 persona、掛上對應 adapter。所以**所有評估指標都必須 per-persona 計算**，
> macro-average（8 persona 平均）只是輔助數字。
>
> Table 1a 報告 macro-avg（給 reviewer 一眼看到整體性能）
> Table 1b 報告 per-persona breakdown（證明 8 個 LoRA 都各自學會了）
> Table 1c 報告分階段（Phase 1 / Phase 2 / Full）的 macro-avg 對比

#### Table 1a: 主結果 Macro-Avg（能力對齊驗證 + 跨模型對照）

```
Method                          | Role         | Persona | Linguistic | Local Rel. | Coherence | Engaging
                                |              | Cons.   | Habits     | (BERTScore)|           | (UniEval)
──────────────────────────────────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot          | 未訓練起點   | 0.42    | 0.38       | 0.45       | 0.50      | 0.45
SimTube (Hung 2024)             | 唯一同類前身 | 0.78    | 0.72       | N/A        | 0.79      | 0.76
                                | (whole-video)|         |            |            |           |
Claude Sonnet 4.5               | Teacher      | 0.76    | 0.70       | 0.66       | 0.79      | 0.73
GPT-4o-mini                     | 獨立家族對照 | 0.71    | 0.65       | 0.62       | 0.76      | 0.70
──────────────────────────────────────────────────────────────────────────────────────────────────
SimLens Full (SFT + DPO) ⭐     | 本研究 (3B)  | 0.83    | 0.81       | 0.66       | 0.78      | 0.77
                                |              | > Tea. | > Tea.     | ≈ Tea.    | ≈ Tea.   | ≈ Tea.
```

> **GPT-4o-mini 預期表現說明**：mini 的 persona / linguistic 能力略低於 Claude Sonnet 4.5（0.71 vs 0.76、0.65 vs 0.70），這是預算決策的接受 trade-off。論文 framing 重點不是「mini 比 Claude 弱」（顯而易見），而是「**SimLens 3B 在 persona-specific 維度上同時超越 Claude 與 GPT-4o-mini 兩個獨立 LLM 家族**」。
>
> **Sonnet 4.5 vs 3.5 預期值微調**：4.5 在 persona-following、instruction-adherence 上比 3.5 略強（OpenAI/Anthropic 公開 benchmark 大約 +0.02），所以原 v4.1 草稿的 (0.74, 0.68, 0.65, 0.78, 0.72) 上調至 (0.76, 0.70, 0.66, 0.79, 0.73)。SimLens Full 預期值維持 (0.83, 0.81, 0.66, 0.78, 0.77) — 「3B student 超越 600B teacher」的論文主張仍成立（0.83 > 0.76、0.81 > 0.70）。

**重點論述**：
- **領域指標 (Persona / Linguistic)**：SimLens (3B) > Claude (Teacher)、> GPT-4o-mini，RLAIF DPO 在 persona-specific 維度上有效超越兩個獨立 LLM 家族的 baseline
- **通用指標 (Coherence / Engaging)**：SimLens ≈ Claude，略 > GPT-4o-mini，蒸餾保留通用能力
- **vs SimTube**：SimTube 只能給整片 1 條評論，SimLens 在 persona 與 linguistic 上勝出

#### Table 1b: Per-Persona Breakdown（8 LoRA 的個別表現）

> 8 個 personas P1–P8 由 PersonaChat top-8 程式取樣產生（§2.3）。
> Activity（高/中/低）由 expected_comment_count_range 自動標記。
> 數字為預期值，實際 personas 與 activity 標籤在 Week 1 取樣腳本跑完後填入。

```
Persona     | Activity | Persona | Linguistic | Local Rel.| Coherence | Engaging
                       |  Cons.  |            |           |           |
─────────────────────────────────────────────────────────────────────────────────
P1 (sampled)| high     | 0.85    | 0.87       | 0.68      | 0.79      | 0.81
P2 (sampled)| low      | 0.82    | 0.79       | 0.71      | 0.83      | 0.74
P3 (sampled)| medium   | 0.84    | 0.82       | 0.66      | 0.78      | 0.78
P4 (sampled)| medium   | 0.83    | 0.80       | 0.65      | 0.77      | 0.75
P5 (sampled)| very low | 0.78    | 0.75       | 0.69      | 0.81      | 0.71
P6 (sampled)| high     | 0.86    | 0.85       | 0.64      | 0.74      | 0.80
P7 (sampled)| med-low  | 0.81    | 0.78       | 0.67      | 0.79      | 0.74
P8 (sampled)| very high| 0.85    | 0.86       | 0.62      | 0.75      | 0.82
─────────────────────────────────────────────────────────────────────────────────
Macro-Avg              | 0.83    | 0.81       | 0.66      | 0.78      | 0.77
Std-Dev                | 0.03    | 0.04       | 0.03      | 0.03      | 0.04
```

**重點論述**：
- **8 個 persona std-dev 都 < 0.05** → 證明 8 LoRA 訓練均勻，沒有特定 persona 失敗
- **高活躍 personas 各項指標較高** → list 較長提供更多訓練訊號
- **低活躍 personas 的 Local Relevance 反而較高** → 留言精選度高、內容更貼合
- **這是 SimTube 完全做不到的維度** → SimTube 只能給整片 1 條評論，無法 per-persona 對比

#### Table 1c: 訓練階段對比（Macro-Avg）

```
Stage                          | Persona | Linguistic | Local Rel.
─────────────────────────────────────────────────────────────────
SimLens Phase 1 only (SFT)     | 0.71    | 0.66       | 0.60
SimLens Phase 2 only (DPO)     | 0.73    | 0.69       | 0.58
                               | (DPO from Llama-3B zero-shot)
SimLens Full (SFT + DPO) ⭐    | 0.83    | 0.81       | 0.66
─────────────────────────────────────────────────────────────────
Improvement: SFT → Full        | +0.12   | +0.15      | +0.06
```

**重點論述**：兩階段訓練（SFT 提供基礎能力 + DPO 提供 alignment）顯著優於單階段，驗證 §3.5 / §4.6 引用的兩階段範式（Thakkar ACL 2024、Multi-MLLM Distillation 2025）。

#### Table 2: Format Compliance Rate

```
Method                              | SCR    | TVR    | Composite FCR
─────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot              | 35%    | 28%    | 28%
Claude zero-shot                    | 92%    | 86%    | 86%
SimLens Phase 1 (SFT)               | 97%    | 93%    | 93%
SimLens Full + Outlines (推理時)¹   | 99%    | 97%    | 97%
```

¹ **Outlines 是什麼？**
  Outlines (Willard et al., 2023, arXiv 2307.09702) 是一個 **constrained decoding** 框架，
  在 LLM 生成階段**逐 token 過濾**：每生成一個 token 前，先檢查「這個 token 會不會違反
  預設的 JSON schema？」若會違反就強制換下一個合法 token，**確保輸出 100% 符合 JSON 結構**。

  替代品：XGrammar (NVIDIA/CMU)、Guidance (Microsoft)。SimLens 推理時掛 Outlines。

  注意：constrained decoding **只能保證結構合規 (SCR)**，**無法保證時序合理性 (TVR)**：
  - SCR 99%+：JSON 結構強制對（key 名、引號、括號）
  - TVR 仍只 97%：模型仍可能輸出超出影片長度的 timestamp（如 03:45 但影片只有 02:30）
                   或重複、亂序的 timestamp
  → 這正是 SimLens 訓練 LoRA 的價值：**時序合理性必須學會，不能靠後處理強制**

**讀法**：
- SCR 高 / TVR 低 = 格式對但 timestamp 不合理（範圍超出 / 重複 / 亂序）
- 兩者皆低 = 模型連 JSON 結構都產不出
- 完整 SimLens（SFT+DPO+Outlines）三者疊加才能在兩個維度同時達 97%+

#### Table 3: List-Level 指標（reward-independent）⭐ v4.2 簡化

```
Method                          | Persona Content
                                | Distinctiveness
─────────────────────────────────────────────────
Llama-3.2-3B zero-shot          | 0.22
Claude zero-shot (Teacher)      | 0.51
SimLens Phase 1 only (SFT)      | 0.55
SimLens Full (SFT + DPO) ⭐     | 0.68
```

**設計說明**：
- **Persona Content Distinctiveness**：量 8 個 LoRA 寫出的評論「內容風格」彼此差異，OpenAI text-embedding-3-small 對 concat comments 嵌入後兩兩 cosine distance 取 28 對平均。reward 只優化「單一 persona 內部品質」，不直接優化「跨 persona 內容差異」，因此此指標獨立於 reward。

> **v4.2 變更**：原 Table 3 包含 Teacher Alignment Score (TAS, F1@5s)，
> 但 TAS 直接量「Student 與 Teacher (Claude) 的 timestamp 重合度」，
> 與「越像 Claude 越高分」的封閉論證風險直接相關。
> v4.2 將 TAS 移至 §5.2.7 / Appendix B 作為**診斷指標**（不是主結果），
> 僅用來檢查「蒸餾過程是否災難性失敗」（合理範圍 0.4 < TAS < 0.9 即可）。
> 「Phase 1 蒸餾是否成功」改由 Table 6 的 Crowd Hotspot Recall 與 Group 1+ 外部錨點驗證。

→ 此指標的提升加上 Group 1+ 外部錨點才是 SimLens 真正的能力證明，非 reward optimization 直接帶來的副作用。

#### Table 6: Real-World Anchor 結果 ⭐ v4.2 新增（P0 主結果）

> **這是 v4.2 最重要的新表**：4 個 distributional 文本錨 + 2 個跨語言時序錨，
> **完全獨立於 Claude / Qwen**，提供「SimLens 是否真的對齊真實世界」的外部驗證。
> 表中所有指標皆為 deterministic 計算（無 LLM judge），可重現性極高。

```
                                │ Length │ Sentiment │ Style │ Embedding │ Hotspot │ Hotspot
Method                          │ KS ↓   │ W-dist ↓  │ χ² ↓  │ Frechet ↓ │ Recall↑ │ Prec.↑
                                │        │           │       │           │ @±5s    │ @±5s
─────────────────────────────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot          │ 0.42   │ 0.38      │ high  │ 0.52      │ 22%     │ 18%
Claude Sonnet 4.5 (Teacher)     │ 0.18   │ 0.13      │ low   │ 0.20      │ 58%     │ 51%
GPT-4o-mini                     │ 0.21   │ 0.16      │ low   │ 0.23      │ 54%     │ 47%
SimTube (whole-video, 1 cmt)    │ 0.31   │ 0.24      │ mid   │ 0.34      │  N/A *  │  N/A *
SimLens Phase 1 only (SFT)      │ 0.20   │ 0.15      │ low   │ 0.21      │ 51%     │ 45%
SimLens Full (SFT + DPO) ⭐     │ 0.13   │ 0.09      │ low   │ 0.13      │ 68%     │ 60%
─────────────────────────────────────────────────────────────────────────────────────────────
Real Reference (YouTube/Bili)   │ 0.00   │ 0.00      │ —     │ 0.00      │ 100%    │ 100%

* SimTube 輸出無 timestamp，不適用 hotspot 指標
```

**重點論述**（這是 v4.2 對應「打破封閉迴路」的核心證據）：

1. **文本層級分布錨（4 個指標）**：SimLens Full 在 Length / Sentiment / Style / Embedding 4 個獨立指標上**全部勝過 Claude Teacher 與 GPT-4o-mini**，且分布距離 < SimTube 一半。**這個結果不依賴任何 LLM judge**，完全打破「只是學會 Qwen 評分習慣」的質疑。

2. **跨語言時序熱區錨（2 個指標）**：SimLens Full 對真實 Bilibili 彈幕熱區的 Recall 達 68%、Precision 達 60%，**勝過 Claude（58/51%）與 GPT-4o-mini（54/47%）**。彈幕來自中文觀眾、SimLens 寫英文，**訊號跨語言**，不可能來自 Claude 蒸餾偏誤 —— 這是 sparse temporal prediction 主貢獻最強的外部證據。

3. **Phase 1 vs Full 的進步**：v4.1 Table 1c 用 LLM judge 算出的「+0.12 / +0.15」可能是 Qwen 偏好遷移；Table 6 的「KS 0.20→0.13 / Hotspot R 51%→68%」是 deterministic 與外部訊號驗證的進步，**reviewer 無法以「LLM judge 偏誤」反駁**。

4. **Group 1+ 與 Group 4 互補**：Table 6 提供「群體層級分布對齊」證據（n = 3,000 真實留言 + 40,000 彈幕 timestamp），Table 5 Human Eval 提供「個別 persona 可信度」證據（n = 25 人 × 24,000 Likert）。兩個獨立外部錨 + 25 人 human eval = 三個獨立外部訊號，**封閉迴路徹底瓦解**。

#### Table 4: 效率比較 — 訓練成本 vs 推理成本（vs SimTube）

> **關鍵價值主張**：SimTube 是 pay-per-use（每跑一次都要付 API），
> SimLens 是 train-once-then-free（前期付一次蒸餾錢，部署後 user 端永遠免費）。
> 這個成本結構差異對任何要規模化使用的創作者 / 研究者都是核心優勢。

```
                            │ 訓練成本（一次性開發者付）│ 推理成本（per video, user 付）│ Latency │ VRAM
                            │  Claude / OpenAI API      │  user 端跑一部影片要付的       │         │
─────────────────────────────────────────────────────────────────────────────────────────────────
SimTube (Hung 2024)         │  $0                       │  ~$0.225 (Claude Sonnet 4.5)   │  ~45s   │ N/A
                            │ (純 prompting，無訓練)    │  每部影片都要付一次 API 費     │         │ (need API)
                            │                           │  (50K input + 5K output token  │         │
                            │                           │   × Sonnet 定價)               │         │
─────────────────────────────────────────────────────────────────────────────────────────────────
SimLens (本研究) ⭐         │  $42-72                   │  $0 ⭐                         │  ~24s   │ 6.5GB
                            │  ($12 Claude 蒸餾 +       │  (本地 Llama-3B + 8 LoRA       │         │ (consumer
                            │   $30-60 雲端 GPU 訓練)   │   完全本地推理，無 API 費)     │         │  GPU)
                            │  一次性前置投資            │                                │         │
─────────────────────────────────────────────────────────────────────────────────────────────────

**Cost Crossover 分析（規模化成本回本點）：**
  SimLens 訓練成本 $42-72 ÷ SimTube per-video $0.225 ≈ **187–320 部影片**

  解讀：
  - 跑 < 200 部影片：SimTube 較划算（不用一次性投資）
  - 跑 200+ 部影片：SimLens 開始回本
  - 跑 1000 部影片：SimTube 累積 $225，SimLens 仍只 $42-72
  - 跑 10000 部影片：SimTube 累積 $2,250，SimLens 仍只 $42-72
  - **長期使用 / 大規模部署：SimLens 成本優勢不可逆**

**附加優勢（無法量化但同等重要）：**
  - **隱私**：SimLens 全本地推理，創作者影片不上傳第三方 API
  - **離線可用**：SimLens 不依賴外部 API 連線
  - **可微調**：SimLens 可針對特定創作者風格進一步 fine-tune（SimTube 無此能力）
  - **消費級硬體**：SimLens 在 24GB VRAM 跑得動（RTX 3090/4090 等級）
```


### 5.5 評估學術依據

| 評估元素 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| 自動指標（NLG）| SimTube (Hung et al., 2024) Section 6.2 | BERTScore + ROUGE |
| Persona 評估 | PersonaGym (EMNLP 2025) | Persona Consistency + Linguistic Habits |
| Engagingness | PersoBench (Huang et al., 2024) | UniEval-based engagingness |
| Coherence | Score Before You Speak (2025) | coherence dimension |
| 25 人 crowd study | SimTube Section 6.1 | quiz + summary + rating protocol |
| **Schema Compliance Rate (SCR)** | **JSONSchemaBench (arXiv 2501.10868)** | JSON schema compliance 評估範式 |
| **Timestamp Validity Rate (TVR)** | **本研究新增（受 IFEval verifiable instruction 啟發）** | SimLens 任務特化指標，deterministic 驗證 |
| **Teacher Alignment Score (TAS, F1@5s) — v4.2 降為診斷指標** | **SoccerNet (CVPR 2018) action spotting paradigm** | 借「±tolerance window 算 true positive」範式量 Student vs Teacher timestamp 對齊度。v4.2 移至 Appendix，僅作蒸餾失敗檢測（避免「越像 Claude 越好」的封閉論證）|
| **Persona Content Distinctiveness (Embedding)** | **本研究新增**（沿用 SimTube IUI 2025 的 OpenAI text-embedding-3-small）| 量 8 個 persona 寫的評論「內容風格」彼此差異，獨立於 reward |
| **⭐ v4.2 新增：Length KS / Sentiment Wasserstein** | scipy.stats.ks_2samp / scipy.stats.wasserstein_distance | 對齊真實 YouTube top comments 文本分布；deterministic 計算，不需 LLM judge |
| **⭐ v4.2 新增：Style Marker χ²** | 本研究新增 | 縮寫率 / emoji 率 / 大寫率 / 問號率 4 維度卡方統計量，捕捉 Claude 蒸餾偏誤 |
| **⭐ v4.2 新增：Embedding Frechet Distance** | Heusel et al., NeurIPS 2017（FID 範式）+ OpenAI text-embedding-3-small | 借 FID 範式（W2 Gaussian distance）量 SimLens corpus 與真實 corpus 整體分布距離 |
| **⭐ v4.2 新增：Crowd Hotspot Recall / Precision @ ±5s** | SoccerNet (CVPR 2018) tolerance window + KDE peak detection | 量 SimLens 預測 timestamps 對 Bilibili 彈幕真實熱區的覆蓋；跨語言可用，群體層級訊號 |
| **⭐ v4.2 新增：Multi-Judge Spot-Check (Cohen's κ)** | Cohen, 1960 / 多 judge 一致性研究 | 對 80 個 (video, persona) 對同時跑 Qwen3-32B-Q4 + GPT-4o-mini，Cohen's κ ≥ 0.6 才視 single-judge 結果可靠 |
| Ablation 設計 | DPO 原論文 (Rafailov et al., 2023) | 標準 ablation 順序 |

---

## 6. 報告生成（Stage C）

### 6.1 為什麼用同一個 Llama-3B（不另外訓練）

```
論述：
  「One model, multiple tasks」是 2024-2026 多任務微調主流（Tülu 3、Llama Stack）。
  
  在 SimLens v4.1 中：
   - LoRA adapter for sparse list generation（8 個）
   - 報告生成直接用 base Llama-3B（不掛 adapter）
   - 情感分類也直接用 base Llama-3B（不掛 adapter，無需訓練）
```

### 6.2 三步驟報告生成流程

```
Step 6.2.1：情感分類（後處理，零訓練成本）
  對 sparse list 中每條評論：
    輸入：該 persona 的評論文字 + 該 timestamp 局部 context
    輸出：positive / negative / neutral
  
Step 6.2.2：逐評論建議（comment-level）
  對每條評論：
    根據情感極性與內容，產出該則評論的具體優化建議
  
Step 6.2.3：整片建議（video-level）
  綜合分析整個 8-persona sparse list 集合：
    - 跨受眾比較（哪些時段哪些 persona 留言）
    - Persona 共鳴熱區（每 persona 集中發言的時段）
    - 沉默熱區（沒人留言的時段——可能是內容低谷）
    - 整體節奏與內容調整方向
```

### 6.3 為什麼情感分類不需要訓練（學術依據）

```
LLM 已被證明在 zero-shot 情境下具備優異的情感分類能力：

[24] Hartmann et al. (Customer Needs and Solutions, 2024)
    證明 LLM zero-shot 在情感分類精度上不僅能與傳統 fine-tuned 
    transfer learning 方法競爭，甚至在某些情境超越。

[25] Lin et al. (JMIR/PMC, 2024)
    GPT-4 zero-shot 達 92-94% accuracy（F1 90-93%），
    Llama 2 達 72-75%。
    YouTube 場景與 SimLens 高度吻合。

結論：
  SimLens 在 Stage C 直接用 Llama-3B base 模型對生成評論做
  情感分類，預期準確率介於上述兩篇論文的 Llama 區間（~75%+），
  完全不需新增訓練資料、reward 或 ablation。
```

### 6.4 報告生成 Prompts

#### Prompt 1：情感分類（每則評論獨立呼叫）

```python
sentiment_prompt = """Classify the sentiment of this YouTube comment.

Comment: "{comment}"
Local context (around {timestamp}): {local_context}
Persona: {persona_brief}

Output ONLY one word: positive, negative, or neutral.
"""
```

#### Prompt 2：逐評論建議（comment-level）

```python
per_comment_prompt = """You are a video coaching assistant.

Below is a comment from a specific audience persona watching a short YouTube video (1–3 min)
at timestamp {timestamp}.

Persona: {persona_brief}
Local segment content: {local_segment_text}
Comment: "{comment}"
Sentiment: {sentiment}

Based on the sentiment, provide ONE specific actionable suggestion:
- If positive: how to amplify this engagement point in editing
- If negative: what specific issue caused this and how to fix it
- If neutral: what's missing that could make it more engaging for this persona

Output: 1-2 sentences, actionable, specific to this persona.
"""
```

#### Prompt 3：整片建議（video-level）

```python
overall_prompt = """You are a video analytics assistant.

Per-persona sparse comment lists (8 personas):
{all_sparse_lists_with_sentiments}

Persona descriptions:
{persona_summaries}

Video timeline:
{timeline_script}

Generate a structured overall report:

1. **Cross-Audience Comment Map** (跨受眾留言圖)
   For each timestamp range (per 10s segment), list which personas commented
   and the dominant sentiment.

2. **Persona Resonance Hotspots** (Persona 共鳴熱區)
   For each persona, identify the segments where they commented most heavily.

3. **Silent Zones** (沉默熱區)
   Identify segments where 0 or 1 persona commented—these may be content low points.

4. **Strategic Improvement Suggestions** (整體策略建議)
   - Negative-sentiment segments: root cause + fix
   - Cross-persona positive segments: how to amplify
   - Audience targeting: which persona this video best fits

Format: Markdown with clear sections.
"""
```

### 6.5 報告生成評估

```
評估維度：

1. 情感分類準確性
   - 從生成的評論中隨機抽樣 200 個
   - 由人類重新標註情感極性
   - 計算與 Llama-3B base 分類的 Cohen's Kappa
   - 預期：κ ≥ 0.65（substantial agreement）

2. 逐評論建議的可操作性
   - 找 5 位 YouTube 創作者
   - 評分維度：具體度、可執行度、與評論的相關度
   - 7-point Likert scale
   - 預期平均：5.0+/7.0

3. 整片建議的綜合性
   - 同樣 5 位創作者
   - 盲測 SimLens 報告 vs Claude 直接生成的報告
   - 預期：與 Claude 報告達到 85%+ 滿意度
```

---

## 7. 實驗時程與里程碑（v4.1 重新規劃）

### 7.1 8 週時程表

```
Week 1: 環境建置 + Persona 取樣
  □ 確認 GPU 環境（最低門檻 RTX 3090 24GB；本研究實際用 RTX 5090 32GB）
  □ 安裝 LLaMA-Factory / TRL / Ollama / Outlines（constrained decoding）
  □ Pull Llama-3.2-3B、Qwen3-32B Q4、LLaVA-NeXT
  □ 下載 PersonaChat 8K 資料集（Hugging Face: bavard/personachat_truecased）
  □ 抓 5-10 部影片做 pipeline sanity check + 提取 keywords
  □ 跑 §2.3 取樣腳本：PersonaChat → top-8 personas（OpenAI text-embedding-3-small + MMR）
  □ 用 Claude zero-shot 為 8 個 personas 估 expected_comment_count_range
  □ 定義 sparse JSON schema（用於 constrained decoding）
  □ 寫好 2 個 reward 函數骨架（R_timing + R_content_quality）+ 5 群評估指標骨架（含 Group 1+ 外錨）
  ★ Milestone 1：環境就緒、pipeline 跑通、8 個 personas 抽取完成

Week 2: 大規模影片素材收集 + Timeline Script Pipeline
  □ 用 YouTube Data API 收集 100 部短 YouTube 影片（嚴格 60–180s）
    篩選條件：medium duration filter + ISO 8601 二次過濾 + 排除 #shorts
  □ 跑 Whisper-Large-v3（含時間戳）
  □ 跑 LLaVA-NeXT 段描述（每 10 秒一段）
  □ UMaT-inspired 時序對齊 → 全局 Timeline Script
  ★ Milestone 2：100 部 YouTube 就緒

Week 3: Phase 1 蒸餾資料生成 + Teacher 雙重驗證

  ─── Day 1: Pre-flight Sanity Check（必過才能繼續）⭐ ───
  □ 【Pre-flight】Teacher Timestamp Fidelity Sanity Check
     成本：~$0.6 USD + 30 分鐘人工檢查
     範圍：5 部 YouTube 影片 × 8 personas = 40 個 sparse list
     步驟：
       1. 跑 Claude 產 40 個 sparse list
       2. 隨機抽 ~50 個 (timestamp, comment) 對
       3. 人工檢查每對：開影片到該 timestamp，看 comment 是否
          與該秒事件語義相關（Yes / No）
     通過標準：
       ≥ 80% Yes  → Claude 能力 OK，繼續 Day 2 大規模生成
       60–80% Yes → 調 Claude prompt 加強 timestamp 約束，重跑 sanity check
       < 60% Yes  → 換 GPT-4o-mini 當 teacher（Limitation L12 fallback），重跑
                    註：理想 fallback 應是 GPT-4o，但 v4.1 預算決策選 mini；
                    若 mini 也 < 60%，啟動 L12 二級緩解（重寫 prompt + few-shot 範例）
     依據：Chapter-Llama (CVPR 2025) 證明 LoRA-tuned LLM 能輸出對應
          timestamp，但未證明 zero-shot LLM (Claude) 同樣可行 → 必須 SimLens
          自己驗證 teacher 在這個任務上的 timestamp fidelity

  ─── Day 2-7: 大規模蒸餾資料生成（Batch API + 1h Prompt Caching）───
  □ Claude Message Batches API：800 個 (影片, persona) 一次提交
     - 每部影片的 8 personas 共用同一 Timeline Script，標 cache_control ttl=1h
     - submit batch → poll status（多數 < 1h，SLA 24h）→ stream results jsonl
  □ 100 影片 × 8 persona = 800 sparse lists
  □ 預算花費：~$3.5 USD（batch 5 折 + cache hit 7/8 personas/影片）
     比 sync API ($12) 省 ~70%
  □ 【驗證 1】Teacher Format Compliance Rate
     計算 Claude 的 SCR / TVR（驗證輸出穩定性，預期 SCR ≥ 92%）
  □ 【驗證 2】Batch 過期 / 失敗的 request 重跑（用 sync 補完）
  ★ Milestone 3：Sanity check 通過 + 蒸餾資料完成 + Teacher FCR 驗證通過

Week 4: Phase 1 SFT 訓練 + Format Compliance 驗證
  □ 對 8 個 persona 各訓練 1 個 LoRA adapter（含 constrained JSON decoding）
  □ 計算 Phase 1 Student 的 FCR（預期從 zero-shot ~32% 躍升到 96%+）
  □ 跑 baseline benchmark：
    - Llama 3B zero-shot
    - Claude zero-shot
    - GPT-4o-mini zero-shot（v4.1 全研究 OpenAI baseline 統一用 mini，預算決策）
  □ 跑 Phase 1 only 結果（SimLens-SFT）
  ★ Milestone 4：Phase 1 完整結果 + FCR 驗證

Week 5: Phase 2 RLAIF (round 1)
  □ 設置 Qwen3-32B 本地 judge
  □ 對每個 (video, persona) 生 4 候選 sparse list → 2-aspect 評分（R_timing + R_content_quality）
  □ 構造 800 個 preference pairs
  □ 跑 DPO 訓練（每個 LoRA 獨立 update）
  ★ Milestone 5：RLAIF round 1 結果

Week 6: Phase 2 RLAIF (round 2) + 全套 ablation
  □ Iterative DPO round 2
  □ 跑完整 ablation（見 §5.3 共 8 組）
  ★ Milestone 6：完整自動評估結果

Week 7: 人類評估（Group 4）
  □ 在 Upwork/Prolific 招募 25 人
  □ 設計 Google Forms（含影片、quiz、評分）
  □ 收集 25 人 × 8 影片的 Likert 評分
  □ 評估維度：Timing Naturalness + Persona Believability + Helpfulness
  □ 統計分析（Wilcoxon + Bonferroni）
  ★ Milestone 7：人類評估結果

Week 8: 論文撰寫 + 投稿準備
  □ 寫 8 頁 ACM MM 短文 / 4 頁 UIST poster
  □ 製作 architecture diagram（Timeline → Sparse JSON）
  □ 錄製 3 分鐘 demo video
  □ GitHub repo 整理 + Hugging Face 上傳 8 個 LoRA adapters
  ★ Milestone 8：論文 + Demo 就緒
```

### 7.2 關鍵風險與緩解

| 風險 | 機率 | 緩解策略 |
|------|------|---------|
| GPU 記憶體不足（max_seq=4096）| 中 | 用 Q3 量化、降到 2048 並切前 2 分鐘 |
| Claude API 預算超支 | 低 | 預估 $12，預留 $30 buffer |
| RLAIF 訓練不穩定 | 中 | 縮小 DPO learning rate，用 conservative β |
| **JSON 格式輸出崩壞** | **中** | **強制用 Outlines/XGrammar constrained decoding** |
| Sparse list 全空（模型懶）| 中 | prompt-level expected_comment_count 區間軟引導 + R_timing 對「該有反應卻沒反應」給低分 |
| 人類評估招募失敗 | 中 | 改用學校系內招募（30 人也夠 LBW） |

---

## 8. 預期硬體與成本

### 8.1 硬體需求

```
最低配置（論文宣稱可重現門檻）：
  - 1× RTX 3090 24GB
  - 64GB RAM、1TB SSD
  - 預估訓練時間：2 週

實際使用配置（本研究）：
  - 1× RTX 5090 32GB（Ada Lovelace 後繼，5th-gen Tensor Cores）
  - 128GB RAM、2TB NVMe SSD
  - 預估訓練時間：~1 週
  - 額外優勢：Llama-3B + Qwen3-32B-Q4 同卡共駐
              （6.5 + 20 = 26.5GB / 32GB），Phase 2 RLAIF online judging
              不需在每 round 切換模型，整輪訓練 wall time 省 30-50%
              ※ 論文仍以「24GB 消費級 GPU 可重現」為部署 framing

理想配置（未來擴展）：
  - 2× RTX 5090 或 1× A100 40GB
  - 256GB RAM
  - 預估訓練時間：3-5 天
```

**Framing 政策**：論文 method/deployment 章節仍以 24GB consumer-grade GPU 為基準陳述
（這是 SimLens 對比 SimTube/Claude 的核心賣點 — 「可消費級部署」）。32GB 5090 只用於
本研究的訓練加速，不進入 paper claim；reproducibility 章節列出最低配置 24GB 並提供
4-bit GPTQ + LoRA 配置確保 24GB 可重現。

### 8.2 成本估算（v4.1 — GPT-4o-mini + Batch API + Prompt Caching 版）

```
雲端 GPU（如果沒有自有硬體）：
  Vast.ai RTX 4090：$0.4/hour
  訓練總時數：~80 hours
  → $30-60 USD
  （本研究實際用 RTX 5090 32GB，無此費用；列出供他人重現參考）

API 成本（v4.1 預算優化策略）：

  雙重優化機制：
    (A) Batch API：所有大規模、非即時的 Claude 呼叫使用 Message Batches API
        → 50% 折扣（Sonnet 4.5: $1.50/M input, $7.50/M output vs sync $3/$15）
        → 多數 batch 1 小時內完成，最多 24 小時 SLA
    (B) Prompt Caching（1-hour ttl）：Timeline Script 在 8 personas 間共用
        → 每部影片寫入 cache 1 次（1.25× = $3.75/M），讀取 7 次（0.1× = $0.30/M）
        → Timeline Script 部分成本省 ~75%
    (C) 兩者疊加套用於蒸餾 + Claude baseline 兩個 800-call 階段

  ── Anthropic（Claude Sonnet 4.5, snapshot claude-sonnet-4-5-20250929）──
  Pre-flight sanity check（Week 3 Day 1, 40 calls, sync API）       ：~$0.6
    （sync 因為要即時看結果決定是否 fallback；不適合 batch）
  Persona activity 估計（Week 1, 8 calls, sync API）                ：~$0.04
    （量太小，batch overhead 不划算）
  Phase 1 蒸餾（Week 3 Day 2-7, 800 calls, Batch + 1h cache）       ：~$3.5
    （sync $12 → batch 5 折 → cache hit 7/8 segments → 約省 70%）
  Claude zero-shot baseline（Week 4, 800 calls, Batch + 1h cache）  ：~$3.5
  Anthropic 小計                                                   ：~$7.6

  ── OpenAI（一律 GPT-4o-mini + text-embedding-3-small）──
  PersonaChat embedding（Week 1, 8K personas × ~50 tok/each）       ：$0.01
  影片 keyword 抽取（Week 1, 100 影片 × ~500 tok, sync）            ：$0.01
  GPT-4o-mini zero-shot baseline（Week 4, 800 calls, OpenAI Batch API 5 折）：~$2
  OpenAI 小計                                                      ：~$2.0

  ── 其他 ──
  YouTube Data API v3                                              ：$0（quota 內免費）
  HuggingFace Hub（gated model token）                             ：$0

  API 總計                                                         ：~$9.6 USD ⭐

人類評估（Week 7，獨立預算）：
  Upwork crowd-sourcing 25 人 × $15 + buffer                      ：$300-400 USD

總成本：
  最低（自有 GPU + 不做 Group 4 人類評估）：~$10 USD ⭐
  標準（自有 GPU + Upwork 25 人）           ：~$310-410 USD
  最高（雲端 GPU + Upwork 25 人）           ：~$340-470 USD

預算節省累積：
  原 GPT-4o + sync API 預算 ：~$80 USD
  改 GPT-4o-mini           ：-$56  →  $24
  改 Batch API（50% 折扣）  ：-$10  →  $14
  改 Prompt Caching         ：-$4   →  $10
  最終預算                  ：~$10 USD（省 87.5%）

  注意：節省的前提是 batch API 24h SLA 可接受。SimLens 蒸餾分階段執行
       （Week 3 Day 2-7 有 6 天 buffer），完全在 batch SLA 範圍內。

Reviewer 預期質疑與回應：
  Q: 「為什麼不用 GPT-4o / sync API 確保品質？」
  A: SimLens 是學生研究預算驅動，且：
     (1) Batch API 與 sync API 結果完全相同（只是處理時序差異）；
     (2) Sonnet 4.5 已超越 3.5，足以勝任 teacher；
     (3) GPT-4o vs mini 在 Persona / Linguistic 落差約 0.05，預期值已反映
         （Table 1a：mini 0.71 vs 4o 0.76 estimated）；
     (4) 若 reviewer 要求 4o subset 對比，rebuttal 補跑 100 樣本 ~$5。
```

---

## 9. 預期論文結構

### 9.1 8 頁 ACM MM 短文版本

```
1. Introduction (1 頁)
   - 痛點：YouTube Analytics 延遲、SimTube 等系統只給整片評論
   - SimLens 願景：lightweight + on-device + temporally-localized + persona-aware
   - 關鍵 framing：post-hoc commentary generation（不偽裝即時性）

2. Related Work (1 頁)
   - SimTube：whole-video persona simulation
   - Chapter-Llama (CVPR 2025) / Socratic Models (ICLR 2023)：text-level video → LLM timestamp output
   - UMaT：multimodal temporal alignment
   - PersonaGym：persona evaluation
   - DPO + RLAIF：訓練範式

3. Method (2 頁)
   - 3.1 Architecture: One-shot Timeline → Sparse JSON
   - 3.2 Phase 1: Distillation from Claude (含 constrained JSON decoding)
   - 3.3 Phase 2: 2-aspect Multi-Reward DPO（R_timing + R_content_quality）

4. Experiments (3 頁)
   - 4.1 Setup: 100 short YouTube videos (1–3min) × 8 personas (sparse lists)
   - 4.2 Format Compliance: Table 2
   - 4.3 Main results (Tier 2): Table 1
   - 4.4 List-Level metrics (Tier 3): Table 3
   - 4.5 Ablation: 8 組 configurations
   - 4.6 User study: 25 人 (Believability + Helpfulness)
   - 4.7 Efficiency: Table 4 (vs SimTube)

5. Discussion + Limitations (0.5 頁)
   - Post-hoc framing 的取捨與優勢
   - 領域 gap：缺真實時序觀影行為資料

6. Conclusion (0.5 頁)
```

### 9.2 學術 Contribution 重述

```
C1. System Contribution
   First lightweight (3B parameter) one-shot timeline-to-sparse-JSON 
   persona-conditioned video commentary generation system.
   8-persona sparse comment list per video on consumer GPU (24GB),
   single Claude API call per (video, persona) pair (~$0.015).

C2. Methodological Contribution
   Two-stage training paradigm for ground-truth-scarce + post-hoc setting:
   - One-shot distillation provides foundational sparse-JSON capability
   - 4-aspect list-level multi-reward RLAIF (含 novel R_frequency_match
     and R_coverage_diversity) provides domain breakthrough
   First to integrate Chapter-Llama-style text-level timestamp output + 
   Socratic Models-style all-modality-as-text + PersonaGym-style persona 
   evaluation in unified framework, applied to persona-conditioned video commentary.

C3. Empirical Contribution
   First evidence that 3B model can match or exceed 600B teacher (Claude) 
   on temporal localization F1 + persona-specific dimensions.
   Format Compliance Rate (FCR) demonstrates 3B model can rise from 32% 
   zero-shot to 96%+ with distillation.

C4. Framing Contribution (v4.1 獨有)
   Honest re-framing of "audience simulation" as "post-hoc commentary 
   generation" — eliminates future-leakage critique while aligning 
   with actual YouTube comment behavior.
```

---

## 10. 完整文獻清單

```
> 註：本節編號為 v4.1 計畫內部編號（[1]–[39]）。
> 完整作者、摘要、URL 等詳細資料請參見獨立檔案 SimLens_References.md
> （該檔對應主索引 [1]–[40]，編號與本節略有差異但內容對齊）。

=== 核心引用 ===

[1] SimTube (Hung et al., 2024) — arXiv 2411.09577
    用於：總體架構、影片理解 pipeline、自動評估指標、人類評估 protocol

[2] UMaT (Bi & Xu, 2025) — arXiv 2503.09081
    "Everything Can Be Described in Words: A Simple Unified Multi-Modal 
     Framework with Semantic and Temporal Alignment"
    用於：時序對齊 pipeline backbone、Timeline Script representation

[3] PersonaChat (Zhang et al., 2018) — ACL 2018
    用於：Persona schema 結構

[4] PersonaGym (Samuel et al., 2025) — EMNLP 2025 Findings
    用於：Persona Consistency + Linguistic Habits 評估 rubric

[5] PersoBench (Huang et al., 2024) — arXiv 2410.03198
    用於：Engagingness 評估維度（UniEval）

[6] Score Before You Speak (Saggar et al., 2025) — arXiv 2508.06886
    用於：Coherence 評估維度

[7] DPO (Rafailov et al., 2023) — NeurIPS 2023
    用於：Phase 2 訓練演算法

[8] RLAIF Survey (Lee et al., 2023) — Google DeepMind
    用於：證明 RLAIF 可達 RLHF 同等效果

[9] OpenCharacter — arXiv 2501.15427
    用於：證明用 teacher LLM 蒸餾 persona 行為

[10] Bias-Adjusted LLM Agents (Kitadai et al., 2025) — arXiv 2508.18600
     用於：Persona-based fine-tuning 概念

[11] LoRA (Hu et al., 2021) — ICLR 2022
     用於：Multi-LoRA per persona 技術基礎

[12] Self-Rewarding LM (Yuan et al., 2024) — arXiv 2401.10020
     用於：僅借鏡 Iterative DPO 概念

[13] LLM-as-Judge (Zheng et al., 2023) — NeurIPS 2023
     用於：LLM-as-Judge 方法論

[14] Replacing the Judge (SambaNova, 2024)
     用於：證明本地 LLM 可取代 GPT-4

[15] UniEval (Zhong et al., 2022) — EMNLP 2022
     用於：Engagingness 評估工具

[16] LLaVA-NeXT (Liu et al., 2024)
     用於：影片視覺理解

[17] Whisper (Radford et al., 2022)
     用於：影片語音轉錄

[18] Llama 3.2 (Meta, 2024)
     用於：Student model

[19] Tülu 3 (Lambert et al., 2024)
     用於：多階段 post-training 範式

[20] Sentiment Analysis in the Age of Generative AI (Hartmann et al., 2024)
     Customer Needs and Solutions, Springer Nature
     https://link.springer.com/article/10.1007/s40547-024-00143-4
     用於：Stage C 情感分類無需訓練的背書

[22] Evaluating LLMs for Sentiment Analysis (Lin et al., 2024)
     JMIR / PMC peer-reviewed
     https://pmc.ncbi.nlm.nih.gov/articles/PMC12526656/
     用於：YouTube 平台情感分類具體效能背書

[23] VideoMultiAgents (Kugo et al., 2025) — arXiv 2504.20091
     用於：Section 1.3 決策 A 背書（感知與推理分離）

[24] QMAVIS (Lin et al., 2026) — arXiv 2601.06573
     用於：Section 1.3 決策 B 背書（chunking + late fusion）

[25] Neeko (Yu et al., EMNLP 2024 Main) — arXiv 2402.13717
     "Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character 
      Role-Playing Agent"
     用於：Multi-LoRA per persona 設計直接前例

[26] Action-Guided Engagement — arXiv 2502.12073
     "Can LLMs Simulate Social Media Engagement?"
     用於：「行為缺失（不留言）也是訊號」的概念背書，
          支持 SimLens 用空陣列隱式編碼「沒反應」的設計

[27] Thakkar et al. (ACL 2024 Main) — arXiv 2406.04879
     "A Deep Dive into the Trade-Offs of Parameter-Efficient 
      Preference Alignment Techniques"
     用於：LoRA SFT → LoRA DPO 兩階段技術背書

[28] Multi-MLLM Knowledge Distillation (Gu et al., 2025) — arXiv 2505.22517
     用於：完整 LoRA SFT + LoRA DPO 兩階段 prior art

=== v4.1 新增引用 ===

[29] Chapter-Llama (Ventura et al., CVPR 2025) — arXiv 2504.00072 ⭐ NEW
     "Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs"
     https://arxiv.org/abs/2504.00072
     https://github.com/lucas-ventura/chapter-llama
     用於：Section 1.3 決策 C 直接前例（text-level video → LLM timestamp output）
          Section 3.5 Phase 1 學術依據（與 SimLens 99% 一致：Llama + LoRA + ASR/Caption timestamp 文字 timeline → 輸出 timestamp + 內容）
          Section 4.6 Reward A 設計背書（text-output timestamp 範式）
     SimLens 對比：Chapter-Llama 用 8B + 1hr 影片 + chapter title；SimLens 用 3B + 1-3min + per-persona LoRA + persona commentary

[30] Socratic Models (Zeng et al., ICLR 2023) — arXiv 2204.00598 ⭐ NEW
     "Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language"
     Google Research / DeepMind
     https://arxiv.org/abs/2204.00598
     https://socraticmodels.github.io/
     用於：Section 1.3 決策 C「all-modality-as-text」哲學祖師爺
          Section 3.5 Phase 1 Stage A Timeline Script 設計依據
     核心概念："language-based world-state history" — 將視覺/音訊全部降維為文字，讓 LLM 純在文字域推理

[31] MMDuet / VideoLLM Knows When to Speak — arXiv 2411.17991 ⭐ NEW
     "VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video 
      Comprehension with Video-Text Duet Interaction Format"
     https://arxiv.org/html/2411.17991
     用於：Section 1.3 決策 C event-driven sparse prediction 概念補充背書

[32] MM-When2Speak / Beyond Words — arXiv 2505.14654 ⭐ NEW
     "Beyond Words: Multimodal LLM Knows When to Speak"
     https://arxiv.org/html/2505.14654v1
     用於：multimodal "when to speak" 概念補充背書

[32] IFEval (Zhou et al., 2023) — arXiv 2311.07911 ⭐ NEW
     "Instruction-Following Evaluation for Large Language Models"
     https://arxiv.org/abs/2311.07911
     用於：Section 5.2 Group 0 verifiable instruction 哲學背書
          支持 SCR + TVR 採用 deterministic 程式可驗證設計

[33] JSONSchemaBench (Geng et al., 2025) — arXiv 2501.10868 ⭐ NEW
     "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs 
      for Language Models"
     https://arxiv.org/abs/2501.10868
     https://github.com/guidance-ai/jsonschemabench
     用於：Section 3.3 constrained JSON decoding 技術背書
          Section 5.2 Group 0 SCR (Schema Compliance Rate) 直接對應

[34] SoccerNet (Giancola et al., CVPR 2018) ⭐ NEW
     "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos"
     https://arxiv.org/abs/1804.04527
     用於：Section 4.3 Reward A (R_timing) action spotting saliency 概念背書

[37] MORLAIF (Williams, 2024) — arXiv 2406.07496
     "Multi-Objective Reinforcement Learning from AI Feedback"
     用於：Section 4.6 multi-aspect reward 學術背書

[38] Bootstrapping with Implicit Rewards (ICLR 2025)
     用於：Section 4.6 Iterative DPO 多輪迭代背書
```

---

## 11. Limitations（誠實面對的限制）

```
寫進論文 Limitations 章節（這是加分項）：

L1. Post-hoc Framing 而非真實即時模擬 ⭐ v4.1 新增
    SimLens v4.1 模擬的是「事後反思型評論」（post-hoc commentary），
    不是即時觀影反應（real-time viewer reaction）。
    這個 framing 與 YouTube 真實留言情境一致（觀眾本就看完才寫），
    但無法用於需要即時行為模擬的場景（如直播彈幕預測）。
    Future work：基於 streaming video LLM (e.g., VideoLLM-online) 
    擴展為真實即時模擬。

L2. No Real-World Persona Behavior Validation (Mitigated in v4.2) ⭐

    Gap：我們無「真實人類 × 已知 persona 標籤」的觀影行為資料集。
         此 gap 是 persona simulation 領域共通限制（SimTube / PersonaLLM / PersonaGym
         同樣面對），無法在本研究尺度內根本解決。

    Mitigation（v4.2 P0/P1 新增 — 三層獨立外部錨點）：

      層 1：Group 1+ 文本錨（同語言、群體層級分布對齊）
        - 採集 100 部影片 × 30 條真實 YouTube top comments = 3,000 條真實英文留言
        - 用 4 個 deterministic 指標驗證 SimLens 整體不偏離真實留言生態：
          Length KS / Sentiment Wasserstein / Style Marker χ² / Embedding Frechet
        - 不需「個別 persona ↔ 個別人」對應，量「群體層級分布距離」

      層 2：Group 1+ 時序錨（跨語言、群體層級熱區對齊）
        - 採集 50 部影片對應的 Bilibili 彈幕 timestamp = ~40,000 個秒級時間點
        - 只取 timestamp 密度，不取彈幕文本內容
        - 用 KDE peak detection 抽 crowd hotspot，計算 SimLens 預測 timestamps 的
          Recall / Precision @ ±5s
        - 跨語言可用：彈幕熱區是「事件層級」訊號，不是「文本層級」訊號

      層 3：Group 4 個體可信度錨（25 人 Upwork crowd study）
        - 沿用 v4.1 設計，提供個別 persona 層級的 third-party believability
        - 與層 1+2 的群體錨互補：群體錨打破封閉迴路，個體錨驗證 persona 可信度

    為什麼能緩解 v4.1 原始顧慮：
      ✗ v4.1 顧慮：「彈幕是匿名群體訊號，無法對應個人 persona」
      ✓ v4.2 回應：本就不要求個人對應，只用「群體層級時序密度」這一維度，
                   從根本上規避了 persona ↔ 個人配對的需求

      ✗ v4.1 顧慮：「Bilibili API 隱私保護與法律風險」
      ✓ v4.2 回應：(1) 優先引用學術界既有資料集（VideoLion / Bili-22）；
                   (2) 自爬僅取 timestamp 不取文本，不涉個人言論；
                   (3) 純學術非商業；(4) 同步報請 IRB；
                   (5) Fallback 到 YouTube most-replayed heatmap 完全合規

    殘餘 gap：
      Group 1+ 三個錨點層皆無法驗證「個別 persona 是否真實對應某類人類觀眾」，
      此最終 gap 由 Group 4 Human Eval 補強，但 25 人樣本量仍有限。
      根本解：F1 Future Work（200 人 × 50 部影片 segment-level ground truth dataset）。

L3. Distillation Bias
    Phase 1 用 Claude 當 teacher，可能繼承 Claude 的偏誤
    （例如過度禮貌、避開敏感話題）。
    Phase 2 RLAIF 部分校正，但無法完全消除。

L4. LLM-as-Judge Limitations
    SimLens 採用單一 Qwen3-32B-Q4 作為 frozen judge（無 multi-judge ensemble、
    無 GPT-4 spot-check）。Qwen3-32B 與 GPT-4 一致性約 85-90%（社群報告），
    這個 ~10-15% 差距可能引入評分偏誤。Future work 可加入 multi-judge ensemble
    或 GPT-4 spot-check 強化評分可靠度。

L5. English-Only & Cultural Bias
    8 個 persona 都是英文 + 美國/亞洲文化導向。
    中文 / 跨文化擴展為 future work。

L6. Length-Generalization Constraint
    SimLens scope 嚴格鎖定在 1–3 分鐘 YouTube 短影片，10 秒固定分段。
    刻意排除：
      - ≤60s Shorts（分段邊際價值低、persona 區分能力受限）
      - > 3min 中長片（timeline script 過長、Claude 蒸餾品質下降）
    這是有意的方法論選擇而非資料限制，詳見 §1.4 Scope Rationale。
    Shorts 場景由 future work F5 處理（需要重新設計 persona 活躍度 baseline）；
    > 5 分鐘長片由 future work F6 處理（需 hierarchical summarization）。

L7. Sparse JSON Format Brittleness ⭐ v4.1 新增
    雖然 constrained decoding 可達 96%+ FCR，但仍有 ~4% 案例輸出失敗。
    生產環境需 fallback 機制（retry、format repair）。

L8. List-Level Reward 設計依賴 persona expected_comment_count_range
    R_frequency_match 依賴我們設定的 expected range，
    沒有真實資料驗證這個假設。
    這是 persona-based simulation 領域的共通限制。

L9. Two-Stage Pipeline Choice
    SimLens 採用 SFT + DPO 兩階段，未採用單階段方法（如 ORPO、DPO+NLL）。
    Trade-off：訓練流程複雜度高（8 LoRA × 2 phase × 2 round = 32 次訓練 run），
    但保留 SFT/DPO 各自獨立的 ablation slot（A2/A3）。
    Future work 可比較 ORPO single-stage 在 3B + persona 領域的表現。

L10. Self-Rewarding 不適用於 SimLens 規模
    社群實驗顯示 3B 規模下 self-judging quality 會崩。
    SimLens 採用 external judge (Qwen3-32B-Q4) 繞開此限制。
    Future work：當 SimLens base model 升級到 7B+ 時，
    可探索切到 self-rewarding 範式以省去 external judge。

L11. Persona Specification from PersonaChat 8K Sampling Constraint
    SimLens 採用 PersonaChat 8K + cosine similarity + MMR 取樣 8 個 personas
    （§2.3）。雖然此範式重現 SimTube (IUI 2025) 並排除手編偏誤，但 PersonaChat
    本身為 2018 年資料集，可能未涵蓋最新世代的觀眾類型（如 TikTok/Shorts 新興
    族群）。Future work 可比較 PersonaChat vs PersonaHub (Tencent 1B) vs
    OpenCharacter 多源取樣的 persona 多樣性影響。

L12. Teacher Timestamp Fidelity Assumption ⭐ v4.1 新增
    SimLens Phase 1 蒸餾依賴一個關鍵假設：Claude Sonnet 4.5 在 zero-shot 設定下
    能正確產生「輸入 timestamp ↔ 輸出 timestamp」的對應關係（即輸出的 [00:15]
    確實對應輸入 timeline 中 00:15 的事件）。

    此能力**已被 Chapter-Llama (CVPR 2025) 在 LoRA-tuned Llama-3.1-8B 上驗證**
    （tIoU=71.8 on VidChapters-7M），但**未在 zero-shot Claude / GPT-4o 上正式
    驗證**（Chapter-Llama 論文未做 zero-shot baseline 的 timestamp fidelity 分析）。

    緩解策略：
      1. Week 3 Day 1 跑 Pre-flight Sanity Check（5 部 × 8 personas = 40 個
         sparse list，人工檢查 ~50 個 timestamp 對應關係，通過標準 ≥ 80%）
      2. 若 sanity check < 60%，fallback 到 **GPT-4o-mini** 當 teacher
         （v4.1 預算決策：OpenAI 一律用 mini；mini 的 IFEval 表現仍優於 Claude
          Sonnet 的某些 instruction-following 維度，可作為合理 fallback）
      3. 若 GPT-4o-mini 也 < 60%，啟動二級緩解（重寫 prompt + 加 3 個 few-shot
         範例）；最後手段才升級到 GPT-4o（額外 ~$25 預算）
      4. 將 sanity check 通過率寫進論文 Section 4.1（Experimental Setup 透明度）

    Future work 可正式評估 zero-shot LLM 在 「text-level video commentary
    generation with timestamp output」 任務上的 fidelity benchmark。
```

---

## 12. Future Work

```
F1. Real persona ground truth via user study
    招募 200 人 × 標註 demographics × 看 50 部影片 + 段層級評論
    → 真正的 segment-level persona ground truth dataset

F2. Adaptive Video Segmentation
    從固定 10 秒 → 基於內容（鏡頭切換、話題轉換）的自適應分段

F3. Hierarchical Persona LoRA
    粗 persona（demographics）+ 細 persona（individual quirks）
    兩層 LoRA 結構

F4. Cross-cultural Extension
    中、英、日、韓四語 persona，端到端中文化版本

F5. Shorts (≤60s) 擴展
    重新設計 persona expected_comment_count baseline 為 30s 級
    重新評估 UMaT/QMAVIS motivation 在 1–6 段場景的適用性
    建立 Shorts 專用 SimTube baseline（SimTube 也未做 Shorts）
    這是 v4.1 嚴格 scope 收緊後最直接的擴展方向

F6. Long-form Video Support (> 3min)
    結合 hierarchical summarization 處理 5+ 分鐘影片
    或借鏡 Chapter-Llama 的 speech-guided frame selection 處理超長影片
    處理 timeline script 超過 4096 max_seq_length 的問題

F7. Real-time Director Mode（移除 post-hoc 限制）
    結合 streaming video LLM (e.g., VideoLLM-online, arXiv 2406.11816)
    從事後反思 → 真實即時觀影反應模擬

F8. Sliding-Window for Very Long Videos
    對於 > 5min 影片，採滑動窗口切片 + 多次 one-shot reflection 後合併
```

---

# 附錄：給你的具體 Action Items

## 本週可立即開始

```
□ Day 1: 確認硬體（最低門檻 24GB VRAM；本研究實際用 RTX 5090 32GB）
□ Day 2: 安裝環境（LLaMA-Factory / TRL / Ollama / Outlines）
□ Day 3: Pull Llama-3.2-3B + Qwen3-32B Q4 + LLaVA-NeXT
□ Day 4: 下載 PersonaChat 8K (Hugging Face: bavard/personachat_truecased)
         定義 sparse JSON schema（用於 constrained decoding）
□ Day 5: 用 YouTube API 收集 5-10 部測試影片（驗證 pipeline）
         提取 keywords，跑 §2.3 PersonaChat → top-8 取樣腳本
□ Day 6-7: 跑通 Whisper + LLaVA + UMaT-inspired Timeline Script pipeline
            對 1 部影片試跑：Claude 一次餵入 → 收 sparse JSON
            驗證 Format Compliance（手動檢查 8 個 personas 的輸出）
```

## 投稿目標確認

```
首選：ACM MM 2026 BNI（deadline ~ 7 月）— 8 頁短文
備選：UIST 2026 Posters / Demos（deadline ~ 7/10）— 4 頁
保底：智慧創新大賞 2026 + GitHub 開源 + Hugging Face release
```

## SimLens 核心設計清單（v4.2）

```
架構：
+ 全局 Timeline Script（UMaT-inspired，一次餵入）
+ Sparse JSON 輸出格式（[{timestamp, comment}, ...]）
+ Constrained JSON decoding（Outlines/XGrammar）

評估體系：
+ Group 0: Schema Compliance Rate + Timestamp Validity Rate
+ Group 1+: ⭐ v4.2 新增 Real-World Anchoring（4 文本錨 + 2 時序錨）
            - YouTube top comments：Length KS / Sentiment Wasserstein / Style χ² / Embedding Frechet
            - Bilibili 彈幕 timestamp：Crowd Hotspot Recall / Precision @ ±5s
+ Group 2: Tier 2 內容指標（PersonaGym + Coherence + Engagingness）
+ Group 3: List-level metrics (reward-independent)：Persona Content Distinctiveness
+ Group 4: 25 人 Likert human eval

⭐ v4.2 新增診斷指標（§5.2.7 Appendix）：
+ D1 TAS（v4.1 主指標 → v4.2 診斷）：僅檢查蒸餾失敗，不發強 claim
+ D2 Multi-Judge Spot-Check（Qwen vs GPT-4o-mini Cohen's κ）
+ D3 Pre-flight Teacher Sanity Check

訓練設計：
+ Phase 1 SFT：Claude 蒸餾 + LoRA per persona（8 個獨立 adapter）
+ Phase 2 DPO：4-aspect list-level reward
  - R_timing (30%)：時機合理性
  - R_frequency_match (25%)：list 長度匹配 persona 活躍度
  - R_content_quality (25%)：persona 一致 + 局部相關
  - R_coverage_diversity (20%)：timestamp 分布均勻

定位：
✓ Post-hoc commentary generation（事後反思評論）
✗ 不 claim 即時觀影行為模擬
✗ 不 claim 個別 persona ↔ 個別真實人類對應（外錨僅為群體層級）

成本（v4.2 更新）：
- Claude API：$12 USD（800 calls × $0.015）
- 雲端 GPU：$30-60 USD
- 人類評估：$300-500 USD（標準）
- Multi-Judge Spot-Check：$3.2 USD（v4.2 新增，GPT-4o-mini × 80 sample × 8 維度）
- Real-World Anchor 採集：$0（YouTube + Bilibili 公開 API）
- 總預算：$345-575 USD
```

---

## 13. v4.2 Changelog（vs v4.1）

```
═══════════════════════════════════════════════════════
總結：v4.2 = v4.1 + Real-World Anchoring（破除 Claude+Qwen 封閉迴路）
═══════════════════════════════════════════════════════

[v4.1 的核心問題]
  訓練訊號鏈：Claude 蒸餾 → Qwen reward → Qwen evaluation judge
  除 Group 4（25 人 Upwork）外，所有訊號皆來自 LLM。
  封閉論證風險：「Student 越像 Claude/Qwen 共識 → 分數越高」

[v4.2 的解決方案：三層獨立外部錨點]
  Group 1+ 文本錨：3,000 條真實 YouTube top comments
  Group 1+ 時序錨：~40,000 個 Bilibili 彈幕 timestamp（不取文本）
  Group 4 個體錨：25 人 Upwork（v4.1 既有，保留）

═══════════════════════════════════════════════════════
🔴 P0 改動（必要 — 直接打破封閉迴路）
═══════════════════════════════════════════════════════

[+] §3.2 Step 1.4：Real-World Anchor 資料採集
    - 100 部 × 30 條 YouTube top comments = 3,000 條真實英文留言
    - 50 部對應 Bilibili 彈幕的秒級 timestamp（不取文本，規避隱私）
    - Fallback：YouTube most-replayed heatmap（合規退路）

[+] §5.2.5 Group 1+: Real-World Anchoring（新評估章節）
    文本錨（4 個 deterministic 指標）：
      - Length KS-statistic ↓
      - Sentiment Wasserstein distance ↓
      - Style Marker χ² ↓
      - Embedding Frechet Distance ↓
    時序錨（2 個跨語言指標）：
      - Crowd Hotspot Recall @ ±5s ↑
      - Crowd Hotspot Precision @ ±5s ↑
    Persona plausibility（弱錨點）：
      - Real Comment LoRA Coverage（Avg PPL + Coverage %）

[+] §5.4 Table 6: Real-World Anchor 結果（新主結果表）
    對比：Llama-3B / Claude / GPT-4o-mini / SimTube / SimLens Phase 1 / SimLens Full
    所有指標 deterministic 計算，無 LLM judge

═══════════════════════════════════════════════════════
🟡 P1 改動（強烈建議 — 進一步堵 reviewer 攻擊面）
═══════════════════════════════════════════════════════

[~] §5.2 Group 3：Teacher Alignment Score (TAS) 從主指標降為診斷
    v4.1：Table 3 List-Level 主指標，列為「驗證蒸餾成功」
    v4.2：Appendix B 診斷指標，僅檢查蒸餾災難性失敗（合理區間 0.5–0.85）
    動機：避免「越像 Claude → 越高分」的封閉論證

[+] §5.2.7 Diagnostic D2: Multi-Judge Spot-Check（新章節）
    - 80 sample × Qwen3-32B-Q4 + GPT-4o-mini 平行評分
    - 計算 Pearson r + Cohen's κ
    - κ ≥ 0.6 → 主結果穩定
    - κ < 0.6 → 主結果註明 judge sensitivity，必要時 GPT-4o-mini 重跑
    - 成本：$3.2 USD

[+] §5.2.7 Diagnostic D3: Pre-flight Teacher Sanity Check
    沿用 v4.1 §11 L12 緩解策略，正式納入 Diagnostic 層

═══════════════════════════════════════════════════════
🟢 P3 改動（建議 — framing 更新）
═══════════════════════════════════════════════════════

[~] §11 L2 重寫
    v4.1：「Bilibili 彈幕已移除（無法對應個人 persona + 隱私風險）」
    v4.2：「Bilibili 用於 cross-lingual temporal hotspot anchoring（mitigated）」
          - 三層緩解：文本錨 / 時序錨 / 個體錨
          - 群體層級訊號規避 persona 配對需求
          - 只取 timestamp 不取文本規避隱私法律風險

[~] §0 一頁摘要：新增貢獻 C4 Real-World Anchoring 範式
[~] §0 「明確不做」清單：新增「不 claim 個別 persona ↔ 個別真實人類對應」
[~] 標題副標：新增 "Anchored on Real-World Comment & Danmaku Signals"

═══════════════════════════════════════════════════════
v4.2 未改動（仍是 v4.1 設計）
═══════════════════════════════════════════════════════

⚠ 以下三個「硬加技術」嫌疑點，v4.2 暫未動，原因見 §13.1：
  - Iterative DPO 2 輪
  - R_coverage_diversity（與 sparse event-driven 哲學的潛在矛盾）
  - R_frequency_match（從 expected_comment_count 反推的循環）
  - 預期結果表「3B 全面超越 600B Teacher」的數字

  Rationale：v4.2 的 P0/P1/P3 改動目標明確 = 打破封閉迴路。
            上述四點屬於「方法簡化 / 結果降溫」範疇，與封閉迴路問題正交，
            另列 §13.1 作為 v4.3 候選改動。

### 13.1 v4.3 候選改動（不在 v4.2 範圍內）

下列改動已被識別但暫不納入 v4.2，理由：v4.2 的設計目標是「最小改動破環」，
保留可發表性與時程穩定，避免一次大改造成風險擴散。

[ ] R_coverage_diversity 移除或改設計
    潛在問題：「timestamp 應均勻分布」與 sparse event-driven motivation 矛盾
    候選方案 A：直接移除，權重併入 R_timing（30 → 50%）
    候選方案 B：改為 R_hotspot_recall（錨於 Bilibili 彈幕熱區，從外部錨派生 reward）

[ ] R_frequency_match 移除為 reward，降為 prompt-level constraint
    潛在問題：expected_comment_count 是 Claude 估的 → 用它當 reward 形成循環
    候選方案：在 prompt 裡告知 expected count，不做為 RLAIF 訊號
    權重重分配：R = 0.40 R_timing + 0.40 R_content + 0.20 R_coverage（或 R_hotspot）

[ ] Iterative DPO 從 2 輪降為 1 輪 + 權重 sweep
    潛在問題：3B + 800 pairs 訊號量不足以支撐 2 輪迭代
    候選方案：1 輪 DPO + 跑 3-4 組權重 sweep（更具方法論說服力）

[ ] 預期結果數字降溫（§5.4 Table 1a/1c）
    潛在問題：「3B 全面超越 Claude Sonnet 4.5」過於樂觀，pilot evidence 不足
    候選方案：從「全面超越 Teacher」→「在 1-2 維度上接近，其他維度持平」
    時機：Pilot run 後再校準

→ v4.3 是否啟動，視 v4.2 P0/P1 跑出的真實數字決定。
   若 Group 1+ Real-World Anchor 數字證明 Phase 2 真的有額外貢獻，則 R_freq / R_cov 保留；
   若 Phase 1 SFT 已接近 Phase 2 在外錨上的表現，則啟動 v4.3 簡化 reward 設計。
```
