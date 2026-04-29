# SimLens 完整研究計畫 v4.1
## "Event-Driven Persona-Conditioned Video **Commentary Generation** via Sparse Temporal Prediction and RLAIF"

> 版本：v4.1（Post-hoc Reflection 版本）
> 投稿目標：ACM MM 2026 BNI / UIST 2026 Posters / Demos / 智慧創新大賞 2026
> 上一版：v2.0（cumulative segment-by-segment 版本，已棄用）
> v4.0（事件驅動但仍 cumulative）→ v4.1（一次餵入 + 大方承認 post-hoc）

---

## 0. 一頁摘要（The One-Page Summary）

```
研究問題：
   能否用 3B 小模型在「缺乏真實 persona 觀影行為資料」的場景下，
   訓練出能對短 YouTube 影片生成 persona-specific、時序定位精確的多受眾評論
   生成系統？
   （Scope: 1–3 minute YouTube videos，刻意排除 ≤60s Shorts —— 詳見 §1.4 scope rationale）

關鍵 framing 校正（v4.0 → v4.1）：
   ✗ 不再 claim「模擬即時觀影體驗」
   ✓ 重新定位為「事後反思型 AI 評論生成」（post-hoc commentary generation）
   ✓ 此定位與 SimTube、YouTube 真實留言情境一致
       —— 觀眾本來就是「看完整片才寫評論」

核心方法：
   Stage A：UMaT-inspired temporal alignment
     - Whisper（含時間戳）+ LLaVA-NeXT（每 10 秒分段）→ 結構化 Timeline Script
     
   Stage B：兩階段事件驅動訓練（One-Shot 全片輸入）
     - Phase 1（蒸餾）：一次餵全片 Timeline Script → Claude 輸出 Sparse JSON
                       [{timestamp, comment}, ...] → SFT Llama-3B
     - Phase 2（RLAIF）：Qwen3-32B 用 4-aspect reward → DPO

   Stage C：報告生成（同一個 Llama-3B base）
     - 從 (timestamp, comment) 列表整合出跨受眾比較與改善建議

關鍵差異化（vs SimTube vs v4.0）：
   SimTube：影片 → 整體理解 → 1 條整片評論 / persona
   SimLens v2.0（廢棄）：12 段 × 8 persona × cumulative call → 96 次 API call / 影片
   SimLens v4.1：1 次全片輸入 → 8 條 Sparse JSON list → 8 次 API call / 影片
                 每 persona 輸出 [(t1, c1), (t2, c2), ...]，N_comments 由 persona 自決
   
   Token 成本下降約 90%、架構更簡潔、且符合真實 YouTube 留言行為情境。

預期貢獻：
   C1. System：首個 segment-localized persona-conditioned 影片評論生成系統
   C2. Method：one-shot timeline-to-sparse-JSON 蒸餾 + 4-aspect reward + DPO
              (None 反應自然編碼為「該 persona 不在此 timestamp 出現」)
   C3. Empirical：3B student 在時序定位 + persona 區分上接近 Claude，且 on-device

明確不做（誠實 framing）：
   ✗ 留存曲線預測
   ✗ 跳出率預測
   ✗ 任何宣稱「即時行為模擬」的 claim
   ✗ Future-leakage 規避（我們大方承認看完全片才生成，符合 post-hoc 定位）
   ✗ 任何需要對應「真實段層級觀眾行為」的指標
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

### 1.1 v2.0 → v4.1 架構對比

| 維度 | v2.0（廢棄）| **v4.1（新版）**|
|---|---|---|
| 輸入方式 | 每段 cumulative 餵入 | **一次餵入全片 Timeline Script** |
| 推理單位 | (segment_i, persona_p) cell | **(video, persona_p) sparse list** |
| API call 次數 / 影片 | 12 段 × 8 persona = 96 | **1 × 8 persona = 8** |
| 輸出格式 | 每段一個 comment 或 "None" | **稀疏 JSON [(t, c), ...]** |
| "None" 處理 | 顯式 None 標籤 | **隱式（不出現該 timestamp）**|
| Token 成本 | 高（每段重餵 cumulative）| **約原 1/10** |
| 模擬聲稱 | 即時觀影反應（有破口）| **事後反思評論（誠實）** |

### 1.2 架構設計依據

| 設計元素 | 來源文獻 | 為什麼這樣選 |
|---------|---------|-------------|
| Whisper-Large-v3 + LLaVA-NeXT | SimTube (Hung et al., 2024) | 直接借鏡 SimTube multimodal pipeline |
| **時序對齊 + 結構化文字** | **UMaT (Bi & Xu, 2025, arXiv 2503.09081)** | 將視覺與聽覺輸入統一為結構化文字 |
| 10 秒等長分段 | UMaT structured segmentation | 固定段長度避免 fragmentation |
| **One-shot timeline → sparse JSON** | **VTG-LLM (AAAI 2025, arXiv 2405.13382)** | 把 timestamp 當 token 學的範式直接前例 |
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
- **VideoMultiAgents (Kugo et al., arXiv 2504.20091)** 在 Intent-QA 達 79.0%（+6.2% over SOTA），證明專門代理人 + 獨立文字報告能避免單一巨型模型的黑箱干擾。
- **UMaT (Bi & Xu, arXiv 2503.09081)** 主張將視覺與聽覺降維為「統一文字表示」，提供 interpretability 與 structured retrieval 能力。

**學術背書 2：原生多模態模型的時序理解仍有缺陷**
- **VBenchComp / Time Blindness 系列**：頂尖原生多模態模型（GPT-4o、Gemini）對影片存在「shuffling invariance」——影格打亂順序，輸出仍幾乎不變，顯示依賴語言先驗而非真實時序推理。

#### 決策 B：為什麼將影片分段（每 10 秒）？

**學術背書 1：分段是時間軸對齊的最穩健方式**
- **UMaT (Bi & Xu, 2025)** 明確指出，要在影片任務中維持語義與時間一致性，必須將視覺描述與 ASR 轉錄「依時間戳切分為結構化片段」。

**學術背書 2：分段能規避視覺模型的記憶體與品質下降問題**
- **QMAVIS (Lin et al., arXiv 2601.06573)** 證明 chunking + late fusion 在 VideoMME 長影片基準上比端到端原生多模態模型**準確率高 38.75%**。

#### 決策 C ⭐：為什麼採用「一次餵入 + 事後反思」而非「逐段 cumulative」？

**潛在質疑**：「Cumulative 才能模擬即時觀影體驗，一次餵入會 future leakage 啊？」

**學術背書 1：Post-hoc commentary 是更誠實的學術定位**
- 真實 YouTube 留言**本就是看完整片才寫**——觀眾並非邊看邊即時打字。事後反思 framing 與真實留言情境一致。
- SimTube (Hung et al., 2024) 本身也是事後對整片給評論，學界接受此 framing。
- 若用 cumulative 假裝即時，反而引入「假裝有時序資訊」的破口；reviewer 會問：「你怎麼證明 cumulative 反應跟真實即時行為一致？」（你沒這個資料集）。
- 大方承認 post-hoc，反而 scope 乾淨、無破口。

**學術背書 2：Sparse temporal prediction 已有頂會直接前例**
- **VTG-LLM (AAAI 2025, arXiv 2405.13382)**：把 timestamp 當 token 學，要求模型一次輸出多個時間點。SimLens 的 sparse JSON 正是此範式 + commentary generation。
- **MMDuet (arXiv 2411.17991)**：「VideoLLM Knows When to Speak」，事件驅動稀疏預測。
- **MM-When2Speak (arXiv 2505.14654)**：多模態 LLM 判斷何時說話。

**學術背書 3：成本與工程效率**
- v2.0 cumulative 設計：96 cells × Claude API call = ~$1.20/影片
- v4.1 one-shot 設計：8 calls × Claude API call = ~$0.10/影片
- **成本下降 ~90%**，可釋放預算給更多影片或更多 persona。

**對 SimLens 的意義**：v4.1 不是「為了省錢的退讓」，而是「**移除 v2.0 即時性偽裝後的更誠實架構**」。Reviewer 會欣賞這種誠實 framing。

### 1.4 Scope Rationale：為什麼是 1–3 分鐘 YouTube 影片，而不是 Shorts？

**SimLens v4.1 scope 嚴格鎖定：1–3 分鐘 YouTube 短影片，刻意排除 ≤60s Shorts。**

潛在質疑：「TikTok / Reels / YouTube Shorts 才是 short-form 主流，為什麼不做 Shorts？」

#### 排除 Shorts (≤60s) 的四個理由

**理由 1：分段架構在 Shorts 上邊際價值低**
- ≤60s 影片只有 1–6 段，segment-level 分析能展現的「persona 對不同段反應差異」訊息量受限
- LLaVA-NeXT 對 60s 整段直接處理仍在能力範圍內，**UMaT/QMAVIS 引用的「長影片需分段」motivation 站不穩**
- 60s 內 sparse JSON 平均只有 1–3 個 timestamp，sparse 與 dense 預測差別不顯著

**理由 2：Persona 區分能力需要足夠時間軸**
- 8 個 persona 的差異化主要體現在「**對哪些段反應、對哪些段不反應**」的 timestamp 集合差異
- 60s 內全部 persona 可能都集中反應在同一個爆點，**Persona Differentiation (Jaccard) 指標失效**
- 1–3min 對應 6–18 段，足以展現 8 個 persona 在時間軸上的真實差異

**理由 3：與 SimTube baseline 對標基準明確**
- SimTube (Hung et al., 2024) 實驗用的影片長度雖未嚴格限定，但實際分布偏向 1–10min 中短片
- SimLens 鎖定 1–3min 與 SimTube 主流區間有清楚交集，Table 1 baseline 數字可比
- 純 Shorts 場景下 SimTube 沒有對應實驗，跨域比較會失準

**理由 4：訓練資料與 reward 設計連動匹配**
- Persona expected_comment_count 以 2min 影片為 baseline（如 P1 high → 3-6 條 / 2min）
- R_frequency_match 與 R_coverage_diversity 都依賴「足夠長的時間軸來展開」
- 30s 影片若強要 P1 留 3 條評論 = 每 10s 一條，**違反真實留言行為分布**

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

## 2. Persona 設計（不需要任何資料）

### 2.1 為什麼選 8 個 persona

```
理論依據：
1. PersonaChat (ACL 2018)：8K personas，但 SimTube 實驗時只用 top-30
2. PersonaGym (EMNLP 2025)：200 personas 評估，每任務只用 5 個
3. SimTube (2024)：crowd-sourced study 用 8 部不同類型影片

對 SimLens 的數量決策：
- 太少（<5）：persona 多樣性不足
- 太多（>15）：每 persona 訓練資料稀釋
- 8 個是甜蜜點：每 persona 約 100-150 部影片的 sparse list，
                平均每部 0-5 條評論（隨 persona 活躍度浮動）
```

### 2.2 8 個 SimLens Persona Schema

採用 PersonaGym 標準六層描述（Demographics + Interests + Personality + Viewing Habits + Linguistic Style + Engagement Pattern），新增 **expected_comment_count** 欄位以對應 sparse JSON 輸出：

#### **P1: 18-24 大學女性（社群活躍型）**
```yaml
demographics: {age: 18-24, gender: female, occupation: university student, location: urban Asia}
interests: [K-beauty, fashion trends, travel vlogs, K-pop, lifestyle]
personality: extroverted, trend-conscious, peer-influenced
viewing_habits: {length: 5-15min, platforms: [Instagram, TikTok, YouTube Shorts], engagement: heavy}
linguistic_style: {tone: enthusiastic, phrases: [OMG, love this, so cute, obsessed], emoji: very high}
expected_comment_count: high (3-6 條 / 2min 影片，活躍型)
```

#### **P2: 25-34 上班族男性（科技分析型）**
```yaml
demographics: {age: 25-34, gender: male, occupation: tech professional, location: urban}
interests: [tech reviews, gadgets, finance, productivity tools]
personality: analytical, skeptical, data-driven
viewing_habits: {length: 10-20min, platforms: [YouTube, Twitter/X], engagement: rare but substantive}
linguistic_style: {tone: measured, phrases: [actually, IMO, the real question is], emoji: minimal}
expected_comment_count: low (0-2 條 / 2min 影片，潛水型)
```

#### **P3: 25-34 上班族女性（職涯導向型）**
```yaml
demographics: {age: 25-34, gender: female, occupation: marketing/consulting, location: urban}
interests: [career development, work-life balance, finance, premium travel]
personality: goal-oriented, aesthetically aware, time-conscious
linguistic_style: {tone: polished, phrases: [great insight, takeaway, totally relatable], emoji: low-moderate}
expected_comment_count: medium (1-3 條 / 2min 影片)
```

#### **P4: 35-44 已婚父母（家庭實用型）**
```yaml
demographics: {age: 35-44, gender: any, occupation: parent + employed, location: suburban}
interests: [parenting, family travel, home improvement, finance, wellness]
personality: practical, value-focused, time-constrained
linguistic_style: {tone: warm, phrases: [as a parent, my kids, this reminds me], emoji: moderate}
expected_comment_count: medium (1-3 條 / 2min 影片)
```

#### **P5: 45-54 中年男性（傳統權威型）**
```yaml
demographics: {age: 45-54, gender: male, occupation: established professional, location: any}
interests: [news, investment, traditional hobbies, documentaries]
personality: opinionated, traditional, authority-respecting
linguistic_style: {tone: authoritative, phrases: [back in my day, the real issue is, frankly], emoji: very low}
expected_comment_count: low (0-1 條 / 2min 影片，最潛水型)
```

#### **P6: 18-24 大學男性（遊戲動漫宅）**
```yaml
demographics: {age: 18-24, gender: male, occupation: student/entry-level, location: any}
interests: [gaming, esports, anime, meme culture]
personality: playful, ironic, peer-influenced
linguistic_style: {tone: ironic, phrases: [based, W, L, no cap, fr, this slaps], emoji: moderate ironic}
expected_comment_count: high (3-5 條 / 2min 影片，迷因型)
```

#### **P7: 55+ 退休族群（懷舊溫暖型）**
```yaml
demographics: {age: 55+, gender: any, occupation: retired/semi-retired, location: any}
interests: [health, leisurely travel, traditional cooking, philosophy/religion]
personality: reflective, nostalgic, warmth-valuing
linguistic_style: {tone: warm, phrases: [thank you for sharing, brings back memories, blessed], emoji: low-moderate}
expected_comment_count: medium-low (1-2 條 / 2min 影片)
```

#### **P8: 13-17 青少年（潮流迷因型）**
```yaml
demographics: {age: 13-17, gender: any, occupation: middle/high school student, location: any}
interests: [viral content, memes, gaming, music/dance, school life]
personality: peer-conscious, trend-driven, expressive
linguistic_style: {tone: high energy, phrases: [LMAO, fr fr, no way, that's so me], emoji: very high}
expected_comment_count: very high (4-7 條 / 2min 影片，但短)
```

### 2.3 Persona 設計學術依據

| 設計元素 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| 六層 schema 結構 | PersonaGym (Samuel et al., EMNLP 2025) | demographics + linguistic habits + behavior |
| Demographics 細節 | PersonaChat (Zhang et al., ACL 2018) | occupation + location + interests |
| Linguistic style 設計 | Bias-Adjusted LLM Agents (Kitadai et al., arXiv 2508.18600) | individual-level 行為差異化 |
| Viewing habits 加入 | SimTube (Hung et al., 2024) | 影片觀眾模擬特有元素 |
| **expected_comment_count** | **本研究新增** | **取代 v2.0 的 reaction_frequency，直接編碼 sparse list 長度** |

---

## 3. Phase 1：蒸餾（Knowledge Distillation）

### 3.1 目標

讓 Llama-3.2-3B 繼承 Claude-3.5 Sonnet 的「**事後反思型 sparse JSON 評論生成能力**」。

### 3.2 影片資料準備

```
═══════════════════════════════════════════════════════
Step 1.1: 影片素材收集
═══════════════════════════════════════════════════════
   注意：這只是「素材」，不是 ground truth
   
   來源：YouTube Data API v3
   數量：100 部短 YouTube 影片 (1–3min, avg ~2min)
   類型分佈：
     - Vlog/Lifestyle：20 部
     - Tech Review：20 部
     - Food/Cooking：20 部
     - Education/How-to：20 部
     - Entertainment/Comedy：20 部
   篩選條件：
     - 英文（簡化評估）
     - 有 official captions
     - 公開且 view > 10K
     - 無 18+ 標籤
     - **長度嚴格 60–180 秒**（用 YouTube API videoDuration filter "medium"
       後再以 contentDetails.duration ISO 8601 二次過濾）
     - **排除 #shorts tag 與 vertical 9:16 影片**
       （Shorts 與一般影片演算法生態不同，避免污染訓練分布）

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
       不再產生 cumulative narrative。直接產生整片可一次餵入的劇本：
       
       === Timeline Script ===
       [00:00-00:10] Visual: <LLaVA 段描述>
                     Audio: <該段 Whisper 文字>
       [00:10-00:20] Visual: ...
                     Audio: ...
       ...
       [02:50-03:00] Visual: ...
                     Audio: ...
       === End ===
       
   產出：100 個影片 × 1 個 Timeline Script per video（不再是 12 個 cumulative）

═══════════════════════════════════════════════════════
Step 1.3: Claude 蒸餾資料生成（核心 v4.1 改變）
═══════════════════════════════════════════════════════
   
   對每個 (影片 V, persona P)：    ← 注意：不再對每段獨立呼叫
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
   │ would have left a comment, given your persona's:            │
   │ - Expected comment count: {persona.expected_comment_count}  │
   │ - Linguistic style: {persona.linguistic_style}              │
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
     
   為什麼此設計優於 v2.0：
     (1) "無反應" 自然編碼為「不出現該 timestamp」，無需 None 標籤
     (2) Persona 的活躍度直接由 list 長度體現，更貼近真實行為
     (3) Sparse JSON 是 VTG-LLM (AAAI 2025) 已驗證的格式
     (4) Token 成本：800 calls vs v2.0 的 9,600 calls，下降 ~92%
     (5) 大方承認 post-hoc，無 future leakage 破口
   
   成本估算：
     Timeline Script avg ~2200 tokens (in) + persona ~200 tokens (in)
     Output avg ~500 tokens
     Per call: ~$0.015
     Total: 800 calls × $0.015 = $12 USD
     （比 v2.0 的 $115 USD 下降 ~90%）
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
        "epochs": 3,                  # v2.0 是 2，因為樣本變少需多 epoch
        "batch_size": 2,              # 序列變長（整片 timeline）
        "gradient_accumulation": 8,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "max_seq_length": 4096        # v2.0 是 2048，整片 timeline 需更長
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
| **時序對齊 Timeline Script** | **UMaT (Bi & Xu, arXiv 2503.09081)** | structured text representation |
| **One-shot timeline → sparse JSON** | **VTG-LLM (AAAI 2025, arXiv 2405.13382)** | timestamp 當 token 學的直接前例 |
| **Sparse temporal prediction** | **MMDuet (arXiv 2411.17991)** | 「VideoLLM Knows When to Speak」 |
| LoRA per persona | **Neeko (EMNLP 2024, arXiv 2402.13717)** | per-character LoRA 已被證明優於 single LoRA + prompt |
| 4-bit GPTQ + LoRA rank 8 | LLaMA-Factory 官方文件、Thakkar et al. (ACL 2024) | 標準 PEFT 工作流 |
| **LoRA SFT 後續可接 DPO** | **Thakkar et al. (ACL 2024, arXiv 2406.04879)** | 300+ 實驗證明 LoRA-SFT → LoRA-DPO 範式可行 |
| **兩階段 (SFT + DPO) on LoRA** | **Multi-MLLM Distillation (Gu et al., arXiv 2505.22517)** | 直接前例 |
| **Constrained JSON decoding** | **JSONSchemaBench (arXiv 2501.10868)、StructEval (arXiv 2505.20139)** | 確保 sparse list 輸出格式合規 |
| **「沒反應」隱式編碼為空陣列** | 本研究新增（受 VTG-LLM 啟發）| 取代 v2.0 顯式 None 標籤 |

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
Step 2.2: 4-aspect Multi-Reward 評分
═══════════════════════════════════════════════════════
   對每個候選 sparse list 評分（注意：是對整個 list 評分，不是單條評論）
   
   R_total = 0.30 × R_timing               (時機合理性，對應 timestamp 集合)
           + 0.25 × R_frequency_match      (list 長度匹配 persona 活躍度)
           + 0.25 × R_content_quality      (每條評論的 persona-aligned 內容品質)
           + 0.20 × R_coverage_diversity   (list 內 timestamp 的覆蓋多樣性)
   
   ⚠ 注意：v2.0 的 6 reward 重新整併為 v4.1 的 4 reward，
          因為輸出單位從「單條評論」變成「sparse list」，需要 list-level reward。

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
Step 2.5: 迭代 DPO（2 輪）
═══════════════════════════════════════════════════════
   重複 Step 2.1-2.4 共 2 輪
   
   依據：Bootstrapping with Implicit Rewards (ICLR 2025)
   
   重要區別：SimLens 不採用 Self-Rewarding LM 的 self-judge 機制
     - actor (Llama-3.2-3B) 與 judge (Qwen3-32B-Q4) 為不同模型
     - 「強者評弱者」設計，避開 3B 自評的 self-bias 與能力不足
```

### 4.3 4 個 Reward 完整定義（v4.1 重新設計）

#### Reward A: Timing（時機合理性）— 30%

```python
def reward_timing(sparse_list, timeline_script):
    """
    對 list 中每個 timestamp 判斷：該時點是否真的有「值得反應的事件」
    
    來源：VTG-LLM (AAAI 2025)、SoccerNet action spotting paradigm
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

#### Reward B: Frequency Match（list 長度匹配 persona 活躍度）— 25%

```python
def reward_frequency_match(sparse_list, persona_yaml, video_duration):
    """
    本研究新增：取代 v2.0 的 None handling，用 list 長度匹配 persona 活躍度
    
    依據：persona expected_comment_count + 真實 YouTube 留言分布觀察
    """
    expected_range = persona_yaml["expected_comment_count_range"]
    # 例：P1 high → (3, 6) per 2min；P5 low → (0, 1) per 2min
    
    # 標準化到實際影片長度
    duration_factor = video_duration / 120  # baseline 2min
    expected_min = expected_range[0] * duration_factor
    expected_max = expected_range[1] * duration_factor
    
    n = len(sparse_list)
    
    if expected_min <= n <= expected_max:
        return 1.0
    elif n < expected_min:
        # 太少
        deficit = expected_min - n
        return max(0, 1 - 0.3 * deficit)
    else:
        # 太多
        excess = n - expected_max
        return max(0, 1 - 0.3 * excess)
```

**為什麼此設計優於 v2.0 的 None handling**：
- v2.0：每段強迫模型決定「該不該留言」，需顯式 None 標籤
- v4.1：「沒反應」 = list 短 / 空，自然編碼進輸出格式
- 不會出現「模型學到濫輸出 None」這種 reward hacking

#### Reward C: Content Quality（每條評論的 persona-aligned 內容品質）— 25%

```python
def reward_content_quality(sparse_list, persona_yaml, timeline_script):
    """
    對 list 中每條評論評分（內容品質 + persona 一致性 + 局部相關性）
    
    來源：PersonaGym (EMNLP 2025)、Score Before You Speak (2025)
    """
    if not sparse_list:
        return 0.5  # 空陣列不適用，給中性分（list 是否該空由 R_timing 與 R_frequency 決定）
    
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

#### Reward D: Coverage Diversity（list 內 timestamp 覆蓋多樣性）— 20%

```python
def reward_coverage_diversity(sparse_list, video_duration):
    """
    本研究新增：避免 list 中所有 timestamp 都擠在同一段
    
    例：P1 對一個 2min 影片留 4 條，但全在 0:10-0:15 這 5 秒內 → 不自然
    
    依據：UMaT temporal alignment + 真實彈幕/留言分布觀察
    """
    if len(sparse_list) <= 1:
        return 1.0  # 0 或 1 條時不適用，給滿分
    
    timestamps = sorted([parse_timestamp(e["timestamp"]) for e in sparse_list])
    
    # Inter-comment gap 的標準差（越大代表分布越均勻）
    gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    expected_gap = video_duration / (len(timestamps) + 1)
    
    # Normalized variance: 越接近 expected gap 分布越均勻
    if not gaps:
        return 1.0
    avg_gap = sum(gaps) / len(gaps)
    deviation = sum(abs(g - expected_gap) for g in gaps) / len(gaps)
    
    return max(0, 1 - deviation / video_duration)
```

### 4.4 LLM-as-Judge 防偏誤策略

```python
def robust_llm_judge(sparse_list, persona, aspect):
    """
    Multi-Judge Ensemble
    依據：Judging the Judges (Krishna et al., 2024)
    """
    judges = [
        ("qwen3:32b-q4_K_M", 0.5),  # 主 judge
        ("gemma2:27b", 0.3),         # 備 judge  
        ("llama3.1:70b-q3", 0.2)    # 仲裁（如硬體允許）
    ]
    
    weighted_score = sum(
        weight * call_judge(model, sparse_list, persona, aspect)
        for model, weight in judges
    )
    return weighted_score


def gpt4_spotcheck():
    """
    從 800 × 4 = 3,200 candidate lists 隨機抽 200 樣本，
    用 GPT-4 重新評分，計算 Spearman ρ
    
    成本：200 × $0.005 ≈ $1 USD
    """
    samples = random.sample(all_evaluations, 200)
    gpt4_scores = [call_gpt4(s) for s in samples]
    local_scores = [s.local_score for s in samples]
    return spearman_correlation(gpt4_scores, local_scores)
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
        "iterative_rounds": 2
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
| **List-level reward (4 aspects)** | **本研究新增** | v2.0 cell-level → v4.1 list-level，配合 sparse 輸出格式 |
| **R_timing (event saliency)** | **VTG-LLM (AAAI 2025)、SoccerNet 評估範式** | 借鏡 action spotting 的時點顯著性概念 |
| **R_frequency_match** | **本研究新增** | 取代 v2.0 None handling，更自然 |
| **R_coverage_diversity** | **本研究新增** | 避免 list 內 timestamp 擠堆 |
| 本地 LLM-as-Judge | "Replacing the Judge" (SambaNova, 2024) | Llama-3.1 70B ≈ GPT-4 Turbo |
| **AI 訊號驅動的 preference data** | **Multi-MLLM Distillation (Gu et al., 2025/05)** | teacher 不一致即作為 preference signal |
| Iterative DPO | Bootstrapping with Implicit Rewards (ICLR 2025) | 多輪迭代提升 alignment |
| Multi-judge ensemble | Judging the Judges (Krishna et al., 2024) | ensemble 比單一 judge 可靠 |

---

## 5. 評估方案（Evaluation Protocol）

### 5.1 Benchmark 對標模型

| Baseline | 角色 | 為何選它 |
|---------|------|---------|
| **SimTube** (Claude+GPT-4, whole-video, 1 comment) | 直接競爭 SOTA | 同類最相關工作 |
| **Claude-3.5 Sonnet zero-shot** (one-shot sparse JSON) | Teacher 本身 | 證明 student 能否超越 |
| **GPT-4o zero-shot** (one-shot sparse JSON) | 大模型強 baseline | 業界最常見 |
| **Llama-3.2-3B zero-shot** (one-shot sparse JSON) | 未訓練起點 | 證明訓練有效性 |
| **Llama-3.2-3B + Phase 1 only (SFT)** | 蒸餾 ablation | 證明 RLAIF 必要 |
| **Llama-3.2-3B + Phase 2 only (DPO from zero-shot)** | RLAIF ablation | 證明蒸餾必要 |
| **SimLens v2.0 (cumulative segment-level)** | **架構 ablation** | **證明 v4.1 one-shot 設計優於 v2.0 cumulative** |
| **Llama-3.2-3B + SimLens v4.1 (full)** | **本研究方法** | 完整 SimLens v4.1 |

### 5.2 評估指標（四層架構）

#### Group 0: Format Compliance（基礎前提）⭐ v4.1 新增

```python
# 因為輸出是結構化 JSON，必須先驗證合規率
# 依據：IFEval (Zhou et al., 2023)、JSONSchemaBench (arXiv 2501.10868)、
#      StructEval (arXiv 2505.20139)

format_metrics = {
    "JPR (JSON Parsing Rate)":
        "輸出能被 json.loads() 無誤解析的比例（預期 >= 95%）",
    "SCR (Schema Compliance Rate)":
        "解析後資料符合 {timestamp, comment} schema 的比例",
    "TFR (Timestamp Format Rate)":
        "timestamp 字串符合 'MM:SS' 格式且落在影片實際長度內的比例",
    "TLR (Timestamp Legality Rate)":
        "timestamp 不重複、按時間順序排列的比例"
}

# 預期效益：
# - Llama-3B zero-shot: ~30-50% FCR
# - SimLens Phase 1 (SFT): ~95%+ FCR
# - 證明蒸餾賦予 3B 模型嚴格的格式控制能力
```

#### Group 1: Tier 1 — Temporal Localization（時序定位指標）⭐ v4.1 新增

```python
# 學生預測的 timestamps 與 teacher (Claude) 的 timestamps 比對
# 依據：SoccerNet action spotting (CVPR 2018)、VTG-LLM (AAAI 2025)

temporal_metrics = {
    "Temporal F1@5s":
        "學生 ts 落在老師 ts ±5s 內算 TP，計算 P/R/F1",
    "Temporal F1@3s":
        "tight 版本（更嚴格）",
    "Average-mAP@[1,3,5,10]s":
        "SoccerNet 標準：多 tolerance 取曲線下面積",
    "T-MAE":
        "matched ts 對的平均秒數誤差",
    "Timing Distribution Similarity":
        "predicted ts 密度分布 vs ground truth 密度分布的 Earth Mover's Distance"
}
```

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

#### Group 3: Tier 3 — List-Level（列表層級指標）⭐ v4.1 新增

```python
list_metrics = {
    "Frequency Match Rate":
        "list 長度落在 persona expected_comment_count 範圍的比例",
    "Coverage Spread":
        "list 內 timestamp 的時間分布廣度（normalized 0-1）",
    "Empty-List Accuracy":
        "對於 teacher 也輸出空陣列的 (video, persona) 對，"
        "student 是否也輸出空陣列（取代 v2.0 None Prediction F1）",
    "Persona Differentiation":
        "8 個 persona 對同一影片的 list 之間的 Jaccard 距離 "
        "（高代表 persona 真的有區分能力，不只是換口頭禪）"
}
```

#### Group 4: Human Evaluation（25 人 Likert）

```python
human_eval = {
    "participants": 25,
    "platform": "Upwork or Prolific",
    "tasks_per_participant": "1 short YouTube video (1–3min) + 8 personas' sparse lists",
    "video_count": 8,
    "rating_scale": "7-point Likert",
    "dimensions": [
        "Timing Naturalness（這些時間點是否合理留言）",
        "Persona Believability（像不像該 persona 會留的）",
        "Helpfulness（對創作者是否有用）"
    ],
    "quality_control": {
        "must_watch_video": True,
        "must_pass_video_quiz": "80% accuracy",
        "must_write_summary": True
    },
    "estimated_cost": "$300-500 USD"
}
```

#### Group 5: External Anchor — Bilibili 彈幕弱對齊（v4.1 新增 sanity check）

```python
# 依據：VideoIC (ACM MM 2020)、Recommending Highlights via DanMaKu (IEEE 2018)
# 重要：這是「群體爆點識別能力」的弱錨點驗證，不是「個人行為對齊」

external_anchor = {
    "purpose": "驗證 SimLens 識別出的高互動時段，是否與真實群體爆點重合",
    "dataset": "VideoIC (ACM MM 2020) 或自爬 Bilibili 短片區彈幕",
    "sample_size": "30-50 部 Bilibili 短影片（中文，1–3min 範圍對齊）",
    "metric": {
        "Peak Overlap Rate": 
            "SimLens 8 personas 合併的 timestamps 集合，"
            "與真實彈幕密度峰值（top-10 segments）的 ±3s 重合率",
        "expected": ">= 60%（弱對齊，非個人行為對齊）"
    },
    "framing": "weak external anchor / sanity check, not ground truth alignment"
}
```

#### Group 6: GPT-4 Spot-check

```python
# 對 200 個隨機 sparse list 樣本，用 GPT-4o 跑同樣 reward 評分
# 計算與本地 judge 的 Spearman ρ
# 預期：ρ > 0.7（強相關，本地 judge 可信）
# 成本：$5 USD
```

### 5.3 Ablation Study 設計（v4.1 重新規劃）

```
必跑的 ablations（共 10 組，缺一不可）：

A1.  SimLens v4.1 (full)                              ← 完整方法
A2.  - w/o Phase 2 (RLAIF) → SFT only                 ← 證明 RLAIF 必要
A3.  - w/o Phase 1 (Distillation) → DPO from zero-shot ← 證明蒸餾必要
A4.  - w/o Multi-LoRA (single LoRA + persona prompt)   ← 證明多 LoRA 必要
A5.  - w/o Multi-aspect Reward (single R_content)      ← 證明 4 reward 必要
A6.  - w/o R_timing                                    ← 證明時序顯著性必要
A7.  - w/o R_frequency_match                           ← 證明活躍度匹配必要
A8.  - w/o R_coverage_diversity                        ← 證明覆蓋多樣性必要
A9.  - w/o Iterative DPO (1 round only)                ← 證明迭代必要
A10. ⭐ v2.0 cumulative segment-level vs v4.1 one-shot ← 證明架構升級的價值
     （這是 v4.1 最重要的 ablation，直接證明「事後反思」設計優於「逐段 cumulative」）
```

### 5.4 預期結果表（你論文的 main result）

#### Table 1: 主結果（Tier 1 + Tier 2 + Tier 3）

```
Method                          | T-F1@5s | Persona | Linguistic | Local Rel. | Coherence | Engaging
─────────────────────────────────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot          | 0.32    | 0.42    | 0.38       | 0.45       | 0.50      | 0.45
Claude-3.5 Sonnet zero-shot     | 0.71    | 0.74    | 0.68       | 0.65       | 0.78      | 0.72
GPT-4o zero-shot                | 0.73    | 0.76    | 0.70       | 0.66       | 0.80      | 0.74
SimTube (whole-video)           | N/A     | 0.78    | 0.72       | N/A        | 0.79      | 0.76
SimLens v2.0 (cumulative)       | N/A     | 0.81    | 0.79       | 0.62       | 0.76      | 0.74
─────────────────────────────────────────────────────────────────────────────────────────────────
SimLens v4.1 Phase 1 only (SFT) | 0.68    | 0.71    | 0.66       | 0.60       | 0.74      | 0.69
SimLens v4.1 Phase 2 only (DPO) | 0.62    | 0.73    | 0.69       | 0.58       | 0.71      | 0.70
SimLens v4.1 Full (SFT + DPO) ⭐| 0.78    | 0.83    | 0.81       | 0.66       | 0.78      | 0.77
                                | > Tea.  | > Tea. | > Tea.     | ≈ Tea.    | ≈ Tea.   | ≈ Tea.
```

**重點論述**：
- **時序定位（Temporal F1@5s）**：SimLens v4.1 > Claude 蒸餾後可超越 teacher（RLAIF 有效）
- **領域指標**：SimLens > Claude（持續驗證 v2.0 結論）
- **通用指標**：SimLens ≈ Claude（蒸餾有效）
- 對比 v2.0：v4.1 在持平領域指標的同時，新增 Tier 1 時序定位能力

#### Table 2: Format Compliance Rate（v4.1 新增）

```
Method                          | JPR    | SCR    | TFR    | TLR    | Composite FCR
──────────────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot          | 42%    | 35%    | 28%    | 22%    | 32%
Claude zero-shot                | 96%    | 92%    | 88%    | 85%    | 90%
SimLens v4.1 Phase 1 (SFT)      | 99%    | 97%    | 95%    | 92%    | 96%
SimLens v4.1 Full + ConDecode   | 100%   | 99%    | 98%    | 96%    | 98%
```

#### Table 3: List-Level 指標（v4.1 新增）

```
Method                          | Freq Match | Coverage | Empty-List Acc | Persona Diff (Jaccard)
─────────────────────────────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot          | 0.31       | 0.42     | 0.40           | 0.18
Claude zero-shot                | 0.68       | 0.71     | 0.65           | 0.42
SimLens v4.1 Full               | 0.84       | 0.78     | 0.79           | 0.61
                                | > Tea.    | > Tea.  | > Tea.        | > Tea.
```

#### Table 4: 效率比較（v4.1 vs v2.0 vs SimTube）

```
Method            | API Calls / video  | Total Cost / 100 videos | VRAM   | Latency / video
──────────────────────────────────────────────────────────────────────────────
SimTube           | 1 (whole-video)    | $94                     | N/A    | ~45s
SimLens v2.0      | 96 cells × 8 pers  | $115                    | 6.5GB  | ~360s (跑 96 cells)
                  | = 96 calls         |                         |        |
SimLens v4.1 ⭐   | 8 (one-shot/pers)  | $12                     | 6.5GB  | ~24s (8 calls)
                  |                    | -90% vs v2.0           |        | -93% vs v2.0
```

#### Table 5: 外部錨點驗證（v4.1 新增 Bilibili 彈幕）

```
Verification                              | SimLens v4.1 | Claude | GPT-4o
──────────────────────────────────────────────────────────────────────
Peak Overlap Rate (vs VideoIC top-10 seg) | 64%          | 68%    | 70%
                                          | (sanity check, weak anchor)
```

### 5.5 評估學術依據

| 評估元素 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| 自動指標（NLG）| SimTube (Hung et al., 2024) Section 6.2 | BERTScore + ROUGE |
| Persona 評估 | PersonaGym (EMNLP 2025) | Persona Consistency + Linguistic Habits |
| Engagingness | PersoBench (Huang et al., 2024) | UniEval-based engagingness |
| Coherence | Score Before You Speak (2025) | coherence dimension |
| 25 人 crowd study | SimTube Section 6.1 | quiz + summary + rating protocol |
| **Format Compliance Rate** | **IFEval (arXiv 2311.07911)、JSONSchemaBench (arXiv 2501.10868)、StructEval (arXiv 2505.20139)** | 結構化輸出評估 |
| **Temporal F1 / Average-mAP** | **SoccerNet (CVPR 2018)、VTG-LLM (AAAI 2025)** | action spotting 標準評估範式 |
| **Timing Distribution Similarity** | **本研究新增** | EMD 量整體分布而非點對點 |
| **Empty-List Accuracy** | **本研究新增** | 取代 v2.0 None Prediction F1 |
| **Persona Differentiation (Jaccard)** | **本研究新增** | 量 8 個 persona 是否真的有區分 |
| **Bilibili 彈幕弱錨點** | **VideoIC (Wang et al., ACM MM 2020)、Recommending Highlights via DanMaKu (Lv et al., IEEE 2018)** | 群體爆點識別的外部驗證 |
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
Week 1: 環境建置 + Persona 設計
  □ 確認 GPU 環境（最少 RTX 3090 24GB）
  □ 安裝 LLaMA-Factory / TRL / Ollama / Outlines（constrained decoding）
  □ Pull Llama-3.2-3B、Qwen3-32B Q4、LLaVA-NeXT
  □ 撰寫 8 個 persona YAML（含 expected_comment_count_range）
  □ 定義 sparse JSON schema（用於 constrained decoding）
  □ 寫好 4 個 reward 函數骨架 + 4 群評估指標骨架
  □ 抓 5-10 部影片做 pipeline sanity check
  ★ Milestone 1：環境就緒、pipeline 跑通

Week 2: 大規模影片素材收集 + Timeline Script Pipeline
  □ 用 YouTube Data API 收集 100 部短 YouTube 影片（嚴格 60–180s）
    篩選條件：medium duration filter + ISO 8601 二次過濾 + 排除 #shorts
  □ 跑 Whisper-Large-v3（含時間戳）
  □ 跑 LLaVA-NeXT 段描述（每 10 秒一段）
  □ UMaT-inspired 時序對齊 → 全局 Timeline Script
  □ 同時爬 30-50 部 Bilibili 短片彈幕（外部錨點用）
  ★ Milestone 2：100 部 YouTube + 30-50 部 Bilibili 就緒

Week 3: Phase 1 蒸餾資料生成
  □ Claude API 對每個 (影片, persona) 生成 sparse JSON list
  □ 100 影片 × 8 persona = 800 sparse lists
  □ 預算花費：~$12 USD（v2.0 是 $115，下降 90%）
  □ 計算 Teacher 的 Format Compliance Rate（驗證 Claude 輸出穩定性）
  ★ Milestone 3：蒸餾訓練資料完成 + Teacher FCR 報告

Week 4: Phase 1 SFT 訓練 + Format Compliance 驗證
  □ 對 8 個 persona 各訓練 1 個 LoRA adapter（含 constrained JSON decoding）
  □ 計算 Phase 1 Student 的 FCR（預期從 zero-shot ~32% 躍升到 96%+）
  □ 跑 baseline benchmark：
    - Llama 3B zero-shot
    - Claude zero-shot
    - GPT-4o zero-shot
  □ 跑 Phase 1 only 結果（SimLens-SFT）
  ★ Milestone 4：Phase 1 完整結果 + FCR 驗證

Week 5: Phase 2 RLAIF (round 1)
  □ 設置 Qwen3-32B 本地 judge
  □ 對每個 (video, persona) 生 4 候選 sparse list → 4-aspect 評分
  □ 構造 800 個 preference pairs
  □ 跑 DPO 訓練（每個 LoRA 獨立 update）
  ★ Milestone 5：RLAIF round 1 結果

Week 6: Phase 2 RLAIF (round 2) + 全套 ablation
  □ Iterative DPO round 2
  □ 跑完整 ablation：A2-A10（共 9 組）
    - 重點：A10（v2.0 vs v4.1 架構對比）
  □ 跑 GPT-4 spot-check（200 樣本驗證 judge）
  □ 跑 Bilibili 彈幕 Peak Overlap 外部錨點驗證
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
| Sparse list 全空（模型懶）| 中 | R_frequency_match 強制匹配 persona 活躍度 |
| 人類評估招募失敗 | 中 | 改用學校系內招募（30 人也夠 LBW） |
| Qwen judge 與 GPT-4 一致性差 | 低 | 用 multi-judge ensemble |
| Bilibili 彈幕爬取受限 | 低 | 改用公開 VideoIC 資料集 |

---

## 8. 預期硬體與成本

### 8.1 硬體需求

```
最低配置：
  - 1× RTX 3090 24GB
  - 64GB RAM、1TB SSD
  - 預估訓練時間：2 週（v2.0 是 3 週，因 sample 數少）

推薦配置：
  - 1× RTX 4090 24GB
  - 128GB RAM、2TB NVMe SSD
  - 預估訓練時間：1.5 週

理想配置：
  - 2× RTX 4090 或 1× A100 40GB
  - 256GB RAM
  - 預估訓練時間：5 天
```

### 8.2 成本估算（v4.1 vs v2.0）

```
雲端 GPU（如果沒有自有硬體）：
  Vast.ai RTX 4090：$0.4/hour
  訓練總時數：~80 hours（v2.0 是 120 hours）
  → $30-60 USD（v2.0 是 $50-180）

API 成本：
  Claude API（蒸餾資料）：$12 USD（v2.0 是 $115，下降 90%）
  GPT-4o（spot-check）：$5 USD
  → 小計：$17 USD

人類評估：
  Upwork crowd-sourcing 25 人：$300-500 USD

Bilibili 彈幕爬取（外部錨點）：
  VideoIC 公開資料集：$0
  自爬 30-50 部：$0（用 bilibili-api）

總成本：
  最低（自有 GPU + 校內招募）：$17 USD
  標準（雲端 GPU + Upwork）：$350-580 USD
  
  v4.1 vs v2.0 總成本：$580 vs $800（下降 ~28%）
  v4.1 vs v2.0 API 成本：$17 vs $120（下降 ~86%）
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
   - VTG-LLM (AAAI 2025) / MMDuet：event-driven sparse temporal prediction
   - UMaT：multimodal temporal alignment
   - PersonaGym：persona evaluation
   - DPO + RLAIF：訓練範式

3. Method (2 頁)
   - 3.1 Architecture: One-shot Timeline → Sparse JSON
   - 3.2 Phase 1: Distillation from Claude (含 constrained JSON decoding)
   - 3.3 Phase 2: 4-aspect Multi-Reward DPO

4. Experiments (3 頁)
   - 4.1 Setup: 100 short YouTube videos (1–3min) × 8 personas (sparse lists)
   - 4.2 Format Compliance: Table 2
   - 4.3 Main results (Tier 1 + Tier 2): Table 1
   - 4.4 List-Level metrics (Tier 3): Table 3
   - 4.5 Ablation: 10 組 configurations
   - 4.6 User study: 25 人 (Timing Naturalness + Believability + Helpfulness)
   - 4.7 External Anchor: Bilibili Peak Overlap (Table 5)
   - 4.8 Efficiency: Table 4 (vs SimTube, vs v2.0)

5. Discussion + Limitations (0.5 頁)
   - Post-hoc framing 的取捨
   - v4.1 vs v2.0 的設計教訓
   - 領域 gap：缺真實時序觀影行為資料

6. Conclusion (0.5 頁)
```

### 9.2 學術 Contribution 重述

```
C1. System Contribution
   First lightweight (3B parameter) one-shot timeline-to-sparse-JSON 
   persona-conditioned video commentary generation system.
   8-persona sparse comment list per video on consumer GPU (24GB),
   with 90% lower API cost than cumulative segment-level v2.0 approach.

C2. Methodological Contribution
   Two-stage training paradigm for ground-truth-scarce + post-hoc setting:
   - One-shot distillation provides foundational sparse-JSON capability
   - 4-aspect list-level multi-reward RLAIF (含 novel R_frequency_match
     and R_coverage_diversity) provides domain breakthrough
   First to integrate UMaT-style temporal alignment + VTG-LLM-style sparse 
   prediction + PersonaGym-style persona evaluation in unified framework.

C3. Empirical Contribution
   First evidence that 3B model can match or exceed 600B teacher (Claude) 
   on temporal localization F1 + persona-specific dimensions.
   Format Compliance Rate (FCR) demonstrates 3B model can rise from 32% 
   zero-shot to 96%+ with distillation.
   Bilibili VideoIC peak-overlap external anchor provides weak sim-to-real 
   alignment evidence absent from prior work.

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

=== 既有引用（沿用自 v2.0）===

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

[20] Judging the Judges (Krishna et al., 2024)
     用於：Multi-judge ensemble 方法

[21] Sentiment Analysis in the Age of Generative AI (Hartmann et al., 2024)
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
     用於：v2.0 None-reaction 學術前例（v4.1 已不需要 None 標籤，
          但保留作為「行為缺失也是訊號」的概念背書）

[27] Thakkar et al. (ACL 2024 Main) — arXiv 2406.04879
     "A Deep Dive into the Trade-Offs of Parameter-Efficient 
      Preference Alignment Techniques"
     用於：LoRA SFT → LoRA DPO 兩階段技術背書

[28] Multi-MLLM Knowledge Distillation (Gu et al., 2025) — arXiv 2505.22517
     用於：完整 LoRA SFT + LoRA DPO 兩階段 prior art

=== v4.1 新增引用 ===

[29] VTG-LLM (Guo et al., AAAI 2025) — arXiv 2405.13382 ⭐ NEW
     "VTG-LLM: Integrating Timestamp Knowledge into Video LLMs 
      for Enhanced Video Temporal Grounding"
     https://arxiv.org/abs/2405.13382
     https://github.com/gyxxyg/VTG-LLM
     用於：Section 1.3 決策 C 背書（one-shot timeline → sparse JSON）
          Section 3.5 Phase 1 學術依據（timestamp 當 token 學的範式）
          Section 4.6 Reward A (R_timing) 設計背書
          Section 5.5 Temporal F1 / mAP 評估範式背書

[30] MMDuet / VideoLLM Knows When to Speak — arXiv 2411.17991 ⭐ NEW
     "VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video 
      Comprehension with Video-Text Duet Interaction Format"
     https://arxiv.org/html/2411.17991
     用於：Section 1.3 決策 C 背書（event-driven sparse prediction）
          Section 3.5 Phase 1 sparse temporal prediction 直接前例

[31] MM-When2Speak / Beyond Words — arXiv 2505.14654 ⭐ NEW
     "Beyond Words: Multimodal LLM Knows When to Speak"
     https://arxiv.org/html/2505.14654v1
     用於：Section 3.5 Phase 1 multimodal "when to speak" 背書

[32] IFEval (Zhou et al., 2023) — arXiv 2311.07911 ⭐ NEW
     "Instruction-Following Evaluation for Large Language Models"
     https://arxiv.org/abs/2311.07911
     用於：Section 5.2 Group 0 Format Compliance 評估方法論
          Verifiable instructions 概念背書

[33] JSONSchemaBench (Geng et al., 2025) — arXiv 2501.10868 ⭐ NEW
     "JSONSchemaBench: A Rigorous Benchmark of Structured Outputs 
      for Language Models"
     https://arxiv.org/abs/2501.10868
     https://github.com/guidance-ai/jsonschemabench
     用於：Section 3.3 constrained JSON decoding 技術背書
          Section 5.2 Group 0 JSON-specific 評估方法

[34] StructEval — arXiv 2505.20139 ⭐ NEW
     "StructEval: Benchmarking LLMs' Capabilities to Generate 
      Structural Outputs"
     https://arxiv.org/html/2505.20139v1
     用於：Section 5.2 Group 0 結構化輸出評估補充

[35] SoccerNet (Giancola et al., CVPR 2018) ⭐ NEW
     "SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos"
     https://arxiv.org/abs/1804.04527
     用於：Section 5.2 Group 1 Temporal F1 / Average-mAP 評估範式
          tolerance window paradigm

[36] VideoIC (Wang et al., ACM MM 2020) ⭐ NEW
     "VideoIC: A Video Interactive Comments Dataset and Multimodal 
      Multitask Learning for Comments Generation"
     https://dl.acm.org/doi/10.1145/3394171.3413890
     https://github.com/AIM3-RUC/VideoIC
     用於：Section 5.2 Group 5 Bilibili 彈幕外部錨點資料集
          Section 5.5 群體爆點識別的弱對齊驗證

[37] Recommending Highlights via DanMaKu (He & Tang, IntelliSys 2017) ⭐ NEW
     "Recommending highlights in Anime movies: Mining the real-time 
      user comments DanMaKu"
     IEEE IntelliSys 2017 (London, UK), IEEE Xplore 2018-03-12
     https://ieeexplore.ieee.org/document/8324311
     用於：Section 5.2 Group 5 彈幕密度峰值識別 highlight 的方法背書

[38] MORLAIF (Williams, 2024) — arXiv 2406.07496
     "Multi-Objective Reinforcement Learning from AI Feedback"
     用於：Section 4.6 multi-aspect reward 學術背書

[39] Bootstrapping with Implicit Rewards (ICLR 2025)
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

L2. No Real-World Persona Behavior Validation
    我們無真實 persona 觀影行為資料集。所有訓練訊號來自合成資料
    (Claude) + LLM-as-Judge (Qwen)，而非真實觀眾行為。
    Bilibili VideoIC 彈幕僅作為「群體爆點識別」的弱外部錨點，
    不對應個人 persona 行為。
    這是領域 gap，不是 SimLens 獨有問題。

L3. Distillation Bias
    Phase 1 用 Claude 當 teacher，可能繼承 Claude 的偏誤
    （例如過度禮貌、避開敏感話題）。
    Phase 2 RLAIF 部分校正，但無法完全消除。

L4. LLM-as-Judge Limitations
    Qwen3-32B 與 GPT-4 一致性 ~ 85-90%。
    緩解：Multi-judge ensemble + GPT-4 spot-check。

L5. English-Only & Cultural Bias
    8 個 persona 都是英文 + 美國/亞洲文化導向。
    中文 / 跨文化擴展為 future work。
    （但 Bilibili 彈幕外部錨點是中文，這是有意的「群體層級驗證」設計，
      與 persona 個人行為層級分離。）

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
    這是 v2.0 reaction_frequency 問題的延續。

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
    或用 VTG-LLM 的 slot-based token compression
    處理 timeline script 超過 4096 max_seq_length 的問題

F7. Real-time Director Mode（移除 post-hoc 限制）
    結合 streaming video LLM (e.g., VideoLLM-online, arXiv 2406.11816)
    從事後反思 → 真實即時觀影反應模擬

F8. Cumulative vs One-shot Hybrid
    保留 v4.1 one-shot 為主流，但對於極長影片切窗 + cumulative reflection
    雙模式架構
```

---

# 附錄：給你的具體 Action Items

## 本週可立即開始

```
□ Day 1: 確認硬體（至少 24GB VRAM）
□ Day 2: 安裝環境（LLaMA-Factory / TRL / Ollama / Outlines）
□ Day 3: Pull Llama-3.2-3B + Qwen3-32B Q4 + LLaVA-NeXT
□ Day 4: 把 8 個 persona YAML 確認下來（含 expected_comment_count_range）
         定義 sparse JSON schema（用於 constrained decoding）
□ Day 5: 用 YouTube API 收集 5-10 部測試影片（驗證 pipeline）
□ Day 6-7: 跑通 Whisper + LLaVA + UMaT-inspired Timeline Script pipeline
            對 1 部影片試跑：Claude 一次餵入 → 收 sparse JSON
            驗證 Format Compliance（手動檢查 8 個 persona 的輸出）
```

## 投稿目標確認

```
首選：ACM MM 2026 BNI（deadline ~ 7 月）— 8 頁短文
備選：UIST 2026 Posters / Demos（deadline ~ 7/10）— 4 頁
保底：智慧創新大賞 2026 + GitHub 開源 + Hugging Face release
```

## 與 v2.0 的差異總結

```
v2.0（cumulative segment-level）→ v4.1（one-shot post-hoc sparse JSON）

砍掉：
✗ Cumulative narrative 設計（每段重餵）
✗ 顯式 None 標籤（改用空陣列隱式編碼）
✗ Cell-level reward（改用 list-level reward）
✗ 6 個 reward（精簡為 4 個 list-level reward）
✗ 「即時模擬」聲稱（改為 post-hoc commentary）

新增：
+ 全局 Timeline Script（一次餵入）
+ Sparse JSON 輸出格式（[{timestamp, comment}, ...]）
+ Constrained JSON decoding（Outlines/XGrammar）
+ Format Compliance Rate（FCR）評估群（Group 0）
+ Temporal F1@5s / Average-mAP 時序定位指標（Tier 1）
+ List-level metrics（Tier 3：Frequency Match / Coverage / Empty-List Acc / Persona Diff）
+ Bilibili VideoIC 彈幕外部錨點（Group 5 sanity check）
+ R_timing / R_frequency_match / R_coverage_diversity 三個新 reward

保留：
✓ UMaT-inspired 時序對齊
✓ Whisper + LLaVA-NeXT 感知層
✓ Llama-3.2-3B + Multi-LoRA per persona
✓ SFT + DPO 兩階段訓練
✓ Qwen3-32B 本地 judge + GPT-4 spot-check
✓ 25 人 Likert human eval
✓ Stage C 報告生成（同一個 Llama-3B base）

調整：
↻ Reward 從 6 個 cell-level → 4 個 list-level
↻ 訓練資料從 ~9,600 cells → 800 sparse lists
↻ Claude API 成本從 $115 → $12（下降 90%）
↻ Per-video API call 從 96 → 8（下降 92%）
↻ Per-video latency 從 ~360s → ~24s（下降 93%）
↻ Ablation 從 8 組 → 10 組（新增 v2.0 vs v4.1 架構對比 A10）
```
