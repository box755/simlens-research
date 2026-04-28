# SimLens 完整研究計畫 v2.0
## "Distillation + RLAIH for Segment-Level Persona-Conditioned Video Audience Simulation"

> 版本：v2.0（無留存曲線版本，聚焦時序反應分析）  
> 投稿目標：UIST 2026 Posters / Demos / ACM MM 2026 BNI / 智慧創新大賞 2026  
> 上一版：v1.0（含留存曲線，已棄用）

---

## 0. 一頁摘要（The One-Page Summary）

```
研究問題：
   能否用 3B 小模型在「缺乏真實 persona 觀影行為資料」的場景下，
   訓練出能對短影音每個時間段生成 persona-specific 反應的多受眾模擬系統？
   （Scope: short-form video, typical 30s–3min — TikTok / Reels / YouTube Shorts）

核心方法：
   Stage A：UMaT-inspired temporal alignment
     - Whisper（含時間戳）+ LLaVA-NeXT（每 10 秒分段）→ 結構化文字
     
   Stage B：兩階段訓練
     - Phase 1（蒸餾）：Claude 生成段層級 persona 反應 → SFT Llama-3B
     - Phase 2（RLAIH）：Qwen3-32B 用 6-aspect reward → DPO

   Stage C：報告生成（同一個 Llama-3B base）
     - 從評論矩陣整合出跨受眾比較與改善建議

關鍵差異化（vs SimTube）：
   SimTube：影片 → 整體理解 → 一條評論
   SimLens：影片 → N 個 10 秒段 → 8 persona × N 段 = 8N 個段層級反應
            （avg ~12 段 / 96 cells per video，range 3–18 段視影片長度而定）
   
   SimLens 提供「段層級 persona 反應分析」，SimTube 只能給整片評論。

預期貢獻：
   C1. System：首個 segment-level persona-conditioned 影片觀眾模擬器
   C2. Method：缺資料場景下蒸餾 + RLAIH + 6-aspect reward 訓練範式
   C3. Empirical：3B student 在段層級評論生成接近 Claude，且 on-device

明確不做：
   ✗ 留存曲線預測（避開資料來源質疑）
   ✗ 跳出率預測
   ✗ 任何需要對應「真實觀眾行為」的指標
```

---

## 1. 系統架構（Architecture Overview）

```
┌──────────────────────────────────────────────────────┐
│             輸入：短影音 (30s–3min, e.g., TikTok/Reels) │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage A：UMaT-inspired 時序對齊 Pipeline                │
│                                                       │
│   Whisper-Large-v3                                    │
│     → 帶時間戳的整段轉錄                                 │
│     → [(t=0.5s, "Hi"), (t=2.1s, "today..."), ...]    │
│                                                       │
│   LLaVA-NeXT-13B                                      │
│     → 每 10 秒抽 4 frames + audio context              │
│     → N 個段描述（每段 ~150 字視覺敘述）                  │
│       N = ⌈video_duration / 10s⌉, typically 3–18      │
│                                                       │
│   時間軸對齊（UMaT 風格）                                │
│     → 把 Whisper 文字插入對應 LLaVA 段                    │
│     → N 個 enriched segments                           │
│       每段 = {t_start, t_end, visual_desc, transcript}  │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage B：Segment-Level Persona 評論生成                  │
│                                                       │
│   Llama-3.2-3B-Instruct（4-bit GPTQ 量化）              │
│   + 8 個 LoRA Adapter（每個 persona 一個）              │
│                                                       │
│   對每個 (segment_i, persona_p)：                       │
│     Input: 累積敘述至段 i + 段 i 內容 + persona_p 描述    │
│     Output: 該 persona 在該時段的反應                    │
│             - 可能是評論文字                            │
│             - 可能是 None（沒反應）                      │
│                                                       │
│   產出：N 段 × 8 persona = 8N 個 cell 的反應矩陣          │
│        (avg ~96 cells / video)                         │
└──────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────┐
│ Stage C：報告整合（同一個 Llama-3.2-3B，無 adapter）      │
│                                                       │
│   Input: 8N cell 反應矩陣 + 影片敘述                      │
│   Output: 結構化報告                                    │
│     - 跨受眾比較（哪段 P1 反應強、哪段 P5 冷淡）           │
│     - Persona 共鳴度分析（每個 persona 最有共鳴的段）      │
│     - 改善建議（所有 persona 都冷淡的段）                  │
└──────────────────────────────────────────────────────┘
```

### 1.1 架構設計依據

| 設計元素 | 來源文獻 | 為什麼這樣選 |
|---------|---------|-------------|
| Whisper-Large-v3 + LLaVA-NeXT | SimTube (Hung et al., arXiv 2411.09577) | 直接借鏡 SimTube 的 multimodal pipeline |
| **時序對齊 + 結構化文字** | **UMaT (Bi & Xu, arXiv 2503.09081)** | 將視覺與聽覺輸入統一為結構化文字 |
| 10 秒等長分段 | UMaT structured segmentation | 固定段長度避免 fragmentation |
| Llama-3.2-3B 為 student | Meta Llama 3.2 release notes (2024) | 1B 太弱、8B 太大，3B 是甜蜜點 |
| 4-bit GPTQ 量化 | LLaMA-Factory 官方推薦工作流 | 在 24GB VRAM 跑得動 |
| Multi-LoRA per persona | **Neeko (EMNLP 2024)**、LoRA-MoE (2024) | 已驗證 per-character LoRA 優於 single LoRA + prompt |
| Same model for generation + report | Tülu 3 multi-task post-training | 一模多用節省部署成本 |

### 1.2 兩個關鍵設計決策的學術背書

本架構有兩個可能被 reviewer 質疑的設計選擇，以下提供學術背書：

#### 決策 A：為什麼分離「感知（Whisper + LLaVA-NeXT）」與「推理（Llama Agent）」，不直接用原生多模態模型一次到位？

**潛在質疑**：reviewer 可能會問「為什麼不把短影音直接餵給 Gemini 2.5 Flash 或 GPT-4o，讓它一次給出 persona 評論？」

**學術背書 1：感知與推理分離有可解釋性與可除錯性優勢**
- **VideoMultiAgents (Kugo et al., arXiv 2504.20091)** 證明：將感知任務交給專門的代理人處理並產出獨立文字報告，能讓推理 agent 在透明基礎上運作，避免單一巨型模型的黑箱干擾與錯誤傳播。在 Intent-QA 上達到 79.0% (+6.2% over previous SOTA)。
- **UMaT (Bi & Xu, arXiv 2503.09081)** 主張將視覺與聽覺降維為「統一文字表示」，提供 interpretability 與 structured retrieval 能力。

**學術背書 2：原生多模態模型的時序理解仍有缺陷**
- **VBenchComp / Time Blindness 系列研究**：實驗證實當前頂尖原生多模態模型（GPT-4o、Gemini 系列）對影片存在「時間盲區」與「shuffling invariance」——即使打亂影片影格順序，模型輸出仍幾乎不變，顯示其依賴靜態畫面與 language priors 而非真實時序推理。將影片轉為帶時間戳的結構化文字，是強迫 agent 真正理解時序的有效手段。

**對 SimLens 的意義**：SimLens 模擬「persona 在影片每段的具體反應」，反應必須精確對應該段內容。若用原生模型端到端處理，無法判斷錯誤反應源自視覺誤判、語音誤聽、還是推理錯誤；分離設計讓我們能逐層除錯，這對研究嚴謹性與系統迭代不可或缺。

#### 決策 B：為什麼要將影片分段（每 10 秒一段），不直接整段處理？

**潛在質疑**：reviewer 可能會問「短影音直接整段餵給 LLaVA-NeXT 不就好了？」

**學術背書 1：分段是時間軸對齊的最穩健方式**
- **UMaT (Bi & Xu, arXiv 2503.09081)** 明確指出，要在影片任務中維持語義與時間一致性，必須將視覺描述與 ASR 轉錄「依時間戳切分為結構化片段（structured segments based on their timestamps）」。將短片段整合能確保視覺與聽覺在時間與語義上的絕對對齊。
- 對 SimLens 而言，若不分段，LLaVA 的整片視覺描述與 Whisper 的時間戳轉錄將難以在程式端完美對齊；分段建立「絕對時間錨點」，讓 agent 能精確知道「這段畫面對應這段聲音」。

**學術背書 2：分段能規避視覺模型的記憶體與品質下降問題**
- **QMAVIS (Lin et al., arXiv 2601.06573)** 證明：採用 chunking + late fusion 策略（將影片切成短片段、各模態獨立處理、最後 LLM 整合），在 VideoMME 長影片基準上比端到端原生多模態模型（VideoLlaMA2、InternVL2）**準確率高出 38.75%**。論文指出原生模型為了塞進 context window 必須採用暴力 down-sampling，導致關鍵細節遺失。

**對 SimLens 的意義**：即使 SimLens 處理的是短影音，分段設計仍提供三個關鍵價值：(1) 時間軸錨點讓 agent 反應與當下影片內容精確對應；(2) 每段獨立餵給 LLaVA 能維持最高品質的視覺描述；(3) N 個段落產生「N 段 × 8 persona = 8N cells」的反應矩陣（avg ~96 cells），是 SimLens 段層級分析能力的基礎，無法以整片處理達成。

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
- 8 個是甜蜜點：每 persona 約 1500 筆訓練資料（含 None 反應）
```

### 2.2 8 個 SimLens Persona Schema

採用 PersonaGym 標準六層描述（Demographics + Interests + Personality + Viewing Habits + Linguistic Style + Engagement Pattern）：

#### **P1: 18-24 大學女性（社群活躍型）**
```yaml
demographics: {age: 18-24, gender: female, occupation: university student, location: urban Asia}
interests: [K-beauty, fashion trends, travel vlogs, K-pop, lifestyle]
personality: extroverted, trend-conscious, peer-influenced
viewing_habits: {length: 5-15min, platforms: [Instagram, TikTok, YouTube Shorts], engagement: heavy}
linguistic_style: {tone: enthusiastic, phrases: [OMG, love this, so cute, obsessed], emoji: very high}
reaction_frequency: high (對 60-80% 的段都會留言)
```

#### **P2: 25-34 上班族男性（科技分析型）**
```yaml
demographics: {age: 25-34, gender: male, occupation: tech professional, location: urban}
interests: [tech reviews, gadgets, finance, productivity tools]
personality: analytical, skeptical, data-driven
viewing_habits: {length: 10-20min, platforms: [YouTube, Twitter/X], engagement: rare but substantive}
linguistic_style: {tone: measured, phrases: [actually, IMO, the real question is], emoji: minimal}
reaction_frequency: low (對 20-30% 的段才會留言)
```

#### **P3: 25-34 上班族女性（職涯導向型）**
```yaml
demographics: {age: 25-34, gender: female, occupation: marketing/consulting, location: urban}
interests: [career development, work-life balance, finance, premium travel]
personality: goal-oriented, aesthetically aware, time-conscious
viewing_habits: {length: 8-15min, platforms: [YouTube, LinkedIn, Instagram], engagement: moderate}
linguistic_style: {tone: polished, phrases: [great insight, takeaway, totally relatable], emoji: low-moderate}
reaction_frequency: medium (對 40-50% 的段會留言)
```

#### **P4: 35-44 已婚父母（家庭實用型）**
```yaml
demographics: {age: 35-44, gender: any, occupation: parent + employed, location: suburban}
interests: [parenting, family travel, home improvement, finance, wellness]
personality: practical, value-focused, time-constrained
viewing_habits: {length: 5-10min, platforms: [YouTube, Facebook], engagement: experience-sharing}
linguistic_style: {tone: warm, phrases: [as a parent, my kids, this reminds me], emoji: moderate}
reaction_frequency: medium (對 40-50% 的段會留言)
```

#### **P5: 45-54 中年男性（傳統權威型）**
```yaml
demographics: {age: 45-54, gender: male, occupation: established professional, location: any}
interests: [news, investment, traditional hobbies, documentaries]
personality: opinionated, traditional, authority-respecting
viewing_habits: {length: 15-30min, platforms: [YouTube, traditional TV], engagement: occasional opinion}
linguistic_style: {tone: authoritative, phrases: [back in my day, the real issue is, frankly], emoji: very low}
reaction_frequency: low (對 15-25% 的段才會留言)
```

#### **P6: 18-24 大學男性（遊戲動漫宅）**
```yaml
demographics: {age: 18-24, gender: male, occupation: student/entry-level, location: any}
interests: [gaming, esports, anime, meme culture]
personality: playful, ironic, peer-influenced
viewing_habits: {length: 5-30min variable, platforms: [YouTube, Twitch, Reddit], engagement: meme-heavy}
linguistic_style: {tone: ironic, phrases: [based, W, L, no cap, fr, this slaps], emoji: moderate ironic}
reaction_frequency: high (對 50-70% 的段會留言，常用迷因)
```

#### **P7: 55+ 退休族群（懷舊溫暖型）**
```yaml
demographics: {age: 55+, gender: any, occupation: retired/semi-retired, location: any}
interests: [health, leisurely travel, traditional cooking, philosophy/religion]
personality: reflective, nostalgic, warmth-valuing
viewing_habits: {length: 10-30min, platforms: [YouTube, Facebook], engagement: warm personal}
linguistic_style: {tone: warm, phrases: [thank you for sharing, brings back memories, blessed], emoji: low-moderate}
reaction_frequency: medium-low (對 30-40% 的段會留言)
```

#### **P8: 13-17 青少年（潮流迷因型）**
```yaml
demographics: {age: 13-17, gender: any, occupation: middle/high school student, location: any}
interests: [viral content, memes, gaming, music/dance, school life]
personality: peer-conscious, trend-driven, expressive
viewing_habits: {length: 1-5min very short, platforms: [TikTok, Instagram, Shorts], engagement: very high short}
linguistic_style: {tone: high energy, phrases: [LMAO, fr fr, no way, that's so me], emoji: very high}
reaction_frequency: very high (對 70-90% 的段會留言，但短)
```

### 2.3 Persona 設計學術依據

| 設計元素 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| 六層 schema 結構 | PersonaGym (Samuel et al., EMNLP 2025) | demographics + linguistic habits + behavior |
| Demographics 細節 | PersonaChat (Zhang et al., ACL 2018) | occupation + location + interests |
| Linguistic style 設計 | Bias-Adjusted LLM Agents (Kitadai et al., arXiv 2508.18600) | individual-level 行為差異化 |
| Viewing habits 加入 | SimTube (Hung et al., 2024) | 影片觀眾模擬特有元素 |
| **Reaction frequency** | 本研究新增 | 為時序設計獨有，控制 None 反應比例 |

---

## 3. Phase 1：蒸餾（Knowledge Distillation）

### 3.1 目標

讓 Llama-3.2-3B 繼承 Claude-3.5 Sonnet 的「**段層級 persona 反應生成能力**」。

### 3.2 影片資料準備

```
═══════════════════════════════════════════════════════
Step 1.1: 影片素材收集
═══════════════════════════════════════════════════════
   注意：這只是「素材」，不是 ground truth
   
   來源：YouTube Data API v3
   數量：100 部短影音 (30s–3min, avg ~2min)
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

═══════════════════════════════════════════════════════
Step 1.2: UMaT-inspired 時序對齊 Pipeline
═══════════════════════════════════════════════════════
   對每部影片：
   
   (a) Whisper-Large-v3 整段轉錄（含時間戳）
       輸出：[(0.5s, "Hi everyone"), (2.1s, "today..."), ...]
   
   (b) LLaVA-NeXT 段描述（每 10 秒一段，共 N = ⌈duration/10s⌉ 段，typical 3–18）
       對每段：抽 4 frames（在段內 t=0%, 33%, 66%, 100%）
              拼接成 panel image
              連同段內 transcript 餵給 LLaVA-NeXT
              生成段視覺描述（~150 字）
   
   (c) 時序對齊（UMaT 風格的 structured text）
       對每段 segment_i：
         {
           "t_start": i * 10,
           "t_end": (i+1) * 10,
           "visual": "<LLaVA 段描述>",
           "transcript": "<該段時間範圍內的 Whisper 文字>"
         }
   
   (d) 累積敘述生成
       cumulative_narrative(i) = segment_1 + segment_2 + ... + segment_i
       
   產出：100 個影片 × 12 個 enriched segments

═══════════════════════════════════════════════════════
Step 1.3: Claude 蒸餾資料生成（核心）
═══════════════════════════════════════════════════════
   
   對每個 (影片 V, 段 i, persona P)：
     讓 Claude 模擬該 persona 在該時段的反應
     
   Prompt template:
   ┌─────────────────────────────────────────────────┐
   │ You are simulating a YouTube viewer with         │
   │ this persona:                                    │
   │ {persona_yaml}                                   │
   │                                                  │
   │ You're watching a video. Below is what you've    │
   │ seen so far (segments 1 to i):                   │
   │                                                  │
   │ {cumulative_narrative}                           │
   │                                                  │
   │ Current segment (t={t_start}-{t_end}):           │
   │ Visual: {segment_i.visual}                       │
   │ Audio: {segment_i.transcript}                    │
   │                                                  │
   │ Based on your persona's:                         │
   │ - Reaction frequency: {persona.reaction_freq}    │
   │ - Linguistic style: {persona.linguistic}         │
   │                                                  │
   │ What would you say at this moment?               │
   │ - If you'd react: write a comment (10-50 words)  │
   │ - If you wouldn't react: output exactly "None"   │
   │                                                  │
   │ Important: Many segments don't trigger reactions.│
   │ Be selective. Match your persona's frequency.    │
   └─────────────────────────────────────────────────┘
   
   產出量：
     100 影片 × avg 12 段 × 8 persona ≈ 9,600 個 cells
     （actual cell count 隨影片長度浮動，範圍 ~2,400–14,400）
     其中約 50% 是 None（無反應）
     實際生成評論：~4,800 筆
     
   為什麼包含 None：
     "無反應" 也是訓練訊號 → 教模型「該 persona 此刻不會留言」
     這是 SimTube 沒有的設計（SimTube 強迫每個 persona 都對整片評論）
   
   成本估算：
     ~9,600 calls × ~$0.012/call ≈ $115 USD（estimated, 含 ±30% buffer）
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
        "epochs": 2,
        "batch_size": 4,
        "gradient_accumulation": 4,
        "learning_rate": 2e-4,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "lr_scheduler": "cosine",
        "max_seq_length": 2048
    },
    
    "data": {
        "samples_per_persona": ~1200,  # avg, 含 None
        "total_samples": ~9600,        # avg across 100 videos w/ varying length
        "split": {"train": 0.85, "val": 0.10, "test": 0.05}
    }
}
```

### 3.4 Per-Persona LoRA Adapter 訓練策略

```python
# 8 個獨立 LoRA Adapter
for persona_id in [P1, P2, ..., P8]:
    # 該 persona 的所有 (影片段, persona, 反應) 樣本
    persona_data = data[persona_id]  # 約 1200 筆，含 None
    
    # 訓練該 persona 的 LoRA
    lora_p = train_lora(
        base_model=Llama-3.2-3B,
        train_data=persona_data,
        lora_rank=8,
        epochs=2
    )
    save_adapter(lora_p, f"./adapters/{persona_id}")

# 推理時
def generate_segment_reaction(video, segment_i, persona_id):
    """對影片第 i 段生成 persona_id 的反應"""
    base = load(Llama-3.2-3B)
    adapter = load(f"./adapters/{persona_id}")
    model = merge(base, adapter)
    
    prompt = format_segment_prompt(
        cumulative_narrative=video.narrative_until(i),
        current_segment=video.segments[i],
        persona=PERSONAS[persona_id]
    )
    
    output = model.generate(prompt)
    
    # 處理 None 輸出
    if output.strip().lower() == "none":
        return None
    return output
```

### 3.5 Phase 1 學術依據

| 設計選擇 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| Knowledge distillation 為起點 | DistilBERT (Sanh et al., 2019)、Tülu 3 (Lambert et al., 2024) | SFT on synthetic data |
| Claude as teacher | OpenCharacter (arXiv 2501.15427) | 大模型蒸餾 role-playing 行為 |
| Synthetic persona data | PersonaLLM (Jiang et al., 2024) | 合成 persona dialog 訓練 |
| **時序對齊 narrative** | **UMaT (Bi & Xu, arXiv 2503.09081)** | structured text representation |
| LoRA per persona | **Neeko (EMNLP 2024)**、LoRA-MoE (2024) | per-character LoRA 已被證明優於 single LoRA + prompt |
| 4-bit GPTQ + LoRA rank 8 | LLaMA-Factory 官方文件、**Thakkar et al. (ACL 2024)** | 標準 PEFT 工作流；ACL 2024 Main 提供 adapter rank sensitivity 指引 |
| **LoRA SFT 後續可接 DPO** | **Thakkar et al. (ACL 2024 Main, arXiv 2406.04879)** | 300+ 實驗證明 LoRA-SFT → LoRA-DPO 範式可行；SimLens 兩階段訓練的權威背書 |
| **兩階段 (SFT + DPO) on LoRA** | **Multi-MLLM Distillation (Gu et al., arXiv 2505.22517, 2025/05)** | 直接前例：LoRA SFT + 同 LoRA DPO + AI 訊號當 preference，在 ~7B 模型 + 多教師蒸餾場景達 SOTA |
| **包含 None 反應的訓練** | **Action-Guided Engagement (arXiv 2502.12073)** | 平台層級「ignore」action 已有先例，SimLens 延伸至 persona-internal 層級 |

---

## 4. Phase 2：RLAIH（Reinforcement Learning from AI Feedback）

### 4.1 目標

讓 Phase 1 蒸餾後的 Llama-3.2-3B 在**段層級領域指標**上接近或超越 Claude（teacher）。

> 學術背書：<br>
> *"In some settings, e.g., harmless dialogue generation, RLAIF even surpasses RLHF due to more consistent label definition."* (RLAIF survey, Lee et al., 2023)

### 4.2 完整 DPO 訓練流程

```
═══════════════════════════════════════════════════════
Step 2.1: 候選反應生成
═══════════════════════════════════════════════════════
   對每個 (影片段 i, persona P):
     用 Phase 1 蒸餾後的 Llama-3B + LoRA_P 生成 N=4 個候選
     - temperature=0.9, top_p=0.95
   
   注意：候選包含「None」這個選項
   
   產出：~9600 prompts × 4 candidates = 38,400 candidates
        其中 ~50% 候選是 None

═══════════════════════════════════════════════════════
Step 2.2: 6-aspect Multi-Reward 評分
═══════════════════════════════════════════════════════
   對每個非 None 候選評論，計算 6 個 reward 分數：
   
   R_total = 0.25 × R_relevance        (BERTScore + ROUGE-1)
           + 0.20 × R_persona_cons     (PersonaGym, Qwen3-32B judge)
           + 0.20 × R_linguistic       (PersonaGym, Qwen3-32B judge)
           + 0.15 × R_segment_relevance (本研究新增)
           + 0.10 × R_coherence        (Score Before You Speak)
           + 0.10 × R_engagingness     (UniEval)
   
   對 None 候選的特殊處理：
   - 如果該 persona 的 reaction_frequency 低，且其他候選 R < 0.5
     → None 得高分（說明這段該 persona 不該留言）
   - 如果其他候選有高品質回應（R > 0.7）
     → None 得低分（說明該段該 persona 應該留言）

═══════════════════════════════════════════════════════
Step 2.3: Preference Pair 構造
═══════════════════════════════════════════════════════
   對每個 (segment, persona)：
     取 4 候選中 R 最高 → chosen
     取 4 候選中 R 最低 → rejected
   
   產出：~9600 個 (segment, persona, chosen, rejected) preference pairs

═══════════════════════════════════════════════════════
Step 2.4: DPO Update
═══════════════════════════════════════════════════════
   用 trl library 的 DPOTrainer
   每個 LoRA adapter 獨立訓練

═══════════════════════════════════════════════════════
Step 2.5: 迭代 DPO（2 輪）
═══════════════════════════════════════════════════════
   重複 Step 2.1-2.4 共 2 輪
   
   依據：Self-Rewarding LM (Yuan et al., 2024) 的 Iterative DPO 部分
        Bootstrapping with Implicit Rewards (ICLR 2025)
   
   重要區別：SimLens 不採用 Self-Rewarding LM 的 self-judge 機制
     - Self-Rewarding LM: 同一個 70B 模型既當 actor 又當 judge
     - SimLens: actor (Llama-3.2-3B) 與 judge (Qwen3-32B-Q4) 為不同模型
   
   理由：
     (1) Llama-3B 太小，同時做兩件事會兩邊都做不好
         （Self-Rewarding 用 70B 才能可靠自評）
     (2) 自評有 self-bias，偏好自己風格的答案
     (3) Qwen3-32B > Llama-3B，符合「強者評弱者」的 judge 設計準則
     (4) 6-aspect 結構化 reward 需要 external judge + PersonaGym rubric 才好做
   
   僅借鏡的概念：
     Iterative DPO（M1 → M2 → M3 多輪 DPO 可持續改善 alignment）
     —— 此結論不依賴 self-rewarding 機制也成立
```

### 4.3 6 個 Reward 完整定義

#### Reward A: Relevance（與該段相關性）— 25%

```python
def reward_relevance(comment, segment_i):
    """
    來源：SimTube (Hung et al., 2024) Section 6.2.2
    注意：reference 是該段內容，不是整片
    """
    if comment is None:
        return 0.5  # 中性，避免 bias
    
    segment_text = segment_i.visual + " " + segment_i.transcript
    
    bertscore_f1 = bert_score(
        candidates=[comment],
        references=[segment_text],
        model_type='microsoft/deberta-xlarge-mnli'
    ).f1
    
    rouge1_f1 = rouge_scorer.RougeScorer(
        ['rouge1'], use_stemmer=True
    ).score(segment_text, comment)['rouge1'].fmeasure
    
    return 0.5 * bertscore_f1 + 0.5 * rouge1_f1
```

**為什麼 25% 權重**：相關性是基本盤，但因為現在分到 6 個 reward，權重從 30% 降到 25%。

#### Reward B: Persona Consistency（一致性）— 20%

```python
def reward_persona_consistency(comment, persona_yaml):
    """
    來源：PersonaGym (Samuel et al., EMNLP 2025 Findings)
    用本地 Qwen3-32B Q4
    """
    if comment is None:
        # None 反應的 persona consistency 由 reaction_frequency 決定
        return persona.reaction_frequency_consistency_score
    
    prompt = f"""
You are evaluating a YouTube comment for persona consistency.

Persona description:
{persona_yaml}

Generated comment:
"{comment}"

Rate 1-5 based on PersonaGym rubric:
1: Directly contradicts persona's background or beliefs
2: Mostly inconsistent with persona
3: Neutral / unrelated to persona traits
4: Mostly consistent with persona
5: Strongly reflects persona's background, beliefs, and personality

Output ONLY the integer score.
"""
    score = ollama_call("qwen3:32b-q4_K_M", prompt)
    return int(score) / 5.0
```

#### Reward C: Linguistic Habits（語言習慣）— 20%

```python
def reward_linguistic_habits(comment, persona_yaml):
    """來源：PersonaGym Linguistic Habits dimension"""
    if comment is None:
        return 0.5  # 中性
    
    prompt = f"""
You are evaluating word choice and tone alignment.

Expected linguistic style:
{persona_yaml['linguistic_style']}

Generated comment:
"{comment}"

Rate 1-5:
1: Tone and word choice strongly mismatch persona
2: Some mismatches in style
3: Style is generic, neither matches nor contradicts
4: Style matches persona's expected pattern
5: Tone, vocabulary, and emoji usage perfectly match persona

Output ONLY the integer score.
"""
    score = ollama_call("qwen3:32b-q4_K_M", prompt)
    return int(score) / 5.0
```

#### Reward D: Segment Relevance（段內相關性）— 15% ⭐ 新增

```python
def reward_segment_relevance(comment, all_segments, current_idx):
    """
    本研究新增：判斷評論是否「精準對應該段」而非「其他段」
    
    避免時序錯位：例如在開頭段評論「結尾很棒」這種錯誤
    
    依據：UMaT (Bi & Xu, 2025) 的時序對齊概念
    """
    if comment is None:
        return 0.5  # 中性
    
    current_seg = all_segments[current_idx]
    other_segs = [s for i, s in enumerate(all_segments) if i != current_idx]
    
    # 評論與當段的相似度
    sim_current = bert_score(
        [comment],
        [current_seg.visual + " " + current_seg.transcript],
        model_type='microsoft/deberta-xlarge-mnli'
    ).f1
    
    # 評論與其他段的平均相似度
    other_text = " ".join([s.visual + " " + s.transcript for s in other_segs])
    sim_other = bert_score([comment], [other_text]).f1
    
    # 正規化到 [0, 1]
    raw_score = sim_current - 0.7 * sim_other
    return max(0, min(1, raw_score + 0.5))  # shift to [0, 1]
```

**為什麼這是新貢獻**：
- SimTube 不分段，沒有此維度
- UMaT 的 temporal alignment 是 retrieval 而非 generation
- 本研究首次將「時序對齊」作為 generation reward 訊號

#### Reward E: Coherence（連貫性）— 10%

```python
def reward_coherence(comment, cumulative_narrative):
    """
    來源：Score Before You Speak (Saggar et al., 2025)
    """
    if comment is None:
        return 0.5
    
    prompt = f"""
You are evaluating whether a comment is a coherent response to a video.

Video content (up to this segment):
{cumulative_narrative[:1500]}

Generated comment:
"{comment}"

Rate 1-5:
1: Comment doesn't make sense given video content
2: Comment is loosely related but incoherent
3: Comment is acceptable but generic
4: Comment is a sensible, specific response
5: Comment shows clear understanding and meaningful response

Output ONLY the integer score.
"""
    score = ollama_call("qwen3:32b-q4_K_M", prompt)
    return int(score) / 5.0
```

#### Reward F: Engagingness（投入度）— 10%

```python
def reward_engagingness(comment, segment_text):
    """來源：PersoBench (Huang et al., 2024) + UniEval"""
    if comment is None:
        return 0.5
    
    from unieval.scorer import UniEvaluator
    evaluator = UniEvaluator(task='dialogue')
    score = evaluator.evaluate(
        output=comment,
        src=segment_text,
        aspect='engagingness'
    )
    return score
```

### 4.4 LLM-as-Judge 防偏誤策略

```python
def robust_llm_judge(comment, persona, aspect):
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
        weight * call_judge(model, comment, persona, aspect)
        for model, weight in judges
    )
    return weighted_score


def gpt4_spotcheck():
    """
    從 9600 個 evaluations 隨機抽 200 樣本，
    用 GPT-4 重新評分，計算 Spearman ρ
    
    成本：200 × $0.005 ≈ $1 USD
    
    報告：「本地 judges 與 GPT-4 的相關性 ρ = 0.XX」
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
        "batch_size": 2,
        "gradient_accumulation": 8,
        "learning_rate": 5e-7,
        "beta": 0.1,
        "max_length": 2048,
        "max_prompt_length": 1500
    },
    
    "preference_data": {
        "total_pairs": ~9600,
        "iterative_rounds": 2
    }
}
```

### 4.6 Phase 2 學術依據

| 設計選擇 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| RLAIH 用於 alignment | RLAIF (Lee et al., 2023, Google DeepMind) | AI feedback 達 RLHF 同等效果 |
| **不訓練 reward model（用 frozen Qwen-32B）** | **RLAIF (Lee et al., 2023) d-RLAIF** | 證明 frozen LLM 直接給 reward 比訓練 RM 更穩，避開 RM staleness |
| DPO 取代 PPO | DPO (Rafailov et al., NeurIPS 2023) | 小資料更穩定，適合 LoRA |
| **DPO on LoRA adapter（同 LoRA 繼續訓練）** | **Thakkar et al. (ACL 2024 Main)、Multi-MLLM Distillation (arXiv 2505.22517)** | 兩篇皆系統證明「LoRA SFT → 同 LoRA DPO」可行，是 SimLens Phase 2 的權威背書 |
| Multi-aspect reward | MORLAIF (Williams, arXiv 2406.07496) | 多目標 reward 比單一更穩 |
| 5 個 reward dimension | SimTube + PersonaGym + PersoBench + Score Before You Speak | 各有頂會背書 |
| **Segment Relevance reward** | 本研究新增（受 UMaT 啟發） | 時序對齊作為 generation 訊號 |
| 本地 LLM-as-Judge | "Replacing the Judge" (SambaNova, 2024) | Llama-3.1 70B ≈ GPT-4 Turbo |
| **AI 訊號驅動的 preference data** | **Multi-MLLM Distillation (Gu et al., 2025/05)** | 直接前例：teacher 不一致即作為 preference signal |
| Iterative DPO | Bootstrapping with Implicit Rewards (ICLR 2025) | 多輪迭代提升 alignment |
| Multi-judge ensemble | Judging the Judges (Krishna et al., 2024) | ensemble 比單一 judge 可靠 |

---

## 5. 評估方案（Evaluation Protocol）

### 5.1 Benchmark 對標模型

| Baseline | 角色 | 為何選它 |
|---------|------|---------|
| **SimTube** (Claude+GPT-4, whole-video) | 直接競爭 SOTA | 同類最相關工作 |
| **Claude-3.5 Sonnet zero-shot** (segment-level) | Teacher 本身 | 證明 student 能否超越 |
| **GPT-4o zero-shot** (segment-level) | 大模型強 baseline | 業界最常見 |
| **Llama-3.2-3B zero-shot** (segment-level) | 未訓練起點 | 證明訓練有效性 |
| **Llama-3.2-3B + Phase 1 only (SFT)** | 蒸餾 ablation | 證明 RLAIH 必要 |
| **Llama-3.2-3B + Phase 2 only (DPO)** | RLAIH ablation | 證明蒸餾必要 |
| **Llama-3.2-3B + SimLens (full)** | **本研究方法** | 完整 SimLens |

### 5.2 評估指標分為三組

#### Group 1: 自動指標（Layer 2 — 機械化評估）

```python
auto_metrics = {
    # SimTube 風格的 NLG 指標（adapted to segment-level）
    "BERTScore_F1":      "vs current segment text",
    "ROUGE-1_F1":        "vs current segment text",
    "Distinct-1":        "lexical diversity (unigram)",
    "Distinct-2":        "lexical diversity (bigram)",
    
    # Persona-specific 指標（用本地 Qwen3-32B）
    "Persona_Consistency": "PersonaGym rubric, scaled 0-1",
    "Linguistic_Habits":   "PersonaGym rubric, scaled 0-1",
    "Coherence":           "Score Before You Speak rubric",
    "Engagingness":        "UniEval engagingness",
    
    # 本研究新增指標
    "Segment_Relevance":   "對應該段 vs 其他段的相對相似度",
    "None_Accuracy":       "無反應預測的準確率（vs Claude teacher）",
    "Reaction_Frequency_Match": "生成的反應頻率是否符合 persona 預設",
}
```

#### Group 2: GPT-4 校驗（Spot-check）

```python
# 對 200 個隨機樣本，用 GPT-4o 跑同樣 reward 評分
# 計算與本地 judge 的 Spearman ρ
# 預期：ρ > 0.7（強相關，本地 judge 可信）
# 成本：$5 USD
```

#### Group 3: 人類評估（Layer 1 — 25 人 Likert）

```python
# 採用 SimTube 完全相同的 protocol
human_eval = {
    "participants": 25,
    "platform": "Upwork or Prolific",
    "tasks_per_participant": "1 short-form video + N segment reactions × 3 personas",
    "video_count": 8,
    "rating_scale": "7-point Likert",
    "dimensions": [
        "Relevance" (vs segment),
        "Believability" (像不像該 persona),
        "Helpfulness" (對創作者有用)
    ],
    
    "quality_control": {
        "must_watch_video": True,
        "must_pass_video_quiz": "80% accuracy",
        "must_write_summary": True
    },
    
    "estimated_cost": "$300-500 USD"
}
```

#### Group 4: 段層級對齊評估（本研究獨有）

```python
# Segment Alignment Accuracy
# 從生成的評論中隨機抽 50 樣本
# 移除「該評論屬於哪段」的標籤
# 讓人類評估者選擇「這個評論最適合哪一段（1-12）」
# 看 SimLens 的時序定位準確率

expected_result = "≥ 80% 準確率（SimTube 沒此能力）"
```

### 5.3 Ablation Study 設計

```
必跑的 ablations（缺一不可）：

A1. SimLens (full)                              ← 完整方法
A2. - w/o Phase 2 (RLAIH)                       ← 證明 RLAIH 必要
A3. - w/o Phase 1 (Distillation)                ← 證明蒸餾必要
A4. - w/o Multi-LoRA (single LoRA all personas) ← 證明多 LoRA 必要
A5. - w/o Multi-aspect Reward (single reward)   ← 證明 6 reward 必要
A6. - w/o Segment Relevance reward              ← 證明此新 reward 必要
A7. - w/o None handling (force every segment)   ← 證明 None 設計必要
A8. - w/o Iterative DPO (1 round only)          ← 證明迭代必要
```

註：Segment length 採用 10 秒固定分段（borrow from UMaT structured segmentation）。
    Length sweep ablation（5s / 10s / 20s）保留為 future work，不在本研究範圍內。
    本研究 scope 鎖定在 short-form video (30s–3min)，段長為 fixed-length design choice。

### 5.4 預期結果表（你論文的 main result）

#### Table 1: 主結果（自動指標）

```
Method                          | Relevance | Persona | Linguistic | Segment | Coherence | Engaging
                                | (BERTScore)| Cons.   | Habits     | Relev.  |           | (UniEval)
─────────────────────────────────────────────────────────────────────────────────────────────────
Llama-3.2-3B zero-shot          | 0.55      | 0.42    | 0.38       | 0.40    | 0.50      | 0.45
Claude-3.5 Sonnet zero-shot     | 0.62      | 0.74    | 0.68       | 0.65    | 0.78      | 0.72
GPT-4o zero-shot                | 0.63      | 0.76    | 0.70       | 0.66    | 0.80      | 0.74
SimTube (whole-video, no segs)  | N/A       | 0.78    | 0.72       | N/A     | 0.79      | 0.76
─────────────────────────────────────────────────────────────────────────────────────────────────
SimLens Phase 1 only (SFT)      | 0.60      | 0.71    | 0.66       | 0.62    | 0.74      | 0.69
SimLens Phase 2 only (DPO)      | 0.58      | 0.73    | 0.69       | 0.64    | 0.71      | 0.70
SimLens Full (SFT + DPO) ⭐     | 0.62      | 0.83    | 0.81       | 0.78    | 0.78      | 0.77
                                | ≈ Teacher | > Teacher| > Teacher | > Teacher| ≈ Teacher| ≈ Teacher
```

**重點論述**：
- 通用指標（Relevance、Coherence、Engagingness）：SimLens ≈ Claude（蒸餾有效）
- 領域指標（Persona Cons、Linguistic、Segment Relev）：SimLens > Claude（RLAIH 有效）
- SimTube 沒有 Segment Relevance（無此能力）→ 是 SimLens 獨有貢獻

#### Table 2: 段層級對齊準確率（SimTube 完全沒有此能力）

```
Method                | Segment Alignment Accuracy | None Prediction F1
─────────────────────────────────────────────────────────────────────
Random baseline       | 1/N (8.3% @ 12 segs)       | 0.50
Llama-3B zero-shot    | 32%                        | 0.41
Claude zero-shot      | 71%                        | 0.62
SimLens Full          | 84%                        | 0.78
                      | (人類為 ~89%)               |
```

#### Table 3: 效率比較（評估 reference: 2-min short-form video, 12 segs）

```
Method            | Model Size  | VRAM    | Latency (per 2-min vid)| Cost / 1000 evals
─────────────────────────────────────────────────────────────────────────────────
Claude API        | ~600B est.  | N/A     | ~10s × 12 segs = 120s | $42
GPT-4o API        | ~1.7T est.  | N/A     | ~12s × 12 = 144s     | $52
SimTube           | Claude+GPT4 | N/A     | ~45s (whole video)   | $94
SimLens (3B+LoRA) | 3B + 200MB  | 6.5GB   | ~15s (12 segs)       | $0 (on-device)
                                                                ─────────────
                                                                100% saving
```

### 5.5 評估學術依據

| 評估元素 | 引用文獻 | 借鏡之處 |
|---------|---------|---------|
| 自動指標 6 項（NLG）| SimTube (Hung et al., 2024) Section 6.2 | BERTScore + ROUGE + Self-BLEU |
| Persona 評估（4 項）| PersonaGym (EMNLP 2025) | Persona Consistency + Linguistic Habits |
| Engagingness | PersoBench (Huang et al., 2024) | UniEval-based engagingness |
| Coherence | Score Before You Speak (2025) | coherence dimension |
| 25 人 crowd study | SimTube Section 6.1 | quiz + summary + rating protocol |
| **Segment Alignment Acc** | 本研究新增（UMaT 啟發） | 時序對齊作為評估 |
| **None Prediction F1** | 本研究新增 | 無反應預測能力評估 |
| Ablation 設計 | DPO 原論文 (Rafailov et al., 2023) | 標準 ablation 順序 |

---

## 6. 報告生成（Stage C）

### 6.1 為什麼用同一個 Llama-3B（不另外訓練）

```
論述：
  「One model, two tasks」是 2024-2025 多任務微調主流（Tülu 3、Llama Stack）。
  
  在 SimLens 中：
   - LoRA adapter for persona generation（8 個）
   - 報告生成直接用 base Llama-3B（不掛 adapter）
   - 情感分類也直接用 base Llama-3B（不掛 adapter，無需訓練）
  
  這證明：3B 模型不只能模擬 persona，還能整合多 persona 反應成有用報告，
        並能 zero-shot 對評論進行情感分類。
```

### 6.2 三步驟報告生成流程

Stage C 分為三個依序執行的步驟：

```
Step 6.2.1：情感分類（後處理，零訓練成本）
  對反應矩陣中每個非 None 的 cell：
    輸入：該 persona 的評論文字
    輸出：positive / negative / neutral
  
Step 6.2.2：逐評論建議（cell-level）
  對矩陣中每個有評論的 cell：
    根據情感極性與內容，產出該則評論的具體優化建議
    - positive → 如何強化這個吸引點
    - negative → 觀眾的具體不滿與修正方向  
    - neutral → 如何讓內容更突出
  
Step 6.2.3：整片建議（video-level）
  綜合分析整個反應矩陣：
    - 跨受眾比較
    - Persona 共鳴度分析
    - 正負面情感熱區分布
    - 整體節奏與內容調整方向
```

### 6.3 為什麼情感分類不需要訓練（學術依據）

```
LLM 已被證明在 zero-shot 情境下具備優異的情感分類能力：

[24] Hartmann et al. (Customer Needs and Solutions, 2024)
    證明 LLM zero-shot 在情感分類精度上不僅能與傳統 fine-tuned 
    transfer learning 方法競爭，甚至在某些情境超越。
    涵蓋 GPT-3.5、GPT-4、Llama 2 三類模型，
    包含三類分類（positive/negative/neutral），
    與 SimLens 採用粒度完全對齊。

[25] Lin et al. (JMIR/PMC, 2024)
    在 X、Reddit、YouTube 三個社群平台上系統性 benchmark。
    GPT-4 zero-shot 達 92-94% accuracy（F1 90-93%），
    Llama 2 達 72-75%。
    YouTube 場景與 SimLens 高度吻合，提供具體效能上界與下界。

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
Persona context: {persona_brief}

Output ONLY one word: positive, negative, or neutral.
"""

def classify_sentiment(comment, persona):
    if comment is None:
        return None
    response = llama_3b_base.generate(
        sentiment_prompt.format(comment=comment, persona_brief=persona)
    )
    return response.strip().lower()  # "positive" / "negative" / "neutral"
```

#### Prompt 2：逐評論建議（cell-level）

```python
per_comment_prompt = """You are a video coaching assistant.

Below is a comment from a specific audience persona watching segment {seg_idx}
(time {t_start}-{t_end}s) of a short-form video.

Persona: {persona_brief}
Segment content: {segment_text}
Comment: "{comment}"
Sentiment: {sentiment}

Based on the sentiment, provide ONE specific actionable suggestion:
- If positive: how to amplify this engagement point in editing
- If negative: what specific issue caused this and how to fix it
- If neutral: what's missing that could make it more engaging for this persona

Output: 1-2 sentences, actionable, specific to this persona.
"""

def suggest_per_comment(cell):
    if cell["comment"] is None:
        return None
    return llama_3b_base.generate(per_comment_prompt.format(**cell))
```

#### Prompt 3：整片建議（video-level）

```python
overall_prompt = """You are a video analytics assistant.

Reaction matrix (N segments × 8 personas) with sentiments:
{enriched_matrix}

Persona descriptions:
{persona_summaries}

Generate a structured overall report:

1. **Cross-Audience Comparison** (跨受眾比較)
   For each segment, identify which personas reacted positively/negatively/
   neutrally/didn't react.

2. **Persona Resonance Analysis** (Persona 共鳴度分析)
   For each persona, identify their top-3 most engaging segments 
   and bottom-3 disengaging segments based on sentiment + engagement.

3. **Sentiment Hotspot Map** (情感熱區分布)
   Identify segments with concentrated positive sentiment ("hot zones")
   and concentrated negative sentiment ("warning zones").

4. **Strategic Improvement Suggestions** (整體策略建議)
   Provide 3 specific, actionable suggestions for:
   - Negative-sentiment-heavy segments: root cause + fix
   - Cross-persona positive segments: how to amplify
   - Audience targeting: which persona this video best fits

Format: Markdown with clear sections.
"""

# 整合三步驟
def generate_full_report(reaction_matrix, video_segments, personas):
    # Step 1: 情感分類
    enriched_matrix = {}
    for (seg_i, persona_p), cell in reaction_matrix.items():
        enriched_matrix[(seg_i, persona_p)] = {
            **cell,
            "sentiment": classify_sentiment(cell["comment"], personas[persona_p])
        }
    
    # Step 2: 逐評論建議
    per_comment_suggestions = {
        key: suggest_per_comment(cell)
        for key, cell in enriched_matrix.items()
    }
    
    # Step 3: 整片建議
    overall_report = llama_3b_base.generate(
        overall_prompt.format(
            enriched_matrix=format_matrix(enriched_matrix),
            persona_summaries=format_personas(personas)
        )
    )
    
    return {
        "matrix_with_sentiment": enriched_matrix,
        "per_comment_suggestions": per_comment_suggestions,
        "overall_report": overall_report
    }
```

### 6.5 報告生成評估

由於這是衍生功能，**完全不需單獨訓練**，但需驗證品質：

```
評估維度：

1. 情感分類準確性
   - 從生成的 ~4800 個評論中隨機抽樣 200 個
   - 由人類重新標註情感極性
   - 計算與 Llama-3B base 分類的 Cohen's Kappa
   - 預期：κ ≥ 0.65（substantial agreement）
   - 對標：Llama 2 在 YouTube 場景的 72-75% accuracy (Lin et al., 2024)

2. 逐評論建議的可操作性
   - 找 5 位 YouTube 創作者
   - 評分維度：具體度、可執行度、與評論的相關度
   - 7-point Likert scale
   - 預期平均：5.0+/7.0

3. 整片建議的綜合性
   - 同樣 5 位創作者
   - 盲測 SimLens 報告 vs Claude 直接生成的報告
   - 評分維度：實用性、洞察深度、可操作性
   - 預期：與 Claude 報告達到 85%+ 滿意度
```

---

## 7. 實驗時程與里程碑

### 7.1 8 週時程表

```
Week 1: 環境建置 + Persona 設計
  □ 確認 GPU 環境（最少 RTX 3090 24GB）
  □ 安裝 LLaMA-Factory / TRL / Ollama
  □ Pull Llama-3.2-3B、Qwen3-32B Q4、LLaVA-NeXT
  □ 撰寫 8 個 persona YAML（已在本計畫提供）
  □ 寫好評估指標的 Python 函數骨架
  □ 抓 5-10 部影片做 pipeline sanity check
  ★ Milestone 1：環境就緒、pipeline 跑通

Week 2: 大規模影片素材收集 + 時序對齊 Pipeline
  □ 用 YouTube Data API 收集 100 部影片
  □ 跑 Whisper-Large-v3（含時間戳）
  □ 跑 LLaVA-NeXT 段描述（每 10 秒一段）
  □ UMaT-inspired 時序對齊 → 12 個 enriched segments
  □ 累積敘述生成
  ★ Milestone 2：100 部短影音 × N 段（avg ~12）就緒

Week 3: Phase 1 蒸餾資料生成
  □ Claude API 對每個 (影片, 段, persona) 生成反應
  □ 9,600 cells × ~50% 有評論 = ~4,800 評論 + 4,800 None
  □ 預算花費：$115 USD
  ★ Milestone 3：蒸餾訓練資料完成

Week 4: Phase 1 SFT 訓練
  □ 對 8 個 persona 各訓練 1 個 LoRA adapter
  □ 跑 baseline benchmark：
    - Llama 3B zero-shot
    - Claude zero-shot
    - GPT-4 zero-shot
  □ 跑 Phase 1 only 結果（SimLens-SFT）
  ★ Milestone 4：Phase 1 完整結果出爐

Week 5: Phase 2 RLAIH (round 1)
  □ 設置 Qwen3-32B 本地 judge
  □ 對每個 prompt 生 4 候選 → 6-aspect 評分
  □ 構造 ~9600 個 preference pairs
  □ 跑 DPO 訓練（每個 LoRA 獨立 update）
  ★ Milestone 5：RLAIH round 1 結果

Week 6: Phase 2 RLAIH (round 2) + 全套 ablation
  □ Iterative DPO round 2
  □ 跑完整 ablation：A2-A8（共 7 組）
  □ 跑 GPT-4 spot-check（200 樣本驗證 judge）
  ★ Milestone 6：完整自動評估結果

Week 7: 人類評估（Layer 1）
  □ 在 Upwork 招募 25 人
  □ 設計 Google Forms（含影片、quiz、評分）
  □ 收集 25 人 × 8 影片的 Likert 評分
  □ 收集 Segment Alignment Accuracy（50 樣本）
  □ 統計分析（Wilcoxon + Bonferroni）
  ★ Milestone 7：人類評估結果

Week 8: 論文撰寫 + 投稿準備
  □ 寫 6 頁 short paper / 2 頁 demo paper
  □ 製作 architecture diagram（含時序對齊）
  □ 錄製 3 分鐘 demo video
  □ GitHub repo 整理 + Hugging Face 上傳 8 個 LoRA adapters
  ★ Milestone 8：論文 + Demo 就緒
```

### 7.2 關鍵風險與緩解

| 風險 | 機率 | 緩解策略 |
|------|------|---------|
| GPU 記憶體不足 | 中 | 改用 Llama-3.2-1B 或 Q3 量化 |
| Claude API 預算超支 | 低 | 預估 $115，預留 $200 buffer |
| RLAIH 訓練不穩定 | 中 | 縮小 DPO learning rate，用 conservative β |
| None 反應比例失衡 | 中 | 用 reaction_frequency 控制 + Phase 1 SFT 校正 |
| 人類評估招募失敗 | 中 | 改用學校系內招募（30 人也夠 LBW） |
| Qwen judge 與 GPT-4 一致性差 | 低 | 用 multi-judge ensemble |

---

## 8. 預期硬體與成本

### 8.1 硬體需求

```
最低配置：
  - 1× RTX 3090 24GB
  - 64GB RAM、1TB SSD
  - 預估訓練時間：3 週

推薦配置：
  - 1× RTX 4090 24GB
  - 128GB RAM、2TB NVMe SSD
  - 預估訓練時間：2 週

理想配置：
  - 2× RTX 4090 或 1× A100 40GB
  - 256GB RAM
  - 預估訓練時間：1 週
```

### 8.2 成本估算

```
雲端 GPU（如果沒有自有硬體）：
  Vast.ai RTX 4090：$0.4/hour
  RunPod A100 40GB：$1.5/hour
  訓練總時數：~120 hours
  → $50-180 USD

API 成本：
  Claude API（蒸餾資料）：$115 USD
  GPT-4o（spot-check）：$5 USD
  → 小計：$120 USD

人類評估：
  Upwork crowd-sourcing 25 人：$300-500 USD

總成本：
  最低（自有 GPU + 校內招募）：$120 USD
  標準（雲端 GPU + Upwork）：$600-800 USD
```

---

## 9. 預期論文結構

### 9.1 6 頁版本（CHI LBW / UIST Poster）

```
1. Introduction (0.75 頁)
   - 痛點：YouTube Analytics 延遲、SimTube 等系統只給整片評論
   - SimLens 願景：lightweight + on-device + segment-level

2. Related Work (0.5 頁)
   - SimTube：whole-video persona simulation
   - UMaT：multimodal temporal alignment
   - PersonaGym：persona evaluation
   - DPO + RLAIF：訓練範式

3. Method (1.5 頁)
   - 3.1 Architecture: UMaT-inspired pipeline
   - 3.2 Phase 1: Distillation from Claude (含 None 反應設計)
   - 3.3 Phase 2: 6-aspect Multi-Reward DPO

4. Experiments (2 頁)
   - 4.1 Setup: 100 short-form videos × avg ~12 segments × 8 personas
   - 4.2 Main results: Table 1（自動指標）
   - 4.3 Segment alignment: Table 2（SimLens 獨有能力）
   - 4.4 Ablation: 8 組 configurations
   - 4.5 User study: 25 人 + 50 段對齊評估
   - 4.6 Efficiency: Table 3（vs SimTube）

5. Discussion + Limitations (0.75 頁)
   - 為什麼 student 能超越 teacher
   - None 反應的設計意義
   - 領域 gap：缺真實時序觀影行為資料

6. Conclusion (0.5 頁)
```

### 9.2 學術 Contribution 重述

```
C1. System Contribution
   First end-to-end lightweight (3B parameter) segment-level 
   persona-conditioned video audience simulation system.
   Provides 12-segment × 8-persona reaction matrix per video,
   runnable on consumer GPU (24GB).

C2. Methodological Contribution
   Two-stage training paradigm for ground-truth-scarce settings:
   - Distillation provides foundational segment-level reaction capability
   - 6-aspect multi-reward RLAIH (含 novel Segment Relevance reward) 
     provides domain breakthrough
   First to integrate UMaT-style temporal alignment with persona generation.

C3. Empirical Contribution
   First evidence that 3B model can match or exceed 600B teacher (Claude) 
   on segment-level persona-specific dimensions.
   First demonstration of None-reaction modeling (60%+ F1 in predicting 
   when a persona shouldn't react).
```

---

## 10. 完整文獻清單

```
[1] SimTube (Hung et al., 2024) — arXiv 2411.09577
    用於：總體架構、影片理解 pipeline、自動評估指標、人類評估 protocol

[2] UMaT (Bi & Xu, 2025) — arXiv 2503.09081 ⭐ NEW
    "Everything Can Be Described in Words: A Simple Unified Multi-Modal 
     Framework with Semantic and Temporal Alignment"
    用於：時序對齊 pipeline backbone、structured text representation

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
    用於：證明 RLAIH 可達 RLHF 同等效果

[9] OpenCharacter — arXiv 2501.15427
    用於：證明用 teacher LLM 蒸餾 persona 行為

[10] Bias-Adjusted LLM Agents (Kitadai et al., 2025) — arXiv 2508.18600
     用於：Persona-based fine-tuning 概念

[11] LoRA (Hu et al., 2021) — ICLR 2022
     用於：Multi-LoRA per persona 技術基礎

[12] Self-Rewarding LM (Yuan et al., 2024) — arXiv 2401.10020
     用於：僅借鏡 Iterative DPO 概念（多輪 DPO 持續改善 alignment）
          SimLens 不採用 self-judge 機制：
          actor (Llama-3.2-3B) 與 judge (Qwen3-32B) 為不同模型，
          避免 3B 自評的 self-bias 與能力不足問題

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
     用於：證明 LLM zero-shot 情感分類可媲美 fine-tuned 方法，
          支持 Stage C 用 Llama-3B base 直接做情感分類無需訓練

[22] Evaluating LLMs for Sentiment Analysis on Vaccine Posts (Lin et al., 2024)
     JMIR / PMC peer-reviewed
     https://pmc.ncbi.nlm.nih.gov/articles/PMC12526656/
     用於：YouTube 平台情感分類具體效能背書
          GPT-4 zero-shot 92-94% accuracy / Llama 2 zero-shot 72-75%

[23] VideoMultiAgents (Kugo et al., 2025)
     "VideoMultiAgents: A Multi-Agent Framework for Video Question Answering"
     arXiv: 2504.20091
     https://arxiv.org/abs/2504.20091
     用於：Section 1.2 決策 A 背書——感知與推理分離的可解釋性優勢
          證明專門代理人獨立處理各模態能避免單一巨型模型的黑箱干擾

[24] QMAVIS (Lin et al., 2026)
     "QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models"
     arXiv: 2601.06573
     https://arxiv.org/abs/2601.06573
     用於：Section 1.2 決策 B 背書——chunking + late fusion 策略
          在 VideoMME 上比端到端原生多模態模型準確率高 38.75%

[25] Neeko (Yu et al., EMNLP 2024 Main)
     "Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent"
     arXiv: 2402.13717
     https://arxiv.org/abs/2402.13717
     用於：Multi-LoRA per persona 設計直接前例——
          證明 per-character LoRA 優於 single LoRA + persona prompt
          SimLens 延伸：加 RLAIH/DPO 階段、video-grounded segment-level、None abstention

[26] Action-Guided Engagement (arXiv 2025/02)
     "Can LLMs Simulate Social Media Engagement? A Study on Action-Guided Response Generation"
     arXiv: 2502.12073
     https://arxiv.org/abs/2502.12073
     用於：None-reaction modeling 學術前例——
          平台層級「ignore」action 已被當作獨立訓練訊號
          SimLens 將此概念延伸至 persona-internal 層級（觀眾在這段是否會留言）

[27] PEFT Preference Alignment Trade-Offs (Thakkar et al., ACL 2024 Main)
     "A Deep Dive into the Trade-Offs of Parameter-Efficient Preference Alignment Techniques"
     arXiv: 2406.04879
     https://arxiv.org/abs/2406.04879
     用於：LoRA + DPO 整套技術背書——
          ACL 2024 Main 系統性 300+ 實驗證明 LoRA-SFT → LoRA-DPO 可行
          提供 adapter rank sensitivity、收斂行為等具體指引
          SimLens §3.5 + §4.6 兩階段訓練的權威背書

[28] Multi-MLLM Knowledge Distillation (Gu et al., arXiv 2025/05)
     "Multi-MLLM Knowledge Distillation for Out-of-Context News Detection"
     arXiv: 2505.22517
     https://arxiv.org/abs/2505.22517
     用於：完整 LoRA SFT + LoRA DPO 兩階段 prior art——
          直接前例：Stage 1 LoRA SFT + Stage 2 同 LoRA DPO + AI 訊號當 preference
          在 ~7B 級小模型 + 多教師蒸餾場景達 SOTA（< 10% labeled data）
          SimLens 擴展為多 LoRA per persona、多面向 reward、開放式生成
```

---

## 11. Limitations（誠實面對的限制）

```
寫進論文 Limitations 章節（這是加分項）：

L1. No Real-World Behavior Validation
    我們無真實 persona 觀影行為資料集（如：特定人口背景觀眾在影片
    哪些時段會留言、會跳出）。所有訓練訊號來自合成資料 + LLM-as-Judge，
    而非真實觀眾行為。
    這是領域 gap，不是 SimLens 獨有問題。

L2. Distillation Bias
    Phase 1 用 Claude 當 teacher，可能繼承 Claude 的偏誤
    （例如過度禮貌、避開敏感話題）。
    Phase 2 RLAIH 部分校正，但無法完全消除。

L3. LLM-as-Judge Limitations
    Qwen3-32B 與 GPT-4 一致性 ~ 85-90%。
    緩解：Multi-judge ensemble + GPT-4 spot-check。

L4. English-Only & Cultural Bias
    8 個 persona 都是英文 + 美國/亞洲文化導向。
    中文 / 跨文化擴展為 future work。

L5. Length-Generalization Constraint

    SimLens scope 鎖定在 short-form video (30s–3min)，10 秒固定分段。
    這是有意的設計選擇，理由：
      (a) 對應 TikTok / Reels / YouTube Shorts 三大平台主流長度
          —— 創作者經濟最大宗的內容形式
      (b) 訓練資料分布一致（同為短影音範疇），評估指標可比
      (c) UMaT 結構化分段在此粒度有最強學術背書
      (d) Cell 總數隨影片長度浮動（avg ~96 cells / video），但訓練資料量
          可預估（100 影片 × avg 12 段 × 8 persona ≈ 9,600 cells）

    對比：
      SimTube (Hung et al., 2024) 沒有指定影片長度，也沒做 length ablation
      —— SimLens 主動界定 scope 是 SimTube 的方法論增量

    未驗證的泛化邊界（保留為 future work）：
      ✗ < 30 秒（過短）：persona 反應空間可能過於受限
      ✗ > 5 分鐘（過長）：context length 與訓練成本爆炸
      ✗ Segment length sweep（5s / 10s / 20s）：本研究使用 UMaT 推薦的 10s
      ✗ 場景感知分段（PySceneDetect / TextTiling）：未實作

    Future work：
      F2 Adaptive Segmentation
      F6 Long-form Video Support

L7. No Verification of Reaction Frequency Ground Truth
    每個 persona 的 reaction_frequency 是我們設定的（如 P1 是 high），
    沒有真實資料驗證這個假設。
    這影響 None reward 的設計。

L8. Two-Stage Pipeline Choice
    SimLens 採用 SFT + DPO 兩階段，未採用單階段方法（如 ORPO、DPO+NLL）。
    Trade-off：訓練流程複雜度高（8 LoRA × 2 phase × 2 round = 32 次訓練 run），
    但保留 SFT/DPO 各自獨立的 ablation slot（A2/A3），是 SimLens 核心 contribution。
    Future work 可比較 ORPO single-stage 在 3B + persona 領域的表現。

L9. Self-Rewarding 不適用於 SimLens 規模
    Yuan et al. (2024) 的 Self-Rewarding LM 在 Llama-2-70B 成功，
    後續 Meta-Rewarding 在 Llama-3-8B 仍能 work。
    但社群實驗顯示 3B 規模下 self-judging quality 會崩 ——
    模型太小無法可靠自評（self-rewarding 範式的 known frontier failure）。
    SimLens 採用 external judge (Qwen3-32B) 正是繞開此限制：
      - actor (Llama-3.2-3B) 與 judge (Qwen3-32B-Q4) 為不同模型
      - 「強者評弱者」設計避開 3B 自評的 self-bias 與能力不足
      - 6-aspect 結構化 reward 需要 external judge + PersonaGym rubric 才好做
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
    結合 video boundary detection

F3. Hierarchical Persona LoRA
    粗 persona（demographics）+ 細 persona（individual quirks）
    兩層 LoRA 結構

F4. Cross-cultural Extension
    中、英、日、韓四語 persona

F5. Real-time Director Mode
    從事後分析 → 創作中即時建議
    整合進 Premiere Pro / DaVinci 等剪輯軟體

F6. Long-form Video Support
    結合 hierarchical summarization 處理 10+ 分鐘影片
    目前 scope 鎖定 short-form (30s–3min) 以節省 token、確保訓練資料分布一致
```

---

# 附錄：給你的具體 Action Items

## 本週可立即開始

```
□ Day 1: 確認硬體（至少 24GB VRAM）
□ Day 2: 安裝環境（LLaMA-Factory / TRL / Ollama）
□ Day 3: Pull Llama-3.2-3B + Qwen3-32B Q4 + LLaVA-NeXT
□ Day 4: 把 8 個 persona YAML 確認下來（本計畫已提供）
□ Day 5: 用 YouTube API 收集 5-10 部測試影片（驗證 pipeline）
□ Day 6-7: 跑通 Whisper + LLaVA + UMaT-inspired 對齊 pipeline
            重點：確認 12 個 enriched segments 的品質
            注意：5-10 部不是訓練資料，是工程驗證
```

## 投稿目標確認

```
首選：UIST 2026 Posters / Demos（deadline ~ 7/10）
備選：ACM MM 2026 BNI 之後的 workshops（deadline 6-7 月）
保底：智慧創新大賞 2026 + GitHub 開源 + Hugging Face release
```

## 與 v1.0 的差異總結

```
v1.0（含留存曲線）→ v2.0（無留存曲線）

砍掉：
✗ 留存曲線預測（避開資料來源質疑）
✗ Mr. HiSum 對標（不需要了）

保留：
✓ UMaT-inspired 時序對齊
✓ 段層級 persona 反應生成（10 秒一段）
✓ 蒸餾 + RLAIH 兩階段
✓ 6-aspect multi-reward

調整：
↻ Reward 從 5 個變 6 個（新增 Segment Relevance）
↻ 訓練資料從 4000 變 ~9600 筆（含 None）
↻ 評估增加 Segment Alignment Accuracy + None Prediction F1
↻ Claude API 成本從 $50 變 $115

新增評估維度：
+ Segment Alignment Accuracy（段對齊準確率）
+ None Prediction F1（無反應預測能力）
+ Reaction Frequency Match（反應頻率符合度）
```
