# SimLens 研究計畫引用文獻清單
> 共 38 篇引用文獻，涵蓋系統架構、訓練方法、評估指標、設計決策背書、text-level video understanding 前例、結構化輸出評估、persona 取樣演算法、Socratic 多模態哲學等面向
> 對應研究計畫：SimLens_Research_Plan_v4.1.md
> 編號 [20] / [38] / [39] 為 v4.1 演進過程移除的條目（multi-judge ensemble 與 Bilibili 彈幕外部錨點驗證），保留斷號以維持其他引用編號穩定性

---

## A. 核心 baseline 與直接前身

### [1] SimTube
**完整標題**：SimTube: Generating Simulated Video Comments through Multimodal AI and User Personas

**作者**：Hung, Yu-Kai; Huang, Yun-Chien; Su, Ting-Yu; Lin, Yen-Ting; Cheng, Lung-Pan; Wang, Bryan; Sun, Shao-Hua

**發表場域 / 年份**：arXiv (NTU + University of Toronto) (2024)

**arXiv 編號**：`2411.09577`

**網址**：[https://arxiv.org/abs/2411.09577](https://arxiv.org/abs/2411.09577)

**類型**：主要 baseline / 直接前身

**在 SimLens 中的引用用途**：
總體架構靈感、影片理解 pipeline 設計、自動評估指標（BERTScore + ROUGE）、25 人 crowd-sourced 評估 protocol、影片類型選擇。**§2.3 PersonaChat → top-K 取樣範式直接照搬**（SimLens 取 top-8 + MMR，SimTube 取 top-30）。**§2 OpenAI text-embedding-3-small 同款 embedding 模型亦借自此論文**以確保跨研究可重現。

**摘要**：
提出 SimTube 系統，在影片發布前模擬觀眾評論，整合 Whisper + LLaVA-NeXT + LLM 與 PersonaChat 8K personas。透過 quantitative analysis、crowd-sourced（25人 7-point Likert）、qualitative user study（8 創作者）三層評估，證明生成評論在 Relevance、Believability、Helpfulness 上甚至超越真實 YouTube 評論。SimLens 的最直接學術前身。

---

### [2] UMaT
**完整標題**：Everything Can Be Described in Words: A Simple Unified Multi-Modal Framework with Semantic and Temporal Alignment

**作者**：Bi, Xiaowei; Xu, Zheyuan

**發表場域 / 年份**：arXiv (Northwestern University + IEEE) (2025)

**arXiv 編號**：`2503.09081`

**網址**：[https://arxiv.org/abs/2503.09081](https://arxiv.org/abs/2503.09081)

**類型**：分段策略補充背書（v4.1 從「核心方法靈感」降級）

**在 SimLens v4.1 中的引用用途**：
**v4.1 已將「all-modality-as-text」哲學主引用改為 Socratic Models [41]，「text-level video → LLM timestamp 輸出」直接前例改為 Chapter-Llama [32]**。UMaT 在 SimLens v4.1 的角色收斂為唯一獨特貢獻：
- **「10 秒等長分段」策略的細節依據** —— Chapter-Llama 採 ASR-guided 動態 frame selection（非等長）、Socratic Models 採 key moment extraction（非等長），UMaT 是唯一明確主張 fixed-length structured segmentation 的工作。

引用位置：Section 1.2 架構設計依據表（10 秒等長分段策略）、Section 1.3 決策 B 學術背書 1、Section 3.5 Phase 1 學術依據（10 秒等長分段策略）。

**摘要**：
提出 UMaT 框架，將視覺與聽覺輸入統一為「結構化文字」餵給 LLM，解決 semantic alignment、temporal synchronization、sparse information retrieval 三個問題。採用 fixed-length structured segmentation（依時間戳切分為等長片段）以維持語義與時間一致性。在長影片 QA 上比現有方法準確率高 13.7%（長影片 16.9%），透過 redundancy minimization 與 structured textual representation 達成。

**摘要**：
提出 UMaT 框架，將視覺與聽覺輸入統一為「結構化文字」餵給 LLM，解決 semantic alignment、temporal synchronization、sparse information retrieval 三個問題。在長影片 QA 上比現有方法準確率高 13.7%（長影片 16.9%），透過 redundancy minimization 與 structured textual representation 達成。

---

## B. Persona 設計與評估

### [3] PersonaChat
**完整標題**：Personalizing Dialogue Agents: I have a dog do you have pets too?

**作者**：Zhang, Saizheng; Dinan, Emily; Urbanek, Jack; Szlam, Arthur; Kiela, Douwe; Weston, Jason

**發表場域 / 年份**：ACL 2018 (2018)

**arXiv 編號**：`1801.07243`

**網址**：[https://arxiv.org/abs/1801.07243](https://arxiv.org/abs/1801.07243)

**類型**：v4.1 核心 persona 來源（直接從 8K 取樣）

**在 SimLens 中的引用用途**：
**SimLens v4.1 的 8 個 personas 直接從 PersonaChat 8K 取樣產生**（§2.3 描述完整流程）。採用 OpenAI text-embedding-3-small 計算 video keywords ↔ personas cosine similarity，先取 top-80 候選後以 MMR 過濾出 top-8 多樣 personas。**SimLens 不手動編寫 personas**，完全照搬 SimTube (IUI 2025) 的 PersonaChat-based 取樣範式以確保可重現性與避免「為什麼是這 8 個」reviewer 質疑。

**摘要**：
提出 PERSONA-CHAT 資料集，包含 8K+ persona descriptions 與配對的 dialog。每個 persona 由 5 句以上的 sentences 描述其 personality、background、hobbies。是後續 persona-based dialogue generation 研究最常用的資料集，SimTube 也採樣其 top-30 personas 做 cosine similarity ranking。

---

### [4] PersonaGym
**完整標題**：PersonaGym: Evaluating Persona Agents and LLMs

**作者**：Samuel, Vinay; Zou, Henry Peng; Zhou, Yue; Chaudhari, Shreyas; Kalyan, Ashwin; Rajpurohit, Tanmay; Reddy, Ameet Deshpande Karthik R.; Murahari, Vishvak

**發表場域 / 年份**：EMNLP 2025 Findings (2025)

**arXiv 編號**：`2407.18416`

**網址**：[https://arxiv.org/abs/2407.18416](https://arxiv.org/abs/2407.18416)

**類型**：Persona 評估 rubric 來源（**不提供 personas 給 SimLens 用**）

**在 SimLens 中的引用用途**：
PersonaGym 在 SimLens 有 **兩個獨立用途**，需區分：

1. **Persona 評估 rubric 來源（核心用途）**
   - SimLens §4.3 Reward C (R_content_quality) 中的 Persona Consistency 維度
   - SimLens §5.2 Group 2 Persona Consistency + Linguistic Habits 評估指標
   - 採用 PersonaGym 5 維度中的其中 2 個（Persona Consistency + Linguistic Habits）
   - Qwen3-32B-Q4 judge 直接套用 PersonaGym 1-5 分 rubric prompts

2. **數量決策的對比參照（次要用途）**
   - SimLens §2.2「為什麼選 8 個 personas」中引用 PersonaGym「每任務 5 個」作對比
   - 純粹是論述支援，不影響 SimLens 設計

**重要釐清**：PersonaGym 自帶 200 個 personas，**SimLens 完全不使用**。SimLens 的 8 個 personas 來自 PersonaChat 8K 取樣（§2.3），與 PersonaGym 無關。PersonaGym 在 SimLens 中只負責「評估」，不負責「生成」。

**摘要**：
提出首個動態評估 persona agents 的框架，定義 5 個學術界認可的 persona 評估維度：(1) Action Justification、(2) Expected Action、(3) Linguistic Habits、(4) Persona Consistency、(5) Toxicity Control。用 GPT-4o 與 LLaMA-3-70B 雙 evaluator，並用 Spearman/Kendall-Tau 證明與人類判斷高度相關。

---

### [5] PersoBench
**完整標題**：PersoBench: Benchmarking Personalized Response Generation in Large Language Models

**作者**：Afzoon, Saleh; Rahimi, Usman Naseem; Mohammad Shahed; Hussain, Amir; Beheshti, Amin

**發表場域 / 年份**：arXiv (2024)

**arXiv 編號**：`2410.03198`

**網址**：[https://arxiv.org/abs/2410.03198](https://arxiv.org/abs/2410.03198)

**類型**：Engagingness reward 來源

**在 SimLens 中的引用用途**：
Engagingness 評估維度（透過 UniEval 計算）；多維度 personalized response 評估方法

**摘要**：
提出 PersoBench 評估個人化回應生成，涵蓋 fluency、diversity、coherence、personalization (persona consistency + persona coverage)、engagingness、groundedness、instructability 多維度。Engagingness 評估「回應是否有趣、避免無聊或無資訊內容」，是 SimLens 第 5 reward 的依據。

---

### [6] Score Before You Speak (SBS)
**完整標題**：Score Before You Speak: Improving Persona Consistency in Dialogue Generation using Response Quality Scores

**作者**：Saggar, Arpita; Darling, Jonathan C.; Dimitrova, Vania; Sarikaya, Duygu; Hogg, David C.

**發表場域 / 年份**：ECAI 2025 (2025)

**arXiv 編號**：`2508.06886`

**網址**：[https://arxiv.org/abs/2508.06886](https://arxiv.org/abs/2508.06886)

**類型**：Coherence reward 來源

**在 SimLens 中的引用用途**：
Coherence 評估維度的 rubric 設計（與對話/影片內容的合理性）

**摘要**：
提出 SBS 框架統一 supervised finetuning 與 quality alignment，透過 score-conditioned training 改善 persona-consistent dialogue。在 PERSONA-CHAT 與 ConvAI2 上驗證，million 與 billion 參數模型皆有改善。Coherence 維度評估「回應是否與對話 context 連貫」。

---

### [10] Bias-Adjusted LLM Agents
**完整標題**：Bias-Adjusted LLM Agents for Human-Like Decision-Making via Behavioral Economics

**作者**：Kitadai, Ayato; Fukasawa, Yusuke; Nishino, Nariaki

**發表場域 / 年份**：arXiv (University of Tokyo) (2025)

**arXiv 編號**：`2508.18600`

**網址**：[https://arxiv.org/abs/2508.18600](https://arxiv.org/abs/2508.18600)

**類型**：Persona 設計概念支援

**在 SimLens 中的引用用途**：
Persona-based fine-tuning 的概念基礎；individual-level 行為差異化的方法論

**摘要**：
用 individual-level behavioral data（來自 Econographics 1000 人資料）注入 LLM 為 persona，透過 PCA 降到 6 個主成分（Generosity

---

### [22] PersonaLLM
**完整標題**：PersonaLLM: Investigating the Ability of Large Language Models to Express Personality Traits

**作者**：Jiang, Hang; Zhang, Xiajie; Cao, Xubo; Breazeal, Cynthia; Roy, Deb; Kabbara, Jad

**發表場域 / 年份**：NAACL 2024 Findings (2023)

**arXiv 編號**：`2305.02547`

**網址**：[https://arxiv.org/abs/2305.02547](https://arxiv.org/abs/2305.02547)

**類型**：Persona LLM 經典依據

**在 SimLens 中的引用用途**：
Synthetic persona dialog 訓練可行性的依據；prompt-based persona simulation 的學術基礎

**摘要**：
研究 LLM 是否能透過 persona prompt 表現出 Big Five 人格特質。對 GPT-3.5、GPT-4 等模型用 BFI 量表測試，並讓人類評估生成 stories 是否反映指定 persona。是 PersonaLLM 領域被 SimTube 直接引用的工作，也支持 SimLens prompt-based persona 設計。

---

## C. 訓練方法與強化學習

### [7] DPO
**完整標題**：Direct Preference Optimization: Your Language Model is Secretly a Reward Model

**作者**：Rafailov, Rafael; Sharma, Archit; Mitchell, Eric; Ermon, Stefano; Manning, Christopher D.; Finn, Chelsea

**發表場域 / 年份**：NeurIPS 2023 (2023)

**arXiv 編號**：`2305.18290`

**網址**：[https://arxiv.org/abs/2305.18290](https://arxiv.org/abs/2305.18290)

**類型**：核心訓練方法

**在 SimLens 中的引用用途**：
Phase 2 RLAIH 訓練演算法的核心；ablation 設計範式

**摘要**：
提出 DPO 演算法，無需明確 reward model，直接從 preference data 優化 policy。證明 LLM 本身就是隱含 reward model。在 dialogue summarization、sentiment generation 等任務上，DPO 比 PPO 更穩定且效果相當或更好。SimLens 在每個 LoRA adapter 獨立做 DPO update。

---

### [8] RLAIF Survey
**完整標題**：RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback

**作者**：Lee, Harrison; Phatale, Samrat; Mansoor, Hassan; Mesnard, Thomas; Ferret, Johan; Lu, Kellie; Bishop, Colton; Hall, Ethan; Carbune, Victor; Rastogi, Abhinav; Prakash, Sushant

**發表場域 / 年份**：arXiv (Google DeepMind) (2023)

**arXiv 編號**：`2309.00267`

**網址**：[https://arxiv.org/abs/2309.00267](https://arxiv.org/abs/2309.00267)

**類型**：RLAIH 方法論基礎 + d-RLAIF frozen judge 設計依據

**在 SimLens 中的引用用途**：
**借鏡兩個論點**：
- (1) **AI feedback 可達 RLHF 同等效果**（甚至更好，在 label 一致性高的任務上，例如 harmless dialogue generation）。
- (2) **d-RLAIF 證明 frozen off-the-shelf LLM 可直接當 reward source 取代訓練 reward model**，避開 RM staleness 問題（standard RLAIF 訓練的 RM 隨 policy 更新會逐漸 out-of-distribution）。

**SimLens 的 Qwen-32B judge 設計正是繼承 d-RLAIF 的此精神**：不訓練 judge，直接 zero-shot prompting。

**但 SimLens 用 DPO 取代 REINFORCE**，理由：
- (a) DPO 對 LoRA 微調支援更好（[30] Thakkar ACL 2024 系統驗證）
- (b) 24GB 單卡跑不動 REINFORCE 需要的 policy + value + judge 三模型同駐（Lee et al. 用 8× A100 80GB）
- (c) 我們的 6-aspect 結構化 reward 適合 DPO 的 preference pair 格式，而非 REINFORCE 的純量 reward（d-RLAIF 是 1-10 分純量）

**摘要**：
Google DeepMind 證明 RLAIF 可達到 RLHF 同等水準甚至更好，特別在 harmless dialogue generation 中超越 RLHF（因為 label definition 更一致）。論文提出兩個變體：
- **Canonical RLAIF**：LLM 標 preference data → 訓練獨立 Reward Model（PaLM 2 XS，BT loss）→ REINFORCE 更新 policy。RM 在 RL 階段凍結。
- **d-RLAIF（direct RLAIF）**：跳過訓練 RM，直接用 frozen off-the-shelf LLM（PaLM 2 XS）每步給 1-10 分當 reward。**human eval 上 d-RLAIF 優於 canonical RLAIF**，因 RM 訓練本身會引入 distillation loss、性能上限被 LLM judge 限制。

兩個變體都用 REINFORCE + KL penalty（β=0.05），不是 PPO。Base model 為 PaLM 2 XS。

---

### [12] Self-Rewarding LM
**完整標題**：Self-Rewarding Language Models

**作者**：Yuan, Weizhe; Pang, Richard Yuanzhe; Cho, Kyunghyun; Sukhbaatar, Sainbayar; Xu, Jing; Weston, Jason

**發表場域 / 年份**：arXiv (Meta) (2024)

**arXiv 編號**：`2401.10020`

**網址**：[https://arxiv.org/abs/2401.10020](https://arxiv.org/abs/2401.10020)

**類型**：Iterative DPO 概念依據（不採用 self-rewarding 機制）

**在 SimLens 中的引用用途**：
**僅借鏡 Iterative DPO 概念**（M1 → M2 → M3 多輪 DPO 可持續改善 alignment quality）。**SimLens 不採用 self-rewarding 機制** —— actor (Llama-3.2-3B) 與 judge (Qwen3-32B-Q4) 為**不同模型**，理由：
- (1) Llama-3B 太小，同時做 actor + judge 兩件事會兩邊都做不好（Self-Rewarding 用 70B 才可靠）
- (2) 自評有 self-bias，偏好自己風格的答案
- (3) Qwen 比 Llama-3B 強，符合「強者評弱者」的 judge 設計準則
- (4) SimLens 用 6-aspect 結構化 reward，需要 external judge 配合 PersonaGym rubric 才好做

**摘要**：
Meta 提出 Self-Rewarding LLM 範式：模型在訓練中既當 actor 又當 judge，透過迭代產生 preference pair 自我改善。Llama-2-70B 在 AlpacaEval 2.0 三輪迭代後達 20.44% win rate（Iter 1: 9.94% → Iter 2: 15.38% → Iter 3: 20.44%），超越 Claude 2、Gemini Pro、GPT-4 0613。**核心發現是 Iterative DPO 對 alignment 持續有效**，這部分結論不需要 self-rewarding 機制也成立 —— SimLens 借鏡此論點作為 Phase 2 round 2 迭代的依據，但用 external judge (Qwen3-32B) 取代 self-judge。

---

### [19] Tülu 3
**完整標題**：Tülu 3: Pushing Frontiers in Open Language Model Post-Training

**作者**：Lambert, Nathan; Morrison, Jacob; Pyatkin, Valentina; Huang, Shengyi; Ivison, Hamish; Brahman, Faeze; et al.

**發表場域 / 年份**：AI2 Technical Report (2024)

**arXiv 編號**：`2411.15124`

**網址**：[https://arxiv.org/abs/2411.15124](https://arxiv.org/abs/2411.15124)

**類型**：Post-training 範式背書

**在 SimLens 中的引用用途**：
多階段 post-training 範式（SFT → DPO → PPO）；one model multi-task 的論述基礎

**摘要**：
Allen Institute for AI 開源 Tülu 3 完整 post-training recipe，含 SFT、DPO、新型 RLVR (Reinforcement Learning with Verifiable Rewards)。8B 與 70B 版本在 instruction-following、math、knowledge、reasoning 全面超越 Llama 3.1 Instruct。直接支持 SimLens 「one Llama-3B serves persona generation + report generation」的多任務設計。

---

### [23] MORLAIF
**完整標題**：Multi-Objective Reinforcement Learning from AI Feedback

**作者**：Williams, Marcus

**發表場域 / 年份**：arXiv (2024)

**arXiv 編號**：`2406.07496`

**網址**：[https://arxiv.org/abs/2406.07496](https://arxiv.org/abs/2406.07496)

**類型**：Multi-aspect reward 背書

**在 SimLens 中的引用用途**：
Multi-aspect reward 的學術依據；解釋為何 6 個 reward 比 single reward 更穩定

**摘要**：
提出 MORLAIF 將 RLAIF 的 single reward 拆解為多個獨立 reward signal（如 toxicity、factuality、sycophancy），每個用獨立 preference model 學習後加權合成。改善 transparency、modularity、抗 over-optimization。直接支持 SimLens 6-aspect reward 設計。

---

## D. 蒸餾與 LoRA

### [9] OpenCharacter
**完整標題**：OpenCharacter: Training Customizable Role-Playing LLMs with Large-Scale Synthetic Personas

**作者**：Wang, Xiaoyang; et al.

**發表場域 / 年份**：arXiv (2025)

**arXiv 編號**：`2501.15427`

**網址**：[https://arxiv.org/abs/2501.15427](https://arxiv.org/abs/2501.15427)

**類型**：蒸餾範式背書

**在 SimLens 中的引用用途**：
證明用 teacher LLM（如 LLaMA-3-70B）蒸餾 role-playing 行為到小模型的可行性；Phase 1 蒸餾的方法依據

**摘要**：
證明可以透過大模型生成大規模合成 persona 資料來訓練可控 role-playing LLM。用 LLaMA-3-70B 當 teacher 合成 role-playing dialogues，效果不輸 GPT-4-based 方法。直接支持 SimLens Phase 1 的「Claude 蒸餾到 Llama-3B」設計。

---

### [11] LoRA
**完整標題**：LoRA: Low-Rank Adaptation of Large Language Models

**作者**：Hu, Edward J.; Shen, Yelong; Wallis, Phillip; Allen-Zhu, Zeyuan; Li, Yuanzhi; Wang, Shean; Wang, Lu; Chen, Weizhu

**發表場域 / 年份**：ICLR 2022 (2021)

**arXiv 編號**：`2106.09685`

**網址**：[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

**類型**：核心技術基礎

**在 SimLens 中的引用用途**：
Multi-LoRA per persona 的技術基礎；4-bit 量化 + LoRA rank 8 的標準設定

**摘要**：
提出 LoRA 透過 low-rank 矩陣分解，凍結原模型 weights 只訓練少量 adapter parameters，可在 GPT-3 175B 上將可訓練參數減少 10000 倍、GPU memory 減少 3 倍，且不增加推理延遲。是 SimLens「8 個 persona × 8 個 LoRA adapter」設計的技術基石。

---

### [28] Neeko
**完整標題**：Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent

**作者**：Yu, Xiaoyan; Luo, Tongxu; Wei, Yifan; Lei, Fangyu; Huang, Yiming; Peng, Hao; Zhu, Liehuang

**發表場域 / 年份**：EMNLP 2024 Main Conference (2024)

**arXiv 編號**：`2402.13717`

**網址**：[https://arxiv.org/abs/2402.13717](https://arxiv.org/abs/2402.13717)

**類型**：Multi-LoRA per persona 設計直接前例

**在 SimLens 中的引用用途**：
SimLens「8 個 persona × 8 個獨立 LoRA adapter」設計的最直接學術前例；證明 per-character LoRA 優於 single LoRA + persona prompt。回應 reviewer「為什麼不用 single LoRA + prompt 切換 persona」的關鍵防禦。

**摘要**：
Neeko 用 distinct LoRA blocks per character + dynamic gating 機制建構 multi-character role-playing agent，並提出 incremental learning 階段（fusion / expansion）支援新角色加入。在 multi-character dialogue 任務上顯著優於 single-LoRA + prompt conditioning baseline。SimLens 採用相同的 per-persona LoRA 架構，但延伸至：(1) 加入 RLAIH / DPO 階段（Neeko 只做 SFT）、(2) 處理 video-grounded segment-level 反應而非 free-form character dialogue、(3) 加入 None-reaction abstention 機制。

---

### [29] Action-Guided Engagement Generation
**完整標題**：Can LLMs Simulate Social Media Engagement? A Study on Action-Guided Response Generation

**作者**：（arXiv preprint 作者群）

**發表場域 / 年份**：arXiv (2025/02)

**arXiv 編號**：`2502.12073`

**網址**：[https://arxiv.org/abs/2502.12073](https://arxiv.org/abs/2502.12073)

**類型**：None-reaction modeling 學術前例

**在 SimLens 中的引用用途**：
SimLens「None reaction modeling」設計（讓 persona 顯式選擇不留言）的最直接學術前例。回應 reviewer「為什麼把 None 當作獨立訓練訊號」的防禦。

**摘要**：
提出 two-stage social media simulation pipeline：第一階段預測使用者 engagement action（retweet / quote / rewrite / **ignore**），第二階段在條件 action 下生成對應回應。在 GPT-4o-mini、o1-mini、DeepSeek-R1 上 benchmark。**「ignore」這個顯式 action 直接對應 SimLens 的「None」反應**，證明在 social engagement simulation 領域已有「不互動也是一種訊號」的學術共識。SimLens 將此概念從平台層級（retweet / ignore）延伸至 persona-internal 層級（這位觀眾在這段是否會留言）。

---

### [30] PEFT Preference Alignment Trade-Offs (Thakkar et al.)
**完整標題**：A Deep Dive into the Trade-Offs of Parameter-Efficient Preference Alignment Techniques

**作者**：Thakkar, Megh; Fournier, Quentin; Riemer, Matthew; Chen, Pin-Yu; Zouaq, Amal; Das, Payel; Chandar, Sarath

**發表場域 / 年份**：ACL 2024 Main Conference (2024)

**arXiv 編號**：`2406.04879`

**網址**：[https://arxiv.org/abs/2406.04879](https://arxiv.org/abs/2406.04879)

**類型**：LoRA + DPO 整套技術背書

**在 SimLens 中的引用用途**：
SimLens 整套訓練範式（4-bit + LoRA rank 8 + SFT + DPO）的最直接技術背書。回應 reviewer「LoRA 上接 DPO 會不會有問題」的核心防禦。

**摘要**：
ACL 2024 Main 系統性 300+ 實驗組合，在 LLaMA-1、Vicuna-v1.3、Mistral-7B、Mistral-7B-Instruct 上系統比較 LoRA / QLoRA SFT 後接 LoRA / QLoRA DPO 的多種 trade-offs。提供 adapter rank sensitivity、data informativeness、收斂行為等具體指引。直接證明「LoRA SFT → 同 LoRA DPO」是可行範式，是 SimLens 兩階段訓練的權威背書。SimLens 在此之上擴展為 (1) 多 LoRA per persona、(2) AI judge 取代 human preferences、(3) 多面向 reward。

---

### [31] Multi-MLLM Knowledge Distillation
**完整標題**：Multi-MLLM Knowledge Distillation for Out-of-Context News Detection

**作者**：Gu, Yimeng; Tong, Yi; Castro, Ignacio; Wu, Shu; Tyson, Gareth

**發表場域 / 年份**：arXiv (2025/05)

**arXiv 編號**：`2505.22517`

**網址**：[https://arxiv.org/abs/2505.22517](https://arxiv.org/abs/2505.22517)

**類型**：完整 LoRA SFT + LoRA DPO 兩階段 prior art

**在 SimLens 中的引用用途**：
SimLens Phase 1 + Phase 2 兩階段訓練流程的最直接前例。證明「LoRA SFT + LoRA DPO + AI 訊號當 preference」可在 ~7B 級小模型 + 多教師蒸餾場景下達到 SOTA。

**摘要**：
在 OOC（out-of-context）news detection 任務上，提出兩階段訓練 pipeline：Stage 1 用全部資料做 LoRA SFT；Stage 2 在多個 MLLM teacher 判斷不一致的資料點上做 LoRA DPO（同一個 adapter 繼續訓練）。teacher 不一致即作為 preference signal。在 < 10% labeled data 條件下達 SOTA，明確證明「LoRA SFT → 同 LoRA DPO」+ AI 訊號驅動的 preference 在多教師蒸餾場景可行。SimLens 將此範式擴展為：(1) 多 LoRA per persona（非 single LoRA），(2) 多面向 6-aspect reward（非 single disagreement signal），(3) 開放式生成（非 binary classification）。

---

### [21] DistilBERT
**完整標題**：DistilBERT a distilled version of BERT: smaller faster cheaper and lighter

**作者**：Sanh, Victor; Debut, Lysandre; Chaumond, Julien; Wolf, Thomas

**發表場域 / 年份**：NeurIPS 2019 Workshop (2019)

**arXiv 編號**：`1910.01108`

**網址**：[https://arxiv.org/abs/1910.01108](https://arxiv.org/abs/1910.01108)

**類型**：蒸餾範式經典論文

**在 SimLens 中的引用用途**：
Knowledge distillation 為起點的學術依據；證明 SFT on synthetic data 可有效 transfer 大模型能力

**摘要**：
Hugging Face 提出 DistilBERT，透過 knowledge distillation 將 BERT 縮小 40%、加速 60%，但保留 97% 的語言理解能力。是 knowledge distillation 在 NLP 領域最 seminal 的工作之一。SimLens Phase 1（用 Claude 蒸餾 Llama-3B）的概念源頭。

---

## E. LLM-as-Judge 與評估工具

### [13] LLM-as-Judge / MT-Bench
**完整標題**：Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena

**作者**：Zheng, Lianmin; Chiang, Wei-Lin; Sheng, Ying; Zhuang, Siyuan; Wu, Zhanghao; Zhuang, Yonghao; Lin, Zi; Li, Zhuohan; Li, Dacheng; Xing, Eric P.; Zhang, Hao; Gonzalez, Joseph E.; Stoica, Ion

**發表場域 / 年份**：NeurIPS 2023 (2023)

**arXiv 編號**：`2306.05685`

**網址**：[https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)

**類型**：LLM-as-Judge 方法論

**在 SimLens 中的引用用途**：
LLM-as-Judge 方法論基礎；Reward 計算用 LLM 評分的合理性

**摘要**：
提出 MT-Bench 與 Chatbot Arena 評估 LLM-as-Judge，證明 GPT-4 與人類判斷有 80%+ 一致性（與人類兩兩之間一致性相當）。系統性研究 LLM judge 的 position bias、verbosity bias、self-enhancement bias，並提供緩解策略。SimLens 5 個 LLM-as-Judge reward 的方法論依據。

---

### [14] Replacing the Judge
**完整標題**：Replacing the Judge: Can Llama 405B Outperform GPT-4 in the Court of AI?

**作者**：SambaNova Systems

**發表場域 / 年份**：Technical Report (2024)

**網址**：[https://sambanova.ai/blog/llm-as-a-judge](https://sambanova.ai/blog/llm-as-a-judge)

**類型**：本地 judge 可行性背書

**在 SimLens 中的引用用途**：
證明本地 LLM（如 Llama-3.1-70B）可以取代 GPT-4 當 judge；支持 SimLens 用本地 Qwen3-32B 的選擇

**摘要**：
SambaNova 系統性比較九個 judge models，發現只有 GPT-4 Turbo 與 Llama-3 70B 與人類評分達到 Cohen's Kappa ≥ 0.79 的高度一致性。後續工作證明 Llama-3.1 405B 在人類偏好對齊上達到或超過 GPT-4o。直接支持 SimLens 用本地 Qwen3-32B 取代 GPT-4 API 的設計。

---

### [15] UniEval
**完整標題**：Towards a Unified Multi-Dimensional Evaluator for Text Generation

**作者**：Zhong, Ming; Liu, Yang; Yin, Da; Mao, Yuning; Jiao, Yizhu; Liu, Pengfei; Zhu, Chenguang; Ji, Heng; Han, Jiawei

**發表場域 / 年份**：EMNLP 2022 (2022)

**arXiv 編號**：`2210.07197`

**網址**：[https://arxiv.org/abs/2210.07197](https://arxiv.org/abs/2210.07197)

**類型**：Engagingness 計算工具

**在 SimLens 中的引用用途**：
Engagingness 評估的具體實作工具（用其 dialogue task 的 engagingness module）

**摘要**：
提出 UniEval 統一多維度文本生成評估器，將評估任務轉為 boolean QA 問題。涵蓋 dialogue（4 dimensions：naturalness

---

## F. 基礎模型與工具

### [16] LLaVA-NeXT
**完整標題**：LLaVA-NeXT: Improved reasoning OCR and world knowledge

**作者**：Liu, Haotian; Li, Chunyuan; Li, Yuheng; Lee, Yong Jae

**發表場域 / 年份**：Project Page (LLaVA team) (2024)

**網址**：[https://llava-vl.github.io/blog/2024-01-30-llava-next/](https://llava-vl.github.io/blog/2024-01-30-llava-next/)

**類型**：核心視覺模型

**在 SimLens 中的引用用途**：
Stage A 影片視覺理解的核心模型；對每 10 秒段抽 4 frames 生成段描述

**摘要**：
LLaVA-NeXT (LLaVA 1.6) 改進 LLaVA 1.5，提升 reasoning、OCR、world knowledge 三大能力。支援更高解析度輸入（672×672 → 4 個 336×336 patches），這正是 SimTube 與 SimLens 拼接 4 frames 為 panel image 的依據。13B 版本在多個 benchmark 上超越 Gemini Pro。

---

### [17] Whisper
**完整標題**：Robust Speech Recognition via Large-Scale Weak Supervision

**作者**：Radford, Alec; Kim, Jong Wook; Xu, Tao; Brockman, Greg; McLeavey, Christine; Sutskever, Ilya

**發表場域 / 年份**：arXiv (OpenAI) (2022)

**arXiv 編號**：`2212.04356`

**網址**：[https://arxiv.org/abs/2212.04356](https://arxiv.org/abs/2212.04356)

**類型**：核心 ASR 模型

**在 SimLens 中的引用用途**：
Stage A 影片語音轉錄的核心模型；提供帶時間戳的 transcript（UMaT 對齊的關鍵）

**摘要**：
OpenAI 提出 Whisper 用 680K 小時多語言 weakly supervised data 訓練 robust ASR。輸出包含詞級時間戳，正是 SimLens UMaT-style 時序對齊的關鍵。Whisper-Large-v3 是目前 open-source ASR 中接近 SOTA 的選擇。SimTube 與 SimLens 都採用 medium 或 large variant。

---

### [18] Llama 3.2
**完整標題**：Llama 3.2: Revolutionizing edge AI and vision with open customizable models

**作者**：Meta AI Team

**發表場域 / 年份**：Meta Blog (2024)

**網址**：[https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

**類型**：核心 student 模型

**在 SimLens 中的引用用途**：
SimLens 的 student 模型（Llama-3.2-3B-Instruct）；證明 1B/3B 在特定 instruction-following 上可超越 Gemma 2 (2.6B) 與 Phi 3.5-mini

**摘要**：
Meta 發布 Llama 3.2，含 1B/3B（text-only edge）與 11B/90B（vision-language）。3B 版本針對 edge 部署優化，支援 128K context，在多個 instruction-following benchmark 上超越 Gemma 2 (2.6B) 與 Phi 3.5-mini。SimLens 選 3B 版本作為 quality-cost 甜蜜點。

---

## G. 情感分類功能

### [24] LLM Sentiment Analysis (Hartmann)
**完整標題**：Sentiment Analysis in the Age of Generative AI

**作者**：Hartmann, Jochen; Bergner, Anton; Hildebrand, Christian

**發表場域 / 年份**：Customer Needs and Solutions (Springer Nature) (2024)

**網址**：[https://link.springer.com/article/10.1007/s40547-024-00143-4](https://link.springer.com/article/10.1007/s40547-024-00143-4)

**類型**：情感分類功能背書

**在 SimLens 中的引用用途**：
Stage C 報告生成階段使用 LLM zero-shot 進行情感分類的學術依據。證明 SimLens 不需專門訓練即可達成情感分類功能

**摘要**：
系統性比較 GPT-3.5 / GPT-4 / Llama 2 在 sentiment classification 上的 zero-shot 表現與傳統 fine-tuned transfer learning 模型。研究結果顯示「儘管是 zero-shot LLM 在情感分類精度上不僅能與傳統方法競爭甚至超越」。涵蓋二類與三類（positive/negative/neutral）分類，文本來源涵蓋 Twitter、消費者評論等。直接支持 SimLens 在報告階段以 Llama-3B base 模型直接做情感分類無需額外訓練的設計選擇。

---

### [25] LLM Sentiment on Social Media
**完整標題**：Evaluating Large Language Models for Sentiment Analysis and Hesitancy Analysis on Vaccine Posts From Social Media

**作者**：Lin, Junyao; Lin, Jia-Ren; et al.

**發表場域 / 年份**：JMIR / PMC (peer-reviewed) (2024)

**網址**：[https://pmc.ncbi.nlm.nih.gov/articles/PMC12526656/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12526656/)

**類型**：情感分類數據背書

**在 SimLens 中的引用用途**：
Stage C 報告生成階段使用 LLM 進行情感分類的具體效能背書（92-94% accuracy）。涵蓋 YouTube 平台

**摘要**：
對 GPT-3.5 / GPT-4 / Claude-3 Sonnet / Llama 2 在 X / Reddit / YouTube 三個社群平台上的情感分類能力進行 benchmark。zero-shot setting 下 GPT-4 達 92-94% accuracy（F1 90-93%）；Claude-3 達 82-88%；Llama 2 達 72-75%。涵蓋 positive / negative / neutral 三類分類，與 SimLens 採用的分類粒度完全對齊。為 SimLens 「直接用 LLM 做情感分類無需訓練」提供具體量化背書。

---

## H. 設計決策背書 (Section 1.2)

### [26] VideoMultiAgents
**完整標題**：VideoMultiAgents: A Multi-Agent Framework for Video Question Answering

**作者**：Kugo, Noriyuki; et al. (Panasonic Connect)

**發表場域 / 年份**：arXiv (2025)

**arXiv 編號**：`2504.20091`

**網址**：[https://arxiv.org/abs/2504.20091](https://arxiv.org/abs/2504.20091)

**類型**：設計決策背書

**在 SimLens 中的引用用途**：
Section 1.2 決策 A 背書——SimLens 為何分離「感知」與「推理」而非用原生多模態端到端處理。證明專門代理人獨立處理各模態能避免單一巨型模型的黑箱干擾與錯誤傳播

**摘要**：
提出 VideoMultiAgents 框架，整合視覺、scene graph、文字三種專門代理人，由 organizer agent 統合各模態獨立輸出。在 Intent-QA 達到 79.0% (+6.2% over previous SOTA)、EgoSchema subset 75.4% (+3.4%)、NExT-QA 79.6% (+0.4%)。論文核心觀點：將感知任務交給專門 agent 並產出獨立文字報告，能讓推理 agent 在透明基礎上運作，避免單一巨型模型的黑箱與錯誤傳播。

---

### [27] QMAVIS
**完整標題**：QMAVIS: Long Video-Audio Understanding using Fusion of Large Multimodal Models

**作者**：Lin, Zixing; et al.

**發表場域 / 年份**：arXiv (2026)

**arXiv 編號**：`2601.06573`

**網址**：[https://arxiv.org/abs/2601.06573](https://arxiv.org/abs/2601.06573)

**類型**：設計決策背書

**在 SimLens 中的引用用途**：
Section 1.2 決策 B 背書——SimLens 為何要將影片分段而非整段處理。chunking + late fusion 策略的學術依據

**摘要**：
提出 QMAVIS 採用 late fusion 策略：將長影片切成 30-60 秒 chunk，分別交由 video LMM、Whisper、LLM 處理後再融合。在 VideoMME (with subtitles) 上比 VideoLlaMA2、InternVL2 等端到端原生多模態模型準確率高 38.75%。論文核心觀點：原生多模態模型為塞進 context window 必須採暴力 down-sampling 導致關鍵細節遺失；chunking 能確保每幀不被遺漏，維持最高品質特徵提取。

---

## I. Text-Level Video Understanding（v4.1 核心前例）

### [32] Chapter-Llama
**完整標題**：Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs

**作者**：Ventura, Lucas; Yang, Antoine; Schmid, Cordelia; Varol, Gül

**發表場域 / 年份**：CVPR 2025 (École des Ponts ParisTech / Inria)

**arXiv 編號**：`2504.00072`

**網址**：[https://arxiv.org/abs/2504.00072](https://arxiv.org/abs/2504.00072)

**GitHub**：[https://github.com/lucas-ventura/chapter-llama](https://github.com/lucas-ventura/chapter-llama)

**Project page**：[https://imagine.enpc.fr/~ventural/chapter-llama/](https://imagine.enpc.fr/~ventural/chapter-llama/)

**類型**：⭐ SimLens 機制最一致的直接前例（Llama + LoRA + ASR/Caption timeline → text timestamp output）

**在 SimLens v4.1 中的引用用途**：
**SimLens 機制 99% 對齊 Chapter-Llama** — 都是用 LoRA 微調的 Llama 模型，輸入「ASR `[HH:MM:SS]: ...` + Caption `[HH:MM:SS]: ...`」按時間排序的純文字 timeline，輸出 timestamp + 內容。差別僅在：
- Chapter-Llama: Llama-3.1-8B + LoRA rank=8 + 1 hr 影片 + chapter title 輸出
- SimLens: Llama-3.2-3B + per-persona LoRA rank=8 + 1–3 min 影片 + persona commentary 輸出

引用位置：Section 1.3 決策 C 直接前例（取代原 VTG-LLM 引用）、Section 3.5 Phase 1 學術依據、Section 4.6 Reward A 範式背書。

**摘要**：
Chapter-Llama 將 hour-long 影片 chaptering 任務在 text domain 解決，輸入為交錯的 ASR transcripts 與 frame captions（皆帶 timestamp），LLM 輸出 chapter boundaries 的 timestamps + 自由文字標題。基於 Llama-3.1-8B 用 LoRA rank=8 微調，~10.3 frames per video（vs Vid2Seq 100 frames）。MiniCPM-V 為 frame captioner。VidChapters-7M (817k 影片) 訓練 + 測試，達 F1 = 45.3（vs Vid2Seq SOTA 26.7），4× H100 訓練 40 min 即可完成。**首篇證明「pure text-level video timeline → LLM 輸出 timestamp + 內容」範式可達 SOTA 的 CVPR 工作**。

---

### [33] MMDuet (VideoLLM Knows When to Speak)
**完整標題**：VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format

**作者**：Wang, Yueqian; Meng, Xiaojun; Wang, Yuxuan; Liang, Jianxin; Wei, Jiansheng; Zhang, Huishuai; Zhao, Dongyan

**發表場域 / 年份**：arXiv preprint (2024)

**arXiv 編號**：`2411.17991`

**網址**：[https://arxiv.org/abs/2411.17991](https://arxiv.org/abs/2411.17991)

**GitHub**：[https://github.com/yellow-binary-tree/MMDuet](https://github.com/yellow-binary-tree/MMDuet)

**類型**：事件驅動稀疏預測直接前例

**在 SimLens v4.1 中的引用用途**：
SimLens v4.1「event-driven sparse prediction」設計的學術前例。MMDuet 證明 VideoLLM 可以自主決定「何時該說話」（在影片播放中的哪個位置插入文字回應），這正是 SimLens 期望 persona LoRA 學會的能力。Section 1.3 決策 C、Section 3.5 Phase 1 sparse temporal prediction 段落引用。

**摘要**：
提出 video-text duet interaction format：影片連續播放，user 與 model 都可在任何 timestamp 插入文字訊息（如二重唱輪流）。MMDuet 透過 informative head + relevance head 判斷是否該對某 frame 產生回應。在 YouCook2 dense captioning 達 76% CIDEr、QVHighlights 達 90% mAP、Charades-STA temporal grounding 達 25% R@0.5。同時釋出 MMDuetIT 訓練資料集與 MAGQA (Multi-Answer Grounded VQA) benchmark。

---

### [34] MM-When2Speak (Beyond Words)
**完整標題**：Beyond Words: Multimodal LLM Knows When to Speak

**作者**：Liao, Zikai; Ouyang, Yi; Lee, Yi-Lun; Yu, Chen-Ping; Tsai, Yi-Hsuan; Yin, Zhaozheng

**發表場域 / 年份**：arXiv preprint (2025)

**arXiv 編號**：`2505.14654`

**網址**：[https://arxiv.org/abs/2505.14654](https://arxiv.org/abs/2505.14654)

**類型**：多模態「何時說話」判斷補充背書

**在 SimLens v4.1 中的引用用途**：
SimLens v4.1 sparse 觸發預測在多模態場景的補充學術前例。MM-When2Speak 強調 short reactive utterances 依賴跨 vision/audio/text 的細微訊號，與 SimLens persona 對影片爆點的選擇性反應行為一致。Section 3.5 Phase 1 multimodal "when to speak" 背書。

**摘要**：
針對 LLM-based chatbot 在「何時說話」上的短板（特別是即時、簡短反應），提出 MM-When2Speak 模型，預測 response type 並強調短反應依賴 vision、audio、text 多模態訊號。屬於「event-driven prediction」概念在多模態 conversational agent 場景的延伸。

---

## J. 結構化輸出評估（v4.1 新增）

### [35] IFEval
**完整標題**：Instruction-Following Evaluation for Large Language Models

**作者**：Zhou, Jeffrey; Lu, Tianjian; Mishra, Swaroop; Brahma, Siddhartha; Basu, Sujoy; Luan, Yi; Zhou, Denny; Hou, Le

**發表場域 / 年份**：arXiv preprint, Google (2023)

**arXiv 編號**：`2311.07911`

**網址**：[https://arxiv.org/abs/2311.07911](https://arxiv.org/abs/2311.07911)

**Hugging Face**：[https://huggingface.co/datasets/google/IFEval](https://huggingface.co/datasets/google/IFEval)

**類型**：v4.1 Format Compliance 評估方法論

**在 SimLens v4.1 中的引用用途**：
SimLens v4.1 Group 0 Format Compliance Rate (FCR) 評估方法論的核心依據。IFEval 提出「verifiable instructions」概念（可被程式客觀驗證的指令），SimLens 借鏡此哲學定義 JSON Parsing Rate / Schema Compliance Rate / Timestamp Format Rate / Timestamp Legality Rate 等規則式評估指標。Section 5.2 Group 0、Section 5.5 評估學術依據引用。

**摘要**：
Google 提出 IFEval：一個易於重現的 instruction-following 評估 benchmark。聚焦於「可驗證指令」（verifiable instructions），如「字數 > 400」「至少提到關鍵字 AI 三次」等可被程式客觀檢查的條件。識別出 25 種可驗證指令類型，建構約 500 prompts。解決「人工評估昂貴慢、LLM-as-Judge 有偏誤」的雙重痛點。是後續所有結構化輸出評估的方法論起點。

---

### [36] JSONSchemaBench
**完整標題**：JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models

**作者**：Geng, Saibo; Cooper, Hudson; Moskal, Michał; Jenkins, Samuel; Berman, Julian; Ranchin, Nathan; West, Robert; Horvitz, Eric; Nori, Harsha

**發表場域 / 年份**：arXiv preprint, EPFL + Microsoft Research (2025)

**arXiv 編號**：`2501.10868`

**網址**：[https://arxiv.org/abs/2501.10868](https://arxiv.org/abs/2501.10868)

**GitHub**：[https://github.com/guidance-ai/jsonschemabench](https://github.com/guidance-ai/jsonschemabench)

**類型**：v4.1 constrained JSON decoding 技術背書

**在 SimLens v4.1 中的引用用途**：
SimLens v4.1 採用 constrained decoding（Outlines / XGrammar）強制輸出合法 JSON 的技術背書。JSONSchemaBench 系統性評估六大 constrained decoding 框架（Guidance, Outlines, Llamacpp, XGrammar, OpenAI, Gemini），提供 SimLens 選型依據。Section 3.3 SFT 訓練 constrained decoding、Section 5.2 Group 0 JSON-specific 評估方法引用。

**摘要**：
提出 JSONSchemaBench：包含 10K 真實 JSON schema 的 constrained decoding benchmark，涵蓋多元複雜度約束。配合官方 JSON Schema Test Suite 評估六種 SOTA constrained decoding 框架。提出三維度評估：efficiency（生成合規輸出的效率）、coverage（涵蓋約束類型的廣度）、quality（生成輸出的品質）。發現 constrained decoding 可比 unconstrained 加速 50%，但 Outlines 因 schema 編譯逾時 compliance 最低。

---

## K. 時序定位（v4.1 新增）

### [37] SoccerNet
**完整標題**：SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos

**作者**：Giancola, Silvio; Amine, Mohieddine; Dghaily, Tarek; Ghanem, Bernard

**發表場域 / 年份**：CVPR 2018 Workshops (KAUST) (2018)

**arXiv 編號**：`1804.04527`

**網址**：[https://arxiv.org/abs/1804.04527](https://arxiv.org/abs/1804.04527)

**Project Page**：[https://www.soccer-net.org/](https://www.soccer-net.org/)

**類型**：v4.1 時序定位評估範式來源

**在 SimLens v4.1 中的引用用途**：
SimLens v4.1 Reward A (R_timing) 設計的方法論來源。SoccerNet 建立了「tolerance window 內算 true positive」的 action spotting 評估範式，提供「時點顯著性」（event saliency）的概念基礎，被 SimLens 借鏡用於評估某 timestamp 是否真有「值得反應的事件」。Section 4.3 Reward A 引用。

**摘要**：
KAUST 團隊提出 SoccerNet：含 500 場完整足球比賽（764 小時）的大規模 action spotting 資料集。建立 mAP-based 評估範式（loose tolerance 5–60s、tight tolerance 1–5s），定義「候選 timestamp 落在 ground truth ±θ 秒內算 true positive」的 tolerance-based evaluation paradigm。後續 SoccerNet-v2 擴展到 17 類 action、SoccerNet 2024/2025 Challenges 加入 ball action spotting、player tracking 等新任務。是 sports video understanding 與 video temporal localization 領域的奠基性工作。

---

## L. Persona 取樣演算法（v4.1 新增）

### [40] MMR (Maximal Marginal Relevance)
**完整標題**：The Use of MMR, Diversity-Based Reranking for Reordering Documents and Producing Summaries

**作者**：Carbonell, Jaime; Goldstein, Jade

**發表場域 / 年份**：SIGIR 1998 (1998)

**DOI**：[https://doi.org/10.1145/290941.291025](https://doi.org/10.1145/290941.291025)

**類型**：Persona 取樣演算法基礎

**在 SimLens v4.1 中的引用用途**：
SimLens §2.3 Persona 取樣流程的關鍵步驟：先用 cosine similarity 從 PersonaChat 8K 取出 top-80 與影片內容相關的 personas，再用 MMR 過濾出 top-8 多樣化 personas（lambda=0.6 平衡相關性與多樣性）。避免 8 個 personas 過於相似（例如全是 K-pop 愛好者）。

**摘要**：
SIGIR 1998 經典論文，提出 Maximal Marginal Relevance (MMR) 演算法，公式為：
`MMR = argmax [λ × Sim(Di, Q) - (1-λ) × max Sim(Di, Dj)]`
在資訊檢索領域用於多樣化 ranking 結果，避免 top-K 結果過於相似。lambda 參數平衡相關性（與 query 的相似度）與多樣性（與已選結果的差異）。是 NLP / IR 領域 diversity-aware selection 的標準演算法，引用次數超過 6000+。

---

## M. All-Modality-as-Text 哲學祖師爺（v4.1 新增）

### [41] Socratic Models
**完整標題**：Socratic Models: Composing Zero-Shot Multimodal Reasoning with Language

**作者**：Zeng, Andy; Attarian, Maria; Ichter, Brian; Choromanski, Krzysztof; Wong, Adrian; Welker, Stefan; Tombari, Federico; Purohit, Aveek; Ryoo, Michael; Sindhwani, Vikas; Lee, Johnny; Vanhoucke, Vincent; Florence, Pete

**發表場域 / 年份**：ICLR 2023 (Google Research / DeepMind)

**arXiv 編號**：`2204.00598`

**網址**：[https://arxiv.org/abs/2204.00598](https://arxiv.org/abs/2204.00598)

**Project page**：[https://socraticmodels.github.io/](https://socraticmodels.github.io/)

**類型**：v4.1 Stage A 文字化哲學祖師爺工作

**在 SimLens v4.1 中的引用用途**：
SimLens Stage A 「Whisper 轉錄 + LLaVA 視覺描述 → 統一文字 Timeline Script」設計的哲學基礎。Socratic Models 提出將所有模態（視覺 / 音訊）降維為「**language-based world-state history**」，讓 LLM 純在文字域推理，不需訓練任何 multimodal model。SimLens 完全採用此範式（vs token-level 路線如 VTG-LLM）。Section 1.3 決策 C「all-modality-as-text」對比表、Section 3.5 Phase 1 Stage A 設計引用。

**摘要**：
Google Research 與 DeepMind 提出 Socratic Models (SMs) — 透過自然語言當共通介面，零樣本組合多個 pretrained foundation models（VLM、LM、ASR）解決多模態推理任務。核心做法：把影片視為 "**reading comprehension**" 問題，先用 VLM/ASR 抽出 key moments → caption → recursive summary → 形成 language-based world-state history → 餵給 LLM 推理。在 egocentric video QA、image captioning、video-text retrieval、機器人 perception/planning 等任務上驗證可行。是後續所有 text-level video understanding 工作（包含 Chapter-Llama、SimLens）的方法論起點，引用次數超過 1,500+。

---

## 附錄：快速索引（按編號）

| 編號 | 簡稱 | 發表場域 | 用途分類 |
|------|------|---------|---------|
| [1](#1-simtube) | SimTube | arXiv (NTU + University of Toronto) | 主要 baseline / 直接前身 |
| [2](#2-umat) | UMaT | arXiv 2025 | 10 秒等長分段策略細節依據（v4.1 從核心方法靈感降級為分段策略補充）|
| [3](#3-personachat) | PersonaChat | ACL 2018 | Persona 設計基礎 |
| [4](#4-personagym) | PersonaGym | EMNLP 2025 Findings | 核心評估指標來源 |
| [5](#5-persobench) | PersoBench | arXiv | Engagingness reward 來源 |
| [6](#6-score-before-you-speak-sbs) | Score Before You Speak (SBS) | ECAI 2025 | Coherence reward 來源 |
| [7](#7-dpo) | DPO | NeurIPS 2023 | 核心訓練方法 |
| [8](#8-rlaif-survey) | RLAIF Survey | arXiv (Google DeepMind) | RLAIH 方法論基礎 |
| [9](#9-opencharacter) | OpenCharacter | arXiv | 蒸餾範式背書 |
| [10](#10-bias-adjusted-llm-agents) | Bias-Adjusted LLM Agents | arXiv (University of Tokyo) | Persona 設計概念支援 |
| [11](#11-lora) | LoRA | ICLR 2022 | 核心技術基礎 |
| [12](#12-self-rewarding-lm) | Self-Rewarding LM | arXiv (Meta) | 迭代訓練方法 |
| [13](#13-llm-as-judge--mt-bench) | LLM-as-Judge / MT-Bench | NeurIPS 2023 | LLM-as-Judge 方法論 |
| [14](#14-replacing-the-judge) | Replacing the Judge | Technical Report | 本地 judge 可行性背書 |
| [15](#15-unieval) | UniEval | EMNLP 2022 | Engagingness 計算工具 |
| [16](#16-llava-next) | LLaVA-NeXT | Project Page (LLaVA team) | 核心視覺模型 |
| [17](#17-whisper) | Whisper | arXiv (OpenAI) | 核心 ASR 模型 |
| [18](#18-llama-3.2) | Llama 3.2 | Meta Blog | 核心 student 模型 |
| [19](#19-tülu-3) | Tülu 3 | AI2 Technical Report | Post-training 範式背書 |
| [21](#21-distilbert) | DistilBERT | NeurIPS 2019 Workshop | 蒸餾範式經典論文 |
| [22](#22-personallm) | PersonaLLM | NAACL 2024 Findings | Persona LLM 經典依據 |
| [23](#23-morlaif) | MORLAIF | arXiv | Multi-aspect reward 背書 |
| [24](#24-llm-sentiment-analysis-hartmann) | LLM Sentiment Analysis (Hartmann) | Customer Needs and Solutions (Springer Nature) | 情感分類功能背書 |
| [25](#25-llm-sentiment-on-social-media) | LLM Sentiment on Social Media | JMIR / PMC (peer-reviewed) | 情感分類數據背書 |
| [26](#26-videomultiagents) | VideoMultiAgents | arXiv | 設計決策背書 |
| [27](#27-qmavis) | QMAVIS | arXiv | 設計決策背書 |
| [28](#28-neeko) | Neeko | EMNLP 2024 Main | Multi-LoRA per persona 直接前例 |
| [29](#29-action-guided-engagement-generation) | Action-Guided Engagement | arXiv (2025/02) | None-reaction modeling 直接前例 |
| [30](#30-peft-preference-alignment-trade-offs-thakkar-et-al) | PEFT Preference Alignment Trade-Offs | ACL 2024 Main | LoRA + DPO 整套技術背書 |
| [31](#31-multi-mllm-knowledge-distillation) | Multi-MLLM Knowledge Distillation | arXiv (2025/05) | Phase 1+2 兩階段 prior art |
| [32](#32-chapter-llama) | Chapter-Llama | CVPR 2025 | ⭐ v4.1 機制最一致直接前例（Llama+LoRA+ASR/Caption timeline → text timestamp output）|
| [33](#33-mmduet-videollm-knows-when-to-speak) | MMDuet | arXiv (2024/11) | event-driven sparse prediction 直接前例 |
| [34](#34-mm-when2speak-beyond-words) | MM-When2Speak | arXiv (2025/05) | 多模態「何時說話」補充背書 |
| [35](#35-ifeval) | IFEval | arXiv, Google (2023) | v4.1 Format Compliance 評估方法論 |
| [36](#36-jsonschemabench) | JSONSchemaBench | arXiv, EPFL+MSR (2025) | constrained JSON decoding 技術背書 |
| [37](#37-soccernet) | SoccerNet | CVPR 2018 Workshops | R_timing 的 event saliency 概念背書 |
| [40](#40-mmr-maximal-marginal-relevance) | MMR | SIGIR 1998 | Persona 取樣多樣性過濾演算法 |
| [41](#41-socratic-models) | Socratic Models | ICLR 2023 (Google Research) | ⭐ All-modality-as-text 哲學祖師爺；SimLens Stage A Timeline Script 設計依據 |
