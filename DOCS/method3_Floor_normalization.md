# 手法3：BI-Floor（中間層BIフロア最大化）— 実験計画 & 実装タスク（Markdown）

> 目的：GPT系（decoder-only Transformer）で観測される **中間層の冗長性**（層入出力がほぼ同じ／層を抜いても性能が落ちにくい）を、ShortGPTで導入された **Block Influence（BI）** を“学習中の正則化”として用い、**中間層のBIが一定値を下回らない（= 何もしない層を減らす）**ように誘導する。

---

## 0. 背景・関連研究（論文導入で使う要点）

### 0.1 LLMの層冗長性とBI（ShortGPT）

* ShortGPTは、LLMでは層ごとに **入力と出力が高いcos類似を持つ（=変換が小さい）層が存在**するとし、層重要度指標として **Block Influence（BI）** を提案しています。
* BIは「層入力 (X_i) と層出力 (X_{i+1}) の token-wise cosine similarity」を平均し、それを **1から引いた量**として定義されています（低BIほど“ほぼ同じ”で冗長）。

### 0.2 なぜ“中間〜深部”で冗長が起きやすいのか（Curse of Depth）

* Pre-LN Transformer（GPT/LLaMA系で一般的）では、学習が進むにつれて深部層が**恒等写像に近い挙動**になり、層の寄与が小さくなる現象が報告され、原因として **Pre-LNの出力分散が深さ方向に増大し、深部層の導関数が恒等に近づく**ことが挙げられています。([arXiv][1])

### 0.3 “中間表現の多様性”を増やす正則化の系譜

* **Seq‑VCR（ICLR 2025）**は、Transformerの中間層での“表現崩壊（多様性の低下）”を問題視し、分散・共分散に基づく正則化で中間表現の多様性を増やす方向を提案しています。
* **DeCov（2015）**は、隠れ表現の相関を下げて冗長を抑える正則化として古典的です。([arXiv][2])
* 近年のTransformer表現解析では、**sample-wise cosine similarity が層間類似の指標としてCKAと整合しうる**ことも示され、cosを使う測度設計の妥当性が補強されています。([arXiv][3])

### 0.4 近いが“目的が逆”な先行：層間cos類似を「上げる」正則化

* decoderの層変換が強い線形性を持つことを分析し、**連続層の埋め込みをcosで近づける正則化**（角度差を小さくする）を導入した研究があります。あなたのBI-Floorはこれと違い、**“近づける”のではなく、中間層が“ちゃんと変える”ことを促す（BIを下げない）**のが狙いです。

### 0.5 最近の“冗長変換を抑える”方向（比較対象として重要）

* ICLR 2026投稿の **CR-Guided Transformers** は、中間〜深部で冗長変換が生じることを前提に、独自の冗長度指標（分布のcoherence）と **冗長性損失**でそれを抑える、と主張しています。BI-Floorはこれより **指標が単純で実装コストが低い**一方、**ShortGPT由来のBIで「層が何もしない」問題を直接叩く**という立ち位置を作れます。([OpenReview][4])

---

## 1. 手法の概要（BI-Floor）

### 1.1 直感

* 冗長層は「出力が入力に似すぎている（cosが高すぎる）」
* ならば学習中に、**中間層のcos（入力↔出力）が高止まりしない**ように制約を入れる
* ただし「平均で押し上げる」と、一部層だけが頑張って他は死ぬ（低BI層が残る）ので、**“フロア（下限）”を上げる**のが狙い

### 1.2 BIの定義（ShortGPT）

ShortGPTの定義に従う（層 (i) ）：
[
BI_i ;=; 1 - \mathbb{E}*{X,t}\Bigg[\frac{X*{i,t}^\top X_{i+1,t}}{|X_{i,t}|*2|X*{i+1,t}|_2}\Bigg]
]
低BI（=cosが高い）ほど「その層は hidden state をほとんど変えていない」。

---

## 2. BI-Floor 正則化の設計（論文の方法セクションにそのまま書ける形）

### 2.1 中間層集合の定義

全層数 (L) に対して、まずは固定で：

* `mid_start = floor(L/3)`
* `mid_end   = ceil(2L/3)`
* (\mathcal{M}={mid_start,\dots,mid_end-1})

※後でアブレーションで帯域を動かす（§6参照）。

### 2.2 “フロア最大化”の基本形（ヒンジ）

目標下限を (\tau) として、
[
\mathcal{L}*{BI\text{-floor}}
= \frac{1}{|\mathcal{M}|}\sum*{i\in \mathcal{M}} \max(0,\tau - BI_i)
]

* (BI_i < \tau) の層だけ罰する（= 死んでる層だけ起こす）
* (BI_i \ge \tau) は罰しない（= 過剰に変えさせない）

総損失：
[
\mathcal{L}=\mathcal{L}*{LM} + \lambda \mathcal{L}*{BI\text{-floor}}
]

### 2.3 “真のフロア”に近い形（任意：softmin）

より “min(BI)” を狙いたい場合：
[
BI_{\min}^{(\beta)} = -\frac{1}{\beta}\log\sum_{i\in\mathcal{M}} e^{-\beta BI_i}
]
[
\mathcal{L}*{BI\text{-floor}} = \max(0,\tau - BI*{\min}^{(\beta)})
]

* βを大きくすると min に近づく
* 実装は簡単だが、まずはヒンジ（2.2）が扱いやすい

### 2.4 実装上の「BIの平均の取り方」

BI計算は token-wise cosine を平均するが、学習コストを抑えるために選択肢を用意：

* **mean**：全token平均（最も素直）
* **sample_k**：系列ごとにk tokenだけ抽出して平均（コスト削減）
* **pool**：`mean over tokens` してから cos（最安だが粗い）

---

## 3. 本研究の新規性（書き分け例）

### 3.1 既存（ShortGPT）との違い

* ShortGPT：BIで冗長層を **“見つけて削る”**（事後圧縮）
* BI-Floor：BIを **“学習目的に入れる”**ことで、**冗長性が生まれにくい学習**へ誘導（同サイズでも中間層が働く状態を作る）

### 3.2 既存（中間表現多様性正則化）との違い

* Seq‑VCR / DeCov：主に分散・共分散などで **表現の多様性**を促す
* BI-Floor：**「層入力↔層出力が似すぎ」**という冗長性の現象に直結した指標（BI）を、**層ごとに下限を課す形**で制御（“死層”を減らすのが主眼）

### 3.3 最近の冗長性正則化（CR-Guided）との違い

* CR-Guided：分布を周波数領域に写像するなど **測度が重め**な一方、冗長性抑制を狙う([OpenReview][4])
* BI-Floor：cosのみで **実装が軽い**、ShortGPTと整合する評価軸（BI）で **比較が明確**

### 3.4 “層間cos正則化で近づける”研究との違い（対照が綺麗）

* 連続層の埋め込みを **cosで近づける**（角度差を小さくする）正則化が提案されている
* BI-Floorは逆に、**中間層が“ちゃんと変換する”ことを担保**する（低BI層を減らす）
  → “層を似せる/早期飽和を促す”方向性（cos↑）とは別軸

---

## 4. 研究仮説（この手法で「何が起きるはずか」）

### H1：中間層の低BI層が減る

* BI分布の下側（例：下位10–30%）が持ち上がる
* “最低BI”や“τ未満の層の割合”が改善

### H2：中間層の「層抜き耐性」が下がる（=必要になる）

* 中間層を1層スキップした時の PPL 悪化が増える
  （“抜いても大丈夫”が減る＝冗長性低下）

### H3：LM性能は同等以上（少なくとも大崩れしない）

* val PPLが維持される（最終的に改善できれば強い）

---

## 5. 実験計画（段階的：小→主結果→一般性）

### 5.1 実験A：ベースライン計測（τ設計のための校正）

**目的**

* ベースラインGPTの中間層BI分布を取り、**現実的なτ**を決める

**手順**

1. 小規模GPT（例：GPT-2 small相当）を短期事前学習
2. 学習途中のチェックポイントで層別BIを計測
3. 中間層BIの分位点から τ を決定（例：中間層BIのp30を τ にする、など）

**根拠（論文での言い方）**

* ShortGPTのBI定義に沿い、pre-normで入力出力cosが高くなる傾向があるため、固定τよりデータ駆動で置くのが自然。

---

### 5.2 実験B：Sanity（学習が壊れない＋BI-Floorが効いてる）

**目的**

* 正則化が動作し、NaN/発散が起きない
* “τ未満の層”が減る

**設定**

* 学習ステップは短めでOK（まず現象が動くこと）
* (\lambda) は warmup→ramp 推奨（§7参照）

**必須ログ**

* 層別BI（平均、min、p10）
* `num_layers_below_tau`（中間層で τ 未満の層数）
* train loss / val PPL

---

### 5.3 実験C：主結果（冗長性低下を“指標で”示す）

**目的**

* “中間層冗長性が減った”を主要指標で示す

**評価（主）**

1. **BI曲線・BI分布**（ベースライン vs BI-Floor）
2. **層抜き感度**（layer drop / layer skip）：中間層の (\Delta PPL_i)
3. **層間表現類似（任意だが強い）**：CKAまたはcosで、隣接層の表現が似すぎていないか

* CKA基礎：([arXiv][5])
* cosが層類似を捉えるという補強：([arXiv][3])

**評価（従）**

* val PPL
* 可能なら小さめ生成評価（整合性/反復/多様性など簡易指標）

---

### 5.4 実験D：一般性（スケールアップ）

* モデルを大きくする（例：100M→400M級など）
* データ量を増やす
* BI-Floorの効果が「小モデル限定の現象ではない」ことを示す

---

## 6. ベースライン・アブレーション（論文の説得力を作る）

### 6.1 必須ベースライン

* **BL-0**：正則化なし（通常学習）
* **BL-1（平均BI押し上げ）**：
  [
  \mathcal{L}*{BI\text{-mean}}=\max(0,\tau - \frac{1}{|\mathcal{M}|}\sum*{i\in\mathcal{M}} BI_i)
  ]
  → 「平均を上げる」だけではフロアが上がらないことを示す対照
* **BL-2（全層BI-Floor）**：中間だけが重要という主張の対照
* **BL-3（計算対照）**：同じcos計算だけするが損失に入れない（オーバーヘッド対照）

### 6.2 重要アブレーション

1. **τ設計**

   * 固定τ vs 分位点τ（実験A） vs τスケジュール（後半だけ上げる）
2. **(\lambda) スケジュール**

   * 固定 vs warmup→ramp（推奨）
3. **BI計算粒度**

   * block境界BI（基本）
   * Attention後・MLP後BI（“どこが死んでるか”切り分け）
4. **勾配の流し方（安定化アブレーション）**

   * `detach_input=True`：BI計算時の (X_i) をdetachして、層間カップリングを弱める
   * `detach_output=True`：逆（通常は不要だが対照になる）

---

## 7. ハイパラ設計の指針（詰まりやすい所を先に潰す）

### 7.1 τ（フロア目標）

* 初期値は **ベースライン中間層BIの分位点**から置くのが安全

  * 例：中間層BIのp30をτにする（“下位30%の層を起こす”）

### 7.2 λ（正則化強度）

* **いきなり強くすると学習が壊れやすい**（出力を無理に回転させる方向に引っ張るため）
* 推奨：`warmup_steps` までは λ=0 → `ramp_steps` で 0→λmax を線形増加

---

## 8. 実装計画（必要コードを階層的に）

> ここでは「どんなファイル／関数／フックが必要か」を実装タスクとして落とします。

### 8.1 リポジトリ構成案（例）

```
bi-floor-llm/
  configs/
    gpt2_baseline.yaml
    gpt2_bifloor.yaml
  src/
    losses/
      bi_floor.py
    models/
      gpt2_bifloor.py
      hooks.py
    train/
      train_clm.py
      schedules.py
    eval/
      ppl.py
      bi_metric.py
      layer_drop.py
      repr_similarity.py   # (任意) CKA/cos
    utils/
      logging.py
      seed.py
  tests/
    test_bi_floor_loss.py
    test_bi_numerics.py
```

---

### 8.2 コア実装①：BI計算（`src/losses/bi_floor.py`）

#### 関数：`compute_bi(x_in, x_out, attention_mask=None, eps=1e-6, token_reduce="mean", sample_k=32, fp32=True)`

* 入力

  * `x_in, x_out`: `[B, T, D]`（ブロック境界のhidden states）
  * `attention_mask`: `[B, T]`（padding除外が必要なら）
* 出力

  * `bi`: スカラー（または層ごとのベクトル）

#### 実装メモ（疑似コード）

```python
def compute_bi(x_in, x_out, mask=None, eps=1e-6, fp32=True):
    # x_in/out: [B,T,D]
    if fp32:
        a = x_in.float()
        b = x_out.float()
    else:
        a, b = x_in, x_out

    dot   = (a * b).sum(-1)                      # [B,T]
    na    = (a * a).sum(-1).sqrt()               # [B,T]
    nb    = (b * b).sum(-1).sqrt()               # [B,T]
    cos   = dot / (na * nb + eps)                # [B,T]
    if mask is not None:
        cos = cos * mask
        denom = mask.sum().clamp_min(1)
        mean_cos = cos.sum() / denom
    else:
        mean_cos = cos.mean()
    bi = 1.0 - mean_cos
    return bi
```

> ここで `bi = 1 - mean_cos` という形がShortGPTの式(1)と対応します。

---

### 8.3 コア実装②：BI-Floor損失（`src/losses/bi_floor.py`）

#### 関数：`bi_floor_loss(bi_list, tau, mode="hinge", beta=20.0)`

* `bi_list`: 中間層のBI（`torch.stack`して `[num_mid_layers]`）
* `mode="hinge"` の実装：

  * `loss = relu(tau - bi_list).mean()`
* `mode="softmin"` の実装（任意）：

  * `bi_min = -logsumexp(-beta*bi_list)/beta`
  * `loss = relu(tau - bi_min)`

---

### 8.4 コア実装③：モデル側で「層入出力」を捕まえる（`src/models/gpt2_bifloor.py`）

#### 実装方式（おすすめ順）

1. **方式A（最短）**：`output_hidden_states=True` を使って `hidden_states[i]` と `hidden_states[i+1]` からBIを計算

   * 長所：差分が最小
   * 短所：メモリが重い（スケールで厳しい）

2. **方式B（推奨）**：ブロックforward内で `h_in` と `h_out` を使い逐次BIを計算し、`reg_loss` に加算

   * 長所：hidden_statesを保持しない（軽い）
   * 短所：ブロック実装に手を入れる必要あり

#### 方式Bの流れ（概念）

* ブロックループで

  * `h_in = h`
  * `h = block(h, ...)`
  * `if layer_idx in mid: bi = compute_bi(h_in, h, mask)`
  * `collect_bi.append(bi)`
* forwardの返り値に `bi_list` か `bi_reg_loss` を含める（後段で損失合成）

---

### 8.5 学習コード統合（`src/train/train_clm.py`）

#### config項目（最低限）

* `bifloor.enabled: bool`
* `bifloor.mid_start, bifloor.mid_end`
* `bifloor.tau: float`（or 分位点指定）
* `bifloor.lambda_max: float`
* `bifloor.lambda_schedule: constant | warmup_ramp`
* `bifloor.detach_input: bool`
* `bifloor.token_reduce: mean|sample_k|pool`
* `bifloor.eps, bifloor.fp32_proj`

#### 学習ループ（概念）

1. forwardで `lm_loss` と `bi_list`（or `bi_reg_loss`）を得る
2. `lambda_t = schedule(step)`
3. `loss = lm_loss + lambda_t * bi_floor_loss(bi_list, tau)`
4. backward / step

---

### 8.6 ロギング（`src/utils/logging.py`）

最低限、次を出すと論文図が作れます：

* 層別BI（中間層だけでも可）
* `min_bi_mid`, `mean_bi_mid`
* `frac_below_tau_mid`（τ未満の割合）
* PPL（validation）
* 追加（推奨）：層抜き感度評価を回した結果（後述）

---

## 9. 評価実装（`src/eval/`）

### 9.1 BI評価（`bi_metric.py`）

* 学習中のBIログとは別に、**固定チェックポイントで同一データ条件**でBIを再計測（再現性・図のため）

### 9.2 層抜き感度（`layer_drop.py`）

* 各層 (i) をスキップして PPL差分 (\Delta PPL_i) を計測
* BI-Floorの主張に直結：**中間層の“抜ける層”が減る**はず

### 9.3 表現類似（任意：`repr_similarity.py`）

* cosまたはCKAで隣接層の類似度を測る
* CKAの基本：([arXiv][5])
* Transformerではsample-wise cosがCKAと整合しうる：([arXiv][3])

---

## 10. テスト（`tests/`）

* `test_bi_floor_loss.py`

  * hinge/softmin の値が想定通り
  * τを超えている層で勾配が出ない（ReLUの性質確認）
* `test_bi_numerics.py`

  * fp16/bf16でNaNが出ない（fp32計算オプション含む）
  * `mask.sum()==0` でも落ちない（clamp）

---

## 11. 想定される失敗モード & 対策（先回り）

### 11.1 τやλが強すぎて性能が落ちる

* 症状：PPL悪化、生成が崩れる
* 対策：

  * τを下げる（＝cosを少しだけ下げればよい設定にする）
  * λを下げる／rampを長くする
  * 中間層帯域を狭める（中心だけ起こす）

### 11.2 “無意味な回転/ノイズ”でBIだけ上げようとする

* cosを下げるだけなら、LM的には役に立たない変換でも達成できる
* 対策：

  * hingeで **必要最小限だけ**起こす（やりすぎない）
  * τを学習後半で少し上げる（初期に暴れさせない）
  * アブレーションで `detach_input` を試し、過剰な層間カップリングを避ける

### 11.3 mixed precisionで数値不安定

* 対策：cos計算だけfp32で行う（`fp32=True`）、`eps` を適切に

---

## 12. 論文用「貢献（Contributions）」テンプレ

1. **BI-Floor**：ShortGPTのBI（層入出力cosに基づく指標）を、層削除ではなく **学習中の正則化**として導入し、中間層の“低BI層（何もしない層）”を減らす手法を提案。
2. **フロア最大化という目的設計**：平均BIではなく下限（フロア）を押し上げることで、中間層での冗長性集中を抑える設計を提示。
3. **冗長性指標での定量評価**：BI分布・層抜き感度・（任意で）層間類似（cos/CKA）により、中間層冗長性の低減を直接示す。([arXiv][5])
4. **関連正則化との位置づけ**：中間表現多様性を増やす正則化（Seq‑VCR/DeCov）や、冗長変換抑制（CR-Guided）との比較を通じて、BI-Floorの軽量性・実装容易性・ShortGPTとの整合性を示す。

---

## 13. 最短ToDo（最初に回すべき順）

1. ベースラインで **中間層BI分布を計測**して τ を決める（実験A）
2. `compute_bi()` と `bi_floor_loss()` を実装＋数値テスト
3. 方式B（逐次計算）でモデルforwardにBI計算を組み込む
4. 小規模学習で `frac_below_tau_mid` が下がることを確認（実験B）
5. 同一条件で **層抜き感度**まで取り、冗長性低減を主結果として形にする（実験C）

---
[1]: https://arxiv.org/pdf/2502.05795 "The Curse of Depth in Large Language Models"
[2]: https://arxiv.org/abs/1511.06068?utm_source=chatgpt.com "Reducing Overfitting in Deep Networks by Decorrelating Representations"
[3]: https://arxiv.org/html/2406.14479v3 "Tracing Representation Progression: Analyzing and Enhancing Layer-Wise Similarity"
[4]: https://openreview.net/forum?id=YWLMoSmakk "CR-Guided Transformers: Coherence-Based Redundancy Identification and Regularization | OpenReview"
[5]: https://arxiv.org/abs/1905.00414?utm_source=chatgpt.com "Similarity of Neural Network Representations Revisited"
