# 手法7：Pre‑LN由来の冗長化を抑えつつ「中間層を立てる」— 実験計画 & 実装タスク（Markdown）

> ゴール：GPT系（decoder-only Transformer）の **Pre‑LN が引き起こす深部/中間の「効きの悪さ・冗長化」**を、
>
> 1. **正規化（LN配置/スケーリング）側で“根本原因”を抑え**、
> 2. その上で **中間層が実際に寄与するように追加の“中間層立ち上げ”圧力**を入れて、
>    **層の価値（layer utility）とパラメータ効率を上げる**。

---

## 0. 背景・関連研究（導入に入れやすい要点）

### 0.1 Pre‑LN が「深部層が効かない（=ほぼ恒等）」を生む：Curse of Depth

* 大規模LLMで「深い層ほど寄与が小さい」現象を **Curse of Depth** として整理し、原因として **Pre‑LN における出力分散が深さ方向に大きく（指数的に）増大し、深部ブロックの導関数が恒等に近づく**ことで、深部が意味のある変換をしにくくなると報告されています。([arXiv][1])

### 0.2 Pre‑LNの“根本原因”に手を入れる代表手段

以下は「Pre‑LNの安定性を保ちつつ、深部/中間の寄与を回復する」ための有力な設計です。

#### (A) LayerNorm Scaling（LNS）

* LN出力を **層indexに応じて 1/√ℓ でスケール**し、深さ方向の分散成長を抑える方式。式(12)として明示されています。([arXiv][1])
* Pre‑LNの安定性を保ちながら深部寄与を改善する狙い。([arXiv][1])

#### (B) Mix‑LN（Pre‑LNとPost‑LNのハイブリッド）

* 総層数Lに対し、**最初の ⌊αL⌋ 層を Post‑LN、残りを Pre‑LN**にする設計（αはPost‑LN層の比率）。
* 目的は **勾配ノルムの層間バランス**を取り、深部/中間が学習しやすい状況を作ること。

#### (C) Peri‑LN（入力と出力の両方を正規化）

* 各サブレイヤ（Attention/MLP）について **入力も出力も正規化（LNを二回）**し、さらにembedding入出力も正規化する設計として定義されています。([arXiv][2])
* Pre‑LNで問題になりやすい **残差経路のスパイク（massive activations）**を抑えつつ、Post‑LNほど勾配経路を弱めないバランスを狙う説明がなされています。([arXiv][2])

### 0.3 “冗長層”の測定：ShortGPTのBI（Block Influence）

* LLMには層の冗長性があり、層重要度指標 **BI（Block Influence）** を導入して冗長層を削る ShortGPT が提案されています。([arXiv][3])
* 手法7では、この「冗長性の指標」を **学習中の“中間層立ち上げ”圧力**として再利用し、削るのではなく **冗長化を事前に抑える**側に使います（ここが論文化ポイント）。

---

## 1. 手法7のコアアイデア（提案の形）

### 1.1 二段構え：Root-cause mitigation × Mid-layer excitation

* **段1：Pre‑LN由来の冗長化（深さ方向の分散増大/勾配偏り）を正規化で抑える**

  * 候補：LNS / Mix‑LN / Peri‑LN
* **段2：中間層が“実際に”変換するように立ち上げる（冗長層を残さない）**

  * 候補：あなたが実験予定の **BI‑Floor（手法3）**や **MUR（手法4）**など
  * 手法7では、まず **BI‑Floorを標準の中間層立ち上げ**として採用（実装が軽く、ShortGPTのBIと整合的）

> まとめると：
> **(LN設計で深部が学習できる状態に戻す)** ＋ **(中間層の“死層”をBIで起こす)**
> という、原因と症状の両面からの設計。

---

## 2. 手法7の具体化（推奨の“主系”と、比較の“副系”）

### 2.1 主系（まず論文の主張を作りやすい最短ルート）

**LNS + BI‑Floor（Mid）**

* **LNS（正規化側）**：Pre‑LNの各LN出力に
  [
  \mathrm{LNScaled}_\ell(h)=\mathrm{LN}(h)\cdot\frac{1}{\sqrt{\ell}}
  ]
  を適用（ℓは1始まりの層index）。([arXiv][1])

* **BI‑Floor（中間層立ち上げ）**：中間層集合 (\mathcal{M}) について
  [
  \mathcal{L}*{BI\text{-floor}}
  =\frac{1}{|\mathcal{M}|}\sum*{\ell\in\mathcal{M}}\max(0,\tau - BI_\ell)
  ]
  を追加（BIはShortGPTの指標）。([arXiv][3])

**なぜ主系に向くか**

* LNSは **式が単純でハイパラ不要**、Pre‑LNの安定性も維持する設計として提示されているため、比較の軸が作りやすい。([arXiv][1])
* BI‑Floorは「中間層の死層」だけを起こせる（平均ではなく下限を持ち上げる）ため、過剰に全層を振らせにくい。

---

### 2.2 副系（論文の一般性を出す比較）

* **Mix‑LN + BI‑Floor（Mid）**

  * Mix‑LNは最初⌊αL⌋をPost‑LNにする明確な設計。
  * αのチューニングが必要（論文では0.25を採用している記述あり）。
* **Peri‑LN + BI‑Floor（Mid）**

  * Peri‑LNは入力/出力の両正規化で分散と勾配をバランスさせる設計。([arXiv][2])
  * 実装がやや重いが、近年のオープンモデルに近い設計議論と接続できる。

---

## 3. 研究仮説（主張の形）

### H1（原因側）：正規化変更で「深部が恒等に寄る」傾向が緩和される

* LNSはPre‑LNの深さ方向の分散成長を抑えることを目的としており、深部が学習に寄与しやすくなるはず。([arXiv][1])
* Mix‑LN/Peri‑LNも勾配や分散の層間バランス改善を狙う。

### H2（症状側）：中間層の冗長層（低BI）が減る

* ShortGPTはBIが冗長層の検出に効くと示しているため、BI‑Floorで中間層の低BIを押し上げれば“死層”が減るはず。([arXiv][3])

### H3（相乗効果）：LN変更単体より「中間層の寄与の底上げ」が強くなる

* LN変更だけだと「深部が効く」方向に寄る一方で、中間層の冗長性が残る可能性がある
* そこにBI‑Floorを足すことで「中間層を立てる」が達成でき、**層抜き耐性（layer drop耐性）が減る＝各層が必要になる**はず

---

## 4. 実験計画（段階的：現象再現 → 正規化比較 → 相乗効果）

### Phase 0：現象の再現（Pre‑LNの“冗長化”を自分の環境で可視化）

**目的**

* あなたのベース実装（GPT系）でも「深部/中間の効きの悪さ・冗長性」が出ることを示し、手法7の必然性を作る

**測るもの（必須）**

1. **層抜き感度**：層ℓをスキップ/除去したときの (\Delta)PPL
2. **層間類似**：隣接層の hidden state のcos/角度距離（Mix‑LN/Peri‑LN論文でも角度距離を用いて冗長性を見ている）
3. **BIプロファイル**：層別BI（ShortGPT定義）([arXiv][3])
4. **層別の出力分散と勾配ノルム**：Pre‑LNで深さ方向に分散が増える／層間で勾配が偏る、を確認（CoD/Peri‑LNの主張と対応）([arXiv][1])

---

### Phase 1：正規化のみの比較（Root-cause mitigation）

**比較条件（最低限）**

* BL：Pre‑LN（現状）
* A：Pre‑LN + **LNS**
* B：**Mix‑LN**（αを小さく1–2点だけ：例 0.25 と 0.125）
* C：**Peri‑LN**

**評価**

* val PPL / 学習安定性（発散しないか、勾配スパイク頻度）
* 層別分散・勾配ノルムの層間バランス
* BI分布（“正規化だけで冗長層が減るか？”の確認）

> ここで「LNSが最も実装が軽く、比較が綺麗」になりやすいです。LNSは1/√ℓでLN出力をスケールするだけ。([arXiv][1])

---

### Phase 2：相乗効果（手法7の本体：正規化 + 中間層立ち上げ）

**主系**

* BL：Pre‑LN
* R：Pre‑LN + BI‑Floor（= 手法3単体）
* N：LNSのみ
* **提案（手法7）**：LNS + BI‑Floor（中間層）

**副系（余力があれば）**

* Mix‑LN + BI‑Floor
* Peri‑LN + BI‑Floor

**主評価（論文のメイン図）**

1. **BI（層別）**：中間層の p10 / min / “τ未満割合”
2. **層抜き感度**：中間層を抜いた時の (\Delta)PPL が増える（＝抜けない＝必要になる）
3. **PPL**：性能維持 or 改善
4. **分散・勾配ノルム**：LNSで深さ方向の分散爆発が抑えられるか（CoDの狙いと対応）([arXiv][1])

---

## 5. 主要アブレーション（“なぜ効いたか”を分解）

### 5.1 正規化側

* LNSの指数：(\frac{1}{\ell^{p/2}}) の形で (p\in{0.5,1.0,1.5})（論文の主張はp=1相当）([arXiv][1])
* LNSの適用箇所：

  * LN1（attn前）だけ / LN2（mlp前）だけ / 両方
* Mix‑LN：α（Post‑LN比率）を1–3点だけ掃引（例 0.125/0.25/0.33）

  * α定義（最初⌊αL⌋層がPost‑LN）が明記。
* Peri‑LN：Output‑LNの学習可否（learnable scale固定/学習）

  * Peri‑LNが出力LNを含む設計であることの定義に沿う。([arXiv][2])

### 5.2 中間層立ち上げ側（BI‑Floor）

* 中間層帯域：L/3–2L/3（標準） vs もっと狭い（中心だけ）
* τの決め方：ベースラインBIの分位点（p20/p30/p40）から設定
* λスケジュール：warmup後にramp（最初から強いと学習が荒れやすい）

---

## 6. 実装計画（必要コードを階層的に）

> ここでは「どのファイルに何を実装するか」を、実装順に落とします。
> （HF Transformers / nanoGPT どちらでも成立する“構造”として書きます）

### 6.1 推奨リポジトリ構成

```
method7-ln-mid/
  configs/
    preln.yaml
    preln_lns.yaml
    mixln.yaml
    periln.yaml
    lns_bifloor.yaml          # 手法7（主系）
  src/
    models/
      gpt_blocks.py           # Block実装（PreLN/PostLN/PeriLN）
      norms.py                # LNS/Peri-LN用Norm
      build_model.py          # configからモデル組み立て
    regularizers/
      bi.py                   # BI計算
      bi_floor.py             # BI-Floor損失
    train/
      train_clm.py
      schedules.py
      logging.py
    eval/
      ppl.py
      layer_drop.py
      bi_profile.py
      stats_variance_grad.py
  tests/
    test_lns.py
    test_mixln_layout.py
    test_periln_layout.py
    test_bi_floor.py
```

---

## 6.2 実装①：LayerNorm Scaling（LNS）

### 6.2.1 `src/models/norms.py`

#### クラス：`DepthScaledRMSNorm` / `DepthScaledLayerNorm`

* 引数：`dim, eps, layer_idx (1..L), base_norm="rmsnorm|layernorm"`
* forward：

  1. `y = Norm(x)`
  2. `y = y * (1 / sqrt(layer_idx))`（ℓに応じたスケール）([arXiv][1])
  3. return y

> 注意：CoD論文では「LN出力を 1/√ℓ でスケール」と明記されています。([arXiv][1])

### 6.2.2 `src/models/gpt_blocks.py`

* Pre‑LN GPT block なら通常：

  * `x = x + Attn(LN1(x))`
  * `x = x + MLP(LN2(x))`
* これを **LN1/LN2 を DepthScaledNorm に差し替える**だけでOK

### 6.2.3 追加ログ（LNSの検証）

* `layerwise_hidden_var`：各層出力の分散
* `layerwise_grad_norm`：各層パラメータ勾配ノルム
* これらが“深さで暴れない”ことが、LNS導入の主要な根拠図になります。([arXiv][1])

---

## 6.3 実装②：Mix‑LN（PreとPostの混在）

### 6.3.1 `src/models/gpt_blocks.py`

#### 2種類のblockを用意

* `PreLNBlock`：通常のGPTブロック
* `PostLNBlock`：各サブレイヤ後にLNするブロック

  * 例：

    * `x = LN1(x + Attn(x))`
    * `x = LN2(x + MLP(x))`

### 6.3.2 `src/models/build_model.py`

* 総層 L、ハイパラ α に対し

  * `k = floor(alpha * L)`
  * `layers[0:k]` を `PostLNBlock`
  * `layers[k:L]` を `PreLNBlock`
* これはMix‑LNの定義（最初⌊αL⌋層がPost‑LN）に一致させることが重要。

### 6.3.3 注意（安定性）

* Mix‑LNはPost‑LN成分を含むため、学習が不安定になり得る（論文側も安定化の議論あり）。
* まず小モデルで学習が壊れないことを確認してからスケール。

---

## 6.4 実装③：Peri‑LN（入力・出力両正規化）

### 6.4.1 `src/models/gpt_blocks.py`

Peri‑LNの定義に沿って、各サブレイヤで **入力Norm + 出力Norm** を入れる：([arXiv][2])

* Attentionサブレイヤ：

  1. `z = LN_in_attn(x)`
  2. `u = Attn(z)`
  3. `u = LN_out_attn(u)`  ← “出力の正規化”
  4. `x = x + u`
* MLPサブレイヤも同様

### 6.4.2 embedding / final norm

* Peri‑LNはembedding入出力の正規化も含む設計として記述されています（実装はアブレーション可能）。([arXiv][2])

---

## 6.5 実装④：中間層立ち上げ（BI‑Floor）を統合（手法7の“Mid”）

### 6.5.1 `src/regularizers/bi.py`

* `compute_bi(h_in, h_out, attention_mask=None, eps=1e-6, fp32=True)`

  * BIの前提：層入力と層出力の“変化の小ささ”を測る（ShortGPTのBI）([arXiv][3])
  * 実装は token-wise cosine を平均して (1-\mathrm{cos}) など（あなたの手法3実装を流用可）

### 6.5.2 `src/regularizers/bi_floor.py`

* `bi_floor_loss(bi_list_mid, tau, mode="hinge")`

  * `relu(tau - bi).mean()`

### 6.5.3 `src/train/train_clm.py` 統合

* forward中に中間層の `(h_in, h_out)` を取り、`bi_list_mid` を作る
* `loss = lm_loss + lambda_t * bi_floor_loss(...)`

---

## 6.6 実装⑤：評価スクリプト（“立った”を証明する）

### 6.6.1 `eval/layer_drop.py`

* 1層ずつスキップしてPPL差分を出す（中間層が“抜けなくなる”か）

### 6.6.2 `eval/bi_profile.py`

* 層別BI（平均、p10、min、τ未満割合）

### 6.6.3 `eval/stats_variance_grad.py`

* 層別出力分散（hidden var）
* 層別勾配ノルム（param grad norm）
* Pre‑LN → LNS/Peri‑LN/Mix‑LNで、深さ方向の偏りが改善することを示す（CoD/Peri‑LNの議論に対応）([arXiv][1])

---

## 6.7 テスト（最低限）

* `test_lns.py`：スケールが 1/√ℓ になっているか（ℓ=1で1、ℓ=4で0.5）
* `test_mixln_layout.py`：最初⌊αL⌋層がPost‑LNになっているか（定義通り）
* `test_periln_layout.py`：入力/出力normが両方存在するか（定義通り）([arXiv][2])
* `test_bi_floor.py`：τ未満のみペナルティが出るか

---

## 7. 失敗モードと対策（先回り）

### 7.1 正規化変更で学習が崩れる

* Mix‑LNはPost‑LN成分があるため、warmupやLRが敏感になり得る（論文でもαやwarmupへの言及あり）。
  → まず **LNS（軽量・比較的安全）**で主結果を作り、Mix‑LN/Peri‑LNは副系で。

### 7.2 “正規化で深部が効く”だけで中間層が立たない

* これが手法7の存在意義
  → BI‑Floor（またはMUR）を足して「中間層の底上げ」を明確に示す

### 7.3 BI‑Floorが強すぎて性能が落ちる

* τ/λを控えめにし、「死層だけ起こす」運用に寄せる
* τはベースライン中間層BIの分位点から決める（p20–p40の範囲で）

---

## 8. 論文での新規性（言い切りやすい形）

### 8.1 既存は「正規化で深部を救う」か「冗長層を検出/削除する」が中心

* CoDはPre‑LN由来の問題を整理し、LNSで深部寄与を回復する。([arXiv][1])
* Mix‑LN/Peri‑LNも正規化配置で勾配・分散のバランスを取る。
* ShortGPTはBIで冗長層を同定し、削除する（事後圧縮）。([arXiv][3])

### 8.2 手法7の立ち位置（あなたの論文の芯）

* **正規化（原因）× 中間層立ち上げ（症状）**を同時に扱い、

  * 「深部が恒等に寄る」問題を抑えつつ（LNS等）([arXiv][1])
  * 「中間層の死層（低BI）」も残さない（BI‑Floor；BIはShortGPT由来）([arXiv][3])
* “層を削る”のではなく、**同サイズのまま層の価値を上げる学習レシピ**として提示できる

---

## 9. 最短ToDo（実装〜主結果までの最短経路）

1. Phase 0：Pre‑LNで **BI/層抜き/分散/勾配**をログして現象を固める
2. LNS実装（DepthScaledNorm差し替え）＋Phase 1で安定性と分散改善を確認([arXiv][1])
3. BI‑Floor（中間層）を足して Phase 2（手法7）を回す([arXiv][3])
4. 主図：

   * 中間層の `frac(BI<tau)` が下がる
   * 中間層の層抜き (\Delta)PPL が増える（抜けない）
   * PPLが維持/改善
5. 余力で Mix‑LN/Peri‑LN を副系として追加（一般性）

---

[1]: https://arxiv.org/pdf/2502.05795 "The Curse of Depth in Large Language Models"
[2]: https://arxiv.org/html/2502.02732v1 "Peri-LN: Revisiting Layer Normalization in the Transformer Architecture"
[3]: https://arxiv.org/abs/2403.03853?utm_source=chatgpt.com "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"
