# ORU-Mid（中間層のみ直交残差更新）実験計画 & 実装タスクリスト（Markdown）

> 目的：GPT系（decoder-only Transformer）の**中間層で生じる層冗長性（層入出力が似すぎる／層を抜いても性能が落ちにくい）を「減らす」**ために、**中間層だけ residual update を直交化**して「各層が residual stream に新しい方向の情報を書き込む」ことを促す。

---

## 0. 背景・問題設定（論文導入に使う要点）

### 0.1 LLMの層冗長性の代表的観測

* **LLMの層は高い類似性を持ち、一部層は機能への寄与が小さい**という報告があり、層入出力の類似度から層重要度を測る **Block Influence (BI)** が提案されている。([arXiv][1])
* BIの定義は（論文内）「層入力と層出力の平均cos類似度」を用い、**BIが低い層ほど入出力が近く冗長**という立て付け。

### 0.2 なぜ“中間層”が冗長になりやすいのか（関連づけ）

* Pre-LN Transformer（GPT/LLaMA系で一般的）では、深部層が学習に寄与しにくくなる現象を **Curse of Depth** として整理し、原因を **Pre-LNによる出力分散の深さ方向の増大→深部ブロックの微分が恒等に近づく**ことに置く研究がある。([arXiv][2])
* 同系統として、Pre-LN/Post-LNのハイブリッドで深部・中間層の勾配を健康化し「深部が効く」方向を狙う Mix-LN もある（層品質・表現多様性の改善を主張）。
* LN戦略が hidden-state redundancy に影響する、という方向の報告（Peri-LN）もある。([arXiv][3])

### 0.3 ORU（Orthogonal Residual Update）とは

* 画像系（ResNetV2 / ViT）で提案された **Orthogonal Residual Update (ORU)** は、通常の残差更新 (x_{n+1}=x_n+f(x_n)) が **stream方向（平行成分）を強めるだけになりがち**という問題意識から、**モジュール出力を入力streamに対し分解して直交成分のみ加算**する。
* ORUの実装アルゴリズムと、数値安定のための (\epsilon)（例：(10^{-6})）なども提示されている。
* 同論文は、残差更新の幾何（cos・norm等）を診断し、学習安定性・汎化改善を報告している。

---

## 1. 本研究の新規性（論文の主張の核）

### 1.1 ギャップ（既存研究がやっていない/弱い点）

* ShortGPT等は「冗長な層を**見つけて削る**」が主目的。([arXiv][1])
* Curse of Depth / Mix-LN / Peri-LN は「深部が効かない」を **正規化・スケーリングで改善**する方向。([arXiv][2])
* 一方で **「中間層が冗長になりにくい学習幾何を与える」**という、**層の“書き込み方向”そのものを制約**するアプローチは、GPT系でほぼ未開拓（少なくとも標準的な手法として確立していない）。

### 1.2 新規性（ORU-Midの位置づけ）

* ORUを **decoder-only LLM（GPT系）へ移植**し、しかも **中間層だけ**に限定適用することで、

  * 序盤（語彙・局所特徴）と終盤（出力整形）の安定性を維持しつつ
  * 中間層に「新しい方向の更新」を強制して冗長性を抑える
    という設計を提案する点が新しい。
* さらに、冗長性を **BI分布・層抜き感度・層間表現類似**のような“冗長性指標”で直接評価して、

  * **冗長性が減る（＝中間層が必要になる）**
  * その上で **言語モデリング性能も維持/改善**
    を示す構成が論文として強い。

---

## 2. 手法1：ORU-Mid（中間層のみ直交残差更新）

### 2.1 対象：GPT系ブロックの residual update

GPT系は通常、各ブロックで **(A) Attention residual** と **(B) MLP residual** の2回の加算を持つ。

* 通常：
  [
  h \leftarrow h + \Delta_{\text{attn}},\quad
  h \leftarrow h + \Delta_{\text{mlp}}
  ]
* ORU：各 (\Delta) を、加算前の stream (h) に対して直交化してから足す。

### 2.2 直交化の定義（実装に直結）

トークンごと（shape: ([B,T,D])）に、特徴次元 (D) 上で射影：

* 平行成分：
  [
  \Delta_{\parallel} = \frac{\langle \Delta, h\rangle}{\langle h,h\rangle+\epsilon} h
  ]
* 直交成分：
  [
  \Delta_{\perp} = \Delta - \Delta_{\parallel}
  ]
* 更新：
  [
  h \leftarrow h + \Delta_{\perp}
  ]

ORU論文のアルゴリズム表現・(\epsilon)設定の考え方（例：(10^{-6})）は踏襲する。

### 2.3 「中間層のみ」の定義（最初の実験では固定でOK）

全層数を (L) として、例：

* `mid_start = floor(L/3)`
* `mid_end   = ceil(2L/3)`
* 適用層集合：(\ell \in [\text{mid_start}, \text{mid_end}))

※後でアブレーションで範囲を動かす（§5参照）。

---

## 3. 研究仮説（実験目的を明確化）

### H1（冗長性低減：指標で示す）

ORU-Mid により、中間層の **BIが上がる（入出力がより変化する）**、および **低BI層の比率が減る**。

* BIは ShortGPT の定義に従って算出（下記）。

### H2（層の必要性が増す：介入で示す）

ORU-Mid モデルでは、中間層を1層落としたときの性能劣化（PPL上昇）が、ベースラインより大きい
→ **「抜いても大丈夫」な層が減る**（冗長性が減った）ことを示す。

### H3（性能維持 or 改善）

同じ学習トークン量/計算量で、PPL・下流評価が同等以上。
（ORU論文では汎化・安定性が改善した報告があるため、LMでも改善の余地。）

---

## 4. 実験計画（段階的：小→中）

> ここはそのまま「実験セクション」構成に流用できるように書いています。

### 4.1 実験セットA：最小構成（バグ潰し＋現象確認）

#### 目的

* ORU-Mid 実装が正しく動くこと
* 冗長性指標（BI・類似度）が期待通り動くこと
* 学習が発散しないこと

#### 設定案

* モデル：GPT-2 small級（層数がそこそこあり、中間層が定義しやすい）
* データ：小規模コーパス（OpenWebText相当のサブセット等）
* 学習：短い事前学習（“差が見える最短”でOK）

#### 観測（必須ログ）

* `cos(h, Δ)` の層別平均（ベースラインは中間層で大きめに偏る/ORUは0近傍に寄るはず）
* `||h||, ||Δ||, ||Δ_perp||` の推移
* train loss / val PPL

### 4.2 実験セットB：冗長性評価を“主結果”として出す

#### 目的

* ORU-Midが「中間層冗長性」を下げることを定量で示す

#### 指標（主）

1. **Block Influence (BI)**：ShortGPTの定義で層ごとに算出
   [
   BI_i = 1 - \mathbb{E}*{X,t}\frac{\langle X*{i,t}, X_{i+1,t}\rangle}{|X_{i,t}|*2|X*{i+1,t}|_2}
   ]
   （PDF中の式(1)に対応）

* 期待：中間層のBIが上がる／低BI層が減る

2. **層抜き感度（Layer removal sensitivity）**

* 各層 (i) を1層だけスキップして PPL差分 (\Delta PPL_i) を測る
* 期待：中間層の (\Delta PPL_i) が増える（=必要性が上がる）

3. **層間表現類似（cos/角度距離）**

* 隣接層の hidden state の角度距離（Mix-LNが表現多様性評価で使うのと近い見せ方ができる）
* 期待：中間層付近の類似が下がり、距離が上がる

#### 指標（従）

* val PPL（言語モデリング）
* 可能なら簡易下流（ゼロショットの一部、もしくは軽量SFTでの性能）

### 4.3 実験セットC：スケールアップ＆一般性

* モデルサイズを上げる（例：GPT-NeoX系の小モデル、Pythia 410M級など）
* データ量を増やす
* ORU-Midの効果が「小型だけの現象ではない」ことを示す

---

## 5. ベースライン・アブレーション設計（論文の説得力を作る）

### 5.1 ベースライン（最低限）

* **BL-0：標準GPT（Pre-LN）**
* **BL-1：ORUを全層に適用**（「中間だけ」の必要性を示す比較）
* **BL-2：中間層だけ“何もしない追加演算”**（計算オーバーヘッドの対照）

  * 例：同じ回数のdot/denom計算はするがΔは変えない（ダミー）

### 5.2 主要アブレーション（ORU-Midの中身を分解）

1. **適用箇所**

   * `attnのみ` / `mlpのみ` / `attn+mlp両方`
2. **適用範囲**

   * `[L/4, 3L/4]` vs `[L/3, 2L/3]` vs 中間の狭い帯
3. **直交化の粒度（ORU論文の “reduction set D” に相当）**

   * per-token feature-wise（推奨）：D={hidden_dim}
   * global（tokenも含めて射影）：D={T, hidden_dim}（※要検討・副作用が出やすい）
4. **(\epsilon)**

   * (10^{-6})（ORU論文推奨寄り）を中心に、(10^{-5},10^{-8})等

### 5.3 比較対象として入れると綺麗（任意）

* 正規化改善系（Mix-LN, あるいは Curse of Depth の LayerNorm Scaling）と組み合わせ可否

  * 「ORU-Midは正規化変更と独立（orthogonal）に効く」主張ができると強い。

---

## 6. 実装計画（どんなコードが必要か：階層チェックリスト）

### 6.1 リポジトリ構成案（例）

```
oru-mid-llm/
  configs/
    gpt2_baseline.yaml
    gpt2_oru_mid.yaml
  src/
    models/
      ortho.py
      gpt2_oru.py
    train/
      train_clm.py
      optimizer.py
      schedulers.py
    eval/
      ppl.py
      layer_drop.py
      redundancy_metrics.py
    utils/
      logging.py
      seed.py
      hooks.py
  scripts/
    run_train.sh
    run_eval.sh
  tests/
    test_ortho.py
    test_gpt2_forward.py
```

---

## 6.2 コア実装①：直交化ユーティリティ（`src/models/ortho.py`）

### 必須関数：`orthogonalize(delta, stream, reduce_dim=-1, eps=1e-6, fp32=True)`

* 入力：

  * `stream`：加算前の residual stream（例：`hidden_states`） shape `[B,T,D]`
  * `delta`：モジュール出力（attn_out / mlp_out） shape `[B,T,D]`
* 出力：

  * `delta_perp`：直交化された更新

#### 疑似コード（実装の要点）

```python
def orthogonalize(delta, stream, eps=1e-6, fp32=True):
    # stream, delta: [B, T, D]
    if fp32:
        s = stream.float()
        d = delta.float()
    else:
        s, d = stream, delta

    dot = (d * s).sum(dim=-1, keepdim=True)          # <d, s>
    denom = (s * s).sum(dim=-1, keepdim=True) + eps  # <s, s> + eps
    parallel = dot / denom * s
    d_perp = d - parallel

    return d_perp.to(delta.dtype)
```

#### 実装注意（バグりやすいポイント）

* **mixed precision**：射影はfp32推奨（denomが小さいとbf16/fp16で不安定）
* `stream` がゼロに近い場合：`eps`必須（ORU論文も安定性のための(\epsilon)に言及）
* 勾配：`stream` を detach しない（まずは素直に微分可能なまま）

---

## 6.3 コア実装②：Residual Add を置き換えるモジュール（任意だが便利）

### `class OrthogonalResidualAdd(nn.Module)`

* 設定：

  * `eps`
  * `fp32_proj`
  * `enabled_layers: set[int]`
  * `apply_to = {"attn","mlp"}`

* forward：

  * `return stream + orthogonalize(delta, stream)`

→ GPT2/LLaMA/NeoX など実装差があっても「加算箇所」さえ捕まえれば同じ部品で適用できる。

---

## 6.4 コア実装③：GPTブロックの forward 改変（`src/models/gpt2_oru.py`）

### 方針A（推奨）：Hugging Face Transformers を継承して差分最小

* `GPT2Block` 相当を継承して、**residual add の直前**で ORU を挟む
* 中間層判定：`layer_idx` を持たせる（作成時に注入）

#### 変更点（概念）

* Attention後：

  * `h = h + attn_out` → `h = oru_add(h, attn_out)`（中間層なら）
* MLP後：

  * `h = h + mlp_out` → `h = oru_add(h, mlp_out)`（中間層なら）

### 方針B：モデル生成時に “対象層だけ置換”

* `for i, block in enumerate(model.transformer.h):`

  * 中間層なら `block.resid_add = OrthogonalResidualAdd(...)` みたいに差し替え
* メリット：既存実装との衝突が少ない

---

## 6.5 学習コード（`src/train/train_clm.py`）

### 実装が必要な要素

* config読込（yaml/argparse）

  * `oru.enabled: bool`
  * `oru.mid_start, oru.mid_end`
  * `oru.apply_to: attn|mlp|both`
  * `oru.eps, oru.fp32_proj`
* 再現性

  * seed固定、determinism設定

### ログ（`src/utils/logging.py`）

* 追加で記録したい統計（層ごと）

  * `cos(stream, delta)`（ORU前の値）
  * `||delta_parallel|| / ||delta||`（ベースラインで“平行成分が大きい”ことの証拠）
  * `||stream||, ||delta||, ||delta_perp||`
* これらは ORU論文が提示している診断（cos・normの追跡）をLLM版で再現するイメージ。

---

## 6.6 評価コード（`src/eval/`）

### 6.6.1 PPL評価（`ppl.py`）

* 通常のCausal LM評価（validation set で cross entropy → perplexity）

### 6.6.2 BI評価（`redundancy_metrics.py`）

* ShortGPTの BI 定義に従う（式(1)）。
* 実装手順：

  1. forwardで各層の hidden state `X_i` と `X_{i+1}` を保存（メモリ節約のためにバッチごと集約）
  2. tokenごとに cos 類似度を計算して平均
  3. `BI_i = 1 - mean_cos`

### 6.6.3 層抜き感度（`layer_drop.py`）

* 各層 i をスキップするラッパを作る：

  * forward内で `if i == drop_idx: return hidden_states` みたいにバイパス
* drop_idxをスイープして PPL の差分を取る
* 主張：

  * ORU-Midは中間層が“必要になる”→ dropの悪化が増える

---

## 6.7 テスト（`tests/`）※地味に重要

### 6.7.1 `test_ortho.py`

* `delta_perp · stream ≈ 0` が成り立つ（許容誤差つき）
* `stream=0` 近傍でも NaN が出ない（eps有効）
* dtype（bf16/fp16）でも動作する（fp32_proj有無で）

### 6.7.2 `test_gpt2_forward.py`

* 既存モデルと同じ I/O 形状・cache挙動（past_key_values 等）
* `oru.enabled=False` なら完全に一致（回帰テスト）

---

## 7. 解析・可視化（論文の図になる）

### 7.1 “冗長性が減った”を見せる図（必須候補）

1. 層別 BI 曲線（baseline vs ORU-Mid）
2. 層別 drop感度（(\Delta PPL_i)）
3. 層別 `mean cos(stream, delta)`（ORU前の診断として）

### 7.2 “なぜ効くか”の診断図（強い）

* 中間層だけ、`||delta_parallel||/||delta||` がベースラインで大きい → ORUがそこを削る
* これで「中間層の更新がstream方向に偏っていた」→「新方向の更新に変えた」という機構説明が作れる
  （ORU論文の主張と整合）。

---

## 8. 想定される失敗モード & 対策（実験が詰まる前に用意）

### 8.1 性能が落ちる（PPL悪化）

* **原因候補**：LLMでは “stream方向の微調整（スケール調整）” が必要だった
* **対策（ORU-Midの拡張、ただし手法1の範囲内で軽く）**

  * `apply_to="mlpのみ"` にする（MLPのほうが“表現変換”、Attnは壊すと痛い可能性）
  * 中間層範囲を狭める
  * epsやfp32_proj設定を見直す

### 8.2 学習が不安定（NaN/発散）

* projをfp32に固定
* 勾配クリップ
* (\epsilon) を上げる（ORU論文も安定性優先での(\epsilon)選択に言及）

---

## 9. 論文としての「貢献（Contributions）」テンプレ

> そのまま Paper Contributions に近い文章として使える形。

1. **ORU-Mid**：GPT系の中間層に限定して直交残差更新を導入し、中間層が residual stream に“新しい方向”を書き込むよう誘導する手法を提案。
2. **冗長性削減の実証**：ShortGPTのBI指標（層入出力類似に基づく）および層抜き感度により、中間層冗長性が減ることを定量的に示す。
3. **機構解析**：層別の更新幾何（cos・norm・平行成分比）をログし、冗長性低減が「更新方向の偏り是正」に起因することを示す（ORUの考え方をLLMへ拡張）。
4. **（任意）正規化改善との独立性**：Curse of Depth/Mix-LN/Peri-LN等の“層品質改善”文脈と比較し、ORU-Midが別軸で効く可能性を議論。([arXiv][2])

---

## 10. 次にあなたが着手すべき「最短ToDo」

1. `orthogonalize()` を実装 + 単体テスト（直交性・NaN耐性）
2. GPT2Block（または使用するGPT系実装）の residual add 2箇所に ORU を差し込む（中間層だけ）
3. 小規模学習で `cos(stream, delta)` と BI が動くことを確認
4. layer-drop評価（層スキップでPPL差分）を回す
5. ここまで揃ったら、アブレーション（attnのみ/mlpのみ、範囲変更）で論文の形にする

---

必要なら、あなたが使うベースコード（`transformers` の GPT-2 / GPT-NeoX / LLaMA系、もしくは nanoGPT 等）を前提に、**「どのクラスのどの行にフックを入れるか」レベルの実装手順**まで落として、差分パッチ形式（擬似diff）で書けます。

[1]: https://arxiv.org/abs/2403.03853?utm_source=chatgpt.com "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"
[2]: https://arxiv.org/abs/2502.05795 "[2502.05795] The Curse of Depth in Large Language Models"
[3]: https://arxiv.org/html/2502.02732v3?utm_source=chatgpt.com "Peri-LN: Revisiting Normalization Layer in the Transformer ..."
