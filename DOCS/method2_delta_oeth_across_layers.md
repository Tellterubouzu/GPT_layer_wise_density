# 手法2：Δ-orth across layers（層差分の隣接直交化）— 実験計画 & 実装タスク（Markdown）

> 目的：GPT系（decoder-only Transformer）で観測される **中間層の冗長性**（層を抜いても性能が落ちにくい／層間表現が似すぎる）を、**「層が residual stream に書き込む差分（Δ）」の重複**として捉え、**隣接層のΔどうしを直交化（低相関化）する正則化**で冗長性を減らす。

---

## 0. 背景（論文導入に使える要点）

### 0.1 GPT系LLMの「層冗長性」の代表的観測

* LLMは層間の類似性が高く、**寄与が小さい（冗長な）層が存在**するという報告があり、層入力と層出力の類似に基づく **Block Influence（BI）** を用いた層除去（ShortGPT）が提案されている。([arXiv][1])
* 「深い層が効きにくい」現象を **Pre-LNが引き起こす分散増大・ヤコビアンの恒等化**として説明する “Curse of Depth” もあり、**LLMで半分近い層が期待ほど有効でない**ことを多モデル族で確認したとする。([arXiv][2])

### 0.2 冗長性削減の先行アプローチ（あなたの研究の位置づけ）

* **事後圧縮（削る）**：ShortGPTはBIで冗長層を同定して削る（後処理・圧縮が主目的）。([arXiv][1])
* **学習幾何の制約（方向を変える）**：ORU（Orthogonal Residual Update）は、残差更新が入力stream方向に偏る問題意識から、モジュール出力を入力streamに直交な成分だけ加える設計を提案（主に画像系で検証）。([arXiv][3])
* **（広義の）冗長性低減正則化**：DeCovは活性の相関（共分散）を下げて表現の冗長を抑える正則化として古典的。([arXiv][4])
* **表現類似の測定**：層間表現の「似ている／似ていない」を測る指標として CKA が広く使われる。([Proceedings of Machine Learning Research][5])

---

## 1. 本手法（Δ-orth across layers）の中核アイデア

### 1.1 Δ（層が書き込む差分）の定義

GPTブロック境界での hidden state を (h_\ell)（ブロック (\ell) の入力）、(h_{\ell+1})（ブロック (\ell) の出力）とすると、

[
\Delta_\ell := h_{\ell+1} - h_\ell
]

これは「ブロック (\ell) が residual stream に書き込んだ総差分」です。

> 直感：冗長な中間層は、**(\Delta_\ell) が（隣の層の）(\Delta_{\ell-1}) と同方向に近い**＝同じ種類の“上書き/微修正”を繰り返している可能性がある。

### 1.2 正則化（隣接Δの直交化）

隣接層の差分の cosine 類似度を用いて、

[
\cos(\Delta_\ell,\Delta_{\ell-1})=\frac{\langle \Delta_\ell,\Delta_{\ell-1}\rangle}{|\Delta_\ell||\Delta_{\ell-1}|+\epsilon}
]

として、以下のように **cos² を罰する**（符号の影響を消し「平行/反平行」をまとめて抑える）：

[
\mathcal{L}*{\Delta\text{-orth}}=\sum*{\ell\in\mathcal{M}} \mathbb{E}*{b,t}\big[\cos(\Delta*\ell,\Delta_{\ell-1})^2\big]
]

* (\mathcal{M})：適用する層集合（まずは **中間層のみ**）
* (\epsilon)：数値安定化（例：1e-6）
* 期待：中間層で **Δの重複が減り、層ごとの“役割分担”が進む**

### 1.3 最終損失

[
\mathcal{L}=\mathcal{L}*{LM}+\lambda*{\Delta},\mathcal{L}_{\Delta\text{-orth}}
]

* (\lambda_{\Delta})：正則化重み（後述のスケジュール推奨）

---

## 2. 先行研究との関係と「新規性」の主張

### 2.1 先行研究との差分（論文で強調）

* **ShortGPT**：冗長層を見つけて削る（事後圧縮）。あなたの手法は **学習中に冗長性が“生まれにくい”ように誘導**し、同サイズでも「中間層が効く」状態を作るのが狙い。([arXiv][1])
* **ORU**：各層の更新を入力streamに直交化する（Δ ⟂ h）。あなたの手法は **“更新どうし（Δ ⟂ Δprev）”** を狙う。つまり ORU と **直交する（別軸の）冗長性抑制**。([arXiv][3])
* **DeCov/Barlow Twins/VICReg 系**：表現の冗長性（相関）を抑える思想はあるが、多くは「同一層内の次元間」や「ビュー間」中心。あなたは **“層の書き込み（Δ）”という操作量**に直接正則化をかけ、しかも **中間層に局所化**する。([arXiv][4])

### 2.2 新規性（書ける形）

* **Δ-orth across layers**：GPT系の中間層で、**連続するブロックの書き込み差分 (\Delta_\ell) を直交化する正則化**を提案。
* **冗長性“指標”での主評価**：BI（ShortGPT）、層抜き感度、Δ類似度、CKA類似度など、**冗長性を直接測る指標で改善を主張**する。([arXiv][1])

---

## 3. 研究仮説（実験目的を明確化）

### H1：Δ類似度が下がる（狙っている現象そのもの）

* 中間層で (\cos(\Delta_\ell,\Delta_{\ell-1})) の分布が 0 近傍に寄る（平均が下がる、裾が短くなる）。

### H2：中間層の冗長性が減る（BIと層抜きで示す）

* 中間層の BI が上がり（入出力がより変化）、低BI層が減る。([arXiv][1])
* 中間層を1層スキップしたときの PPL 悪化が増える（「抜ける層」が減る）。

### H3：言語モデリング性能を維持（できれば改善）

* 同条件で val PPL が同等以上。過剰な正則化で悪化しない範囲を示す。

---

## 4. 実験計画（段階的：小→中→スケール）

> まずは「現象が動くこと」を小規模で固め、その後にスケールと一般性を取りにいく構成です。

### 4.1 実験A：Sanity（学習が壊れない＋Δ指標が動く）

**目的**

* 実装検証（勾配が流れる、NaNが出ない）
* Δ-orth のログが期待通り動く（cos²が下がる）

**設定（例）**

* モデル：GPT-2 small級（またはnanoGPT相当）
* データ：OpenWebTextの小サブセット等
* 学習：短期（数千〜数万step）

**必須ログ**

* 層別：`mean_cos_delta_adjacent`, `p90_cos_delta_adjacent`
* 層別：`||Δ||`（更新がゼロに潰れてないか）
* train loss / val PPL

---

### 4.2 実験B：主結果（冗長性が減ったことを“指標で”示す）

**目的**

* 「中間層冗長性が減る」を、BI・層抜き感度・Δ類似度で主張する

**評価（主）**

1. **BI（Block Influence）**：ShortGPTに基づく層別指標。([arXiv][1])
2. **層抜き感度**：層スキップでの (\Delta PPL_\ell)
3. **Δ類似度**：(\cos(\Delta_\ell,\Delta_{\ell-1})) の層別分布
4. **層間表現類似（任意だが強い）**：CKAで隣接層が似すぎていないか。([Proceedings of Machine Learning Research][5])

**比較**

* baseline（正則化なし） vs Δ-orth（中間層適用）

---

### 4.3 実験C：スケールアップ（一般性）

* モデル：Pythia 410M、GPT-NeoX系小モデル、LLaMA系小モデルなど（手元の訓練基盤に依存）
* データ量を増やす
* 結果：Δ類似度低下 → BI/層抜き変化 → PPL維持 の再現

---

## 5. ベースライン・アブレーション（論文の説得力を作る）

### 5.1 必須ベースライン

* **BL-0**：標準GPT（正則化なし）
* **BL-1**：Δ-orth を全層に適用（「中間層だけ」が効く主張の対照）
* **BL-2**：計算量対照（同程度の計算をするが損失に足さないダミー計算）

### 5.2 重要アブレーション（本質検証）

1. **適用範囲**

   * 中間層のみ（推奨） vs 後半のみ vs 全層
2. **Δの定義粒度**

   * ブロック境界Δ（まずこれ）
   * Attention後Δ / MLP後Δ（2箇所の residual add ごと）
3. **勾配の流し方**

   * `detach_prev=True`（前のΔを固定ターゲットにして安定化）
   * `detach_prev=False`（両方に勾配を流し、より強い相互作用）
4. **正則化形**

   * `cos^2`（基本）
   * `hinge`：(\max(0, |\cos|-c)^2)（過剰に直交を強制しない）
5. **λスケジュール**

   * 固定 vs warmup後に線形増加（推奨）

### 5.3 関連比較として入れると綺麗（任意）

* ShortGPT（事後削除）との対比：**“削る”のではなく“育てる”**方針
* ORU（Δ ⟂ h）との対比：**Δ ⟂ Δprev** の別軸性 ([arXiv][3])

---

## 6. 実装計画（どんなコードが必要か：階層チェックリスト）

### 6.1 まず決める：実装方式（おすすめ順）

#### 方式A（最初の実験に最適）：`output_hidden_states=True` を使う

* Hugging Face Transformers なら `model(..., output_hidden_states=True)` で層ごとの `hidden_states` が取れる
* **実装が最短**だが、**メモリ負荷が増える**（スケール実験で厳しくなる）

#### 方式B（スケール向け）：forward hook でブロック入出力だけ取る

* 各ブロックの `input` と `output` からΔを逐次計算し、**保存せずに正則化を蓄積**
* gradient checkpointing と相性があるので注意（後述）

#### 方式C（いちばん綺麗）：ブロック実装に「正則化蓄積」を組み込む

* `Block.forward()` 内で Δ を計算し、Contextに `reg_loss += ...` として積む
* 大規模でもメモリ効率が良い

---

### 6.2 コア実装①：Δ-orth loss（`src/losses/delta_orth.py`）

#### 関数シグネチャ案

```python
def delta_orth_loss(
    delta: torch.Tensor,        # [B, T, D] or [B, D]
    prev_delta: torch.Tensor,   # same shape
    eps: float = 1e-6,
    detach_prev: bool = False,
    reduce_tokens: str = "mean",  # "mean" | "sample_k"
    sample_k: int = 32,
) -> torch.Tensor:
    ...
```

#### 実装の要点（トークンごと版）

* `dot = (delta * prev).sum(-1)` → `[B,T]`
* `cos = dot / (norm_delta * norm_prev + eps)`
* `loss = (cos**2).mean()`

> **コスト削減オプション**（推奨で組み込み）

* `sample_k`：各系列からKトークンだけランダム抽出して cos² を計算（BTD→BKD）
* `pool`：`delta.mean(dim=1)` で `[B,D]` にして計算（最安、ただし粗い）

---

### 6.3 コア実装②：Δの抽出（`src/models/delta_capture.py`）

#### 方式A：hidden_statesからΔを作る（最短）

```python
# hidden_states: tuple length L+1, each [B,T,D]
deltas = [hidden_states[i+1] - hidden_states[i] for i in range(L)]
```

#### 方式B：hookで逐次（スケール向け）

* ループでブロックを回す構造（nanoGPT等）なら、ブロック毎に

  * `h_in = h`
  * `h_out = block(h)`
  * `delta = h_out - h_in`
  * `reg += delta_orth_loss(delta, prev_delta)`
  * `prev_delta = delta`
    の形にできる（hidden_statesを保持しない）

#### 方式C：HFでhook（注意）

* HFの内部でブロックが呼ばれるので `register_forward_hook` を付ける
* hook内で `input[0]` と `output[0]` を取り、Δを計算
* **注意**：gradient checkpointing有効時に hook が複数回呼ばれる／順序が変わる可能性がある
  → 最初は checkpointing なしで固め、後から対応するのがおすすめ

---

### 6.4 学習ループ統合（`src/train/train_clm.py`）

#### 設定項目（configに持つ）

* `delta_orth.enabled: bool`
* `delta_orth.lambda_max: float`
* `delta_orth.lambda_schedule: linear_ramp | constant`
* `delta_orth.warmup_steps: int`
* `delta_orth.mid_start, mid_end`
* `delta_orth.detach_prev`
* `delta_orth.token_reduce: mean|sample_k|pool`
* `delta_orth.sample_k`
* `eps`

#### λスケジュール（おすすめ）

* **LM学習初期はΔが大きく揺れる**ので、最初から強い直交制約をかけると学習が壊れやすい
* 例：`warmup_steps` までは λ=0、その後 `ramp_steps` で 0→λmax を線形増加

---

### 6.5 評価コード（`src/eval/`）

#### 6.5.1 PPL（`ppl.py`）

* 通常の次トークン予測 perplexity

#### 6.5.2 Δ診断（`delta_stats.py`）

* 層別 `mean_cos(Δℓ,Δℓ-1)`, `p90`, `p99`
* 層別 `||Δ||` と `||Δ||` のトレンド

#### 6.5.3 BI（`bi_metric.py`）

* ShortGPTのBI（層入出力類似ベース）を同条件で算出し、**中間層の低BIが減るか**を示す。([arXiv][1])

#### 6.5.4 層抜き感度（`layer_drop.py`）

* 層 (\ell) をスキップして PPL 差分を計測
* 期待：Δ-orth で **中間層のスキップが効きやすくなる**（＝必要になる）

#### 6.5.5 層間表現類似（任意：`cka.py`）

* Kornblith et al. の CKA を用いて隣接層の類似を測る。([Proceedings of Machine Learning Research][5])

---

### 6.6 テスト（`tests/`）

* `test_delta_orth_loss_shapes.py`：入出力shape、dtype（fp16/bf16/fp32）
* `test_no_nan.py`：normが小さいケースで NaN が出ない（eps効いてる）
* `test_grad_flow.py`：`detach_prev` on/off で勾配が期待通り（前層へ流れる/流れない）
* `test_hook_order.py`（hook方式の場合）：層順序が崩れていないか（簡易）

---

## 7. 解析・可視化（論文の図になりやすい）

### 7.1 主図（冗長性低減の証拠）

1. 層別 **BI曲線**（baseline vs Δ-orth）([arXiv][1])
2. 層別 **層抜き感度**（(\Delta PPL_\ell)）
3. 層別 **Δ類似度**（mean/p90 cos）

### 7.2 メカニズム図（効いた理由を説明）

* Δ-orth の有無で、**中間層のΔ方向が“揃っている”→“散っている”**に変わることを、

  * cos分布
  * 主成分（PCA）での方向多様性
  * CKAの隣接層類似
    などで補強（CKAは説得力が出やすい）。([Proceedings of Machine Learning Research][5])

---

## 8. 想定される失敗モードと対策

### 8.1 PPLが悪化する（過剰制約）

* **対策**

  * λを下げる／rampを長くする
  * hinge化（(|\cos|>c) の時だけ罰）
  * 適用範囲を狭める（中間のさらに中心だけ）
  * `detach_prev=True` で安定化

### 8.2 Δが小さくなりすぎて“何もしない層”が増える（逆効果）

* **兆候**：`||Δ||` が中間で急減
* **対策**

  * `cos^2` だけでなく、`||Δ||` が一定以下にならないよう軽い下限制約（ただしこれは手法拡張になるので、まずは観測→必要ならアブレーションとして追加）
  * 正則化を「隣接だけ」ではなく「平均との差分」と併用（後日拡張）

### 8.3 hookが不安定（checkpointing等）

* 最初は `output_hidden_states` 方式で固める
* スケール段階でブロック内実装に移行（方式C）

---

## 9. 論文の貢献（Contributions）テンプレ

1. **Δ-orth across layers**：GPT系の中間層において、ブロックの書き込み差分 (\Delta_\ell) を隣接差分 (\Delta_{\ell-1}) と直交化する正則化を提案し、中間層の更新重複を抑える。
2. **冗長性指標での定量評価**：ShortGPTのBI、層抜き感度、Δ類似度、CKA類似度を用いて「中間層冗長性の減少」を直接示す。([arXiv][1])
3. **実用的な実装**：hidden_states方式→逐次計算方式へと移行可能な軽量実装を提示し、学習コスト増を抑えつつ効果を再現する。
4. **先行との補完性**：層削除（ShortGPT）や stream直交（ORU）とは異なる軸の冗長性抑制であることを議論。([arXiv][1])

---

## 10. 最短ロードマップ（次にやる順）

1. **方式A**（hidden_states）でΔ抽出 & `delta_orth_loss()` を実装
2. 小規模学習で **Δ類似度が下がる**ことだけまず確認（Sanity A）
3. 同じチェックポイントで **BI・層抜き感度**を計測（主結果Bの骨格）([arXiv][1])
4. ablation：`detach_prev`、適用範囲、hinge化
5. スケールアップ時に **逐次計算（方式B/C）**へ移行してメモリを落とす

---

必要なら、あなたのコードベース（HF Transformers / nanoGPT / Megatron-LM など）を前提に、

* 「どのクラスのどのforwardでΔを取るか」
* 「gradient checkpointingがあるときのhook回避策」
* 「層抜き評価を最小差分で実装するパッチ案」
  まで、**差分パッチ（擬似diff）形式**で具体化して書けます。

[1]: https://arxiv.org/abs/2403.03853?utm_source=chatgpt.com "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"
[2]: https://arxiv.org/html/2502.05795v3?utm_source=chatgpt.com "The Curse of Depth in Large Language Models"
[3]: https://arxiv.org/html/2505.11881v1?utm_source=chatgpt.com "Orthogonal Updates for Stable and Efficient Deep Networks"
[4]: https://arxiv.org/abs/1511.06068?utm_source=chatgpt.com "Reducing Overfitting in Deep Networks by Decorrelating Representations"
[5]: https://proceedings.mlr.press/v97/kornblith19a.html?utm_source=chatgpt.com "Similarity of Neural Network Representations Revisited"
