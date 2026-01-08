# 手法4：Marginal Utility Regularization（MUR; 層の限界効用正則化）— 実験計画 & 実装タスク（Markdown）

> 目的：GPT系（decoder-only Transformer）における **中間層の冗長性（“何もしない層”が出る／抜いても性能が落ちにくい）**を、**「その層が損失をどれだけ下げたか」**という“目的関数に直結した量”で捉え、**限界効用が小さい層に学習圧力をかける**ことで中間層の寄与を増やし、冗長性を減らす。

---

## 0. 背景・関連研究（論文導入に使える要点）

### 0.1 LLMの層冗長性（レイヤ寄与が小さい層の存在）

* LLMの層には高い類似性があり、**寄与が小さい層（冗長層）が存在**することが報告され、層入力と層出力の類似度に基づく指標 **Block Influence（BI）** を定義して冗長層を削る ShortGPT が提案されています。([arXiv][1])
* また Pre-LN を主因として「深い層の導関数が恒等に近づいて寄与しにくい」現象を **Curse of Depth** として整理する研究もあります。([arXiv][2])

### 0.2 本手法の核となる「限界効用（marginal utility）」の根拠

本手法は、**ある層の出力変化が損失をどれだけ下げたか**を、**一次のテイラー近似（directional derivative）**で評価します。

* 多変数の一次テイラー展開では
  [
  f(x+\Delta x)\approx f(x)+\nabla f(x)^\top \Delta x
  ]
  が成り立ちます。([CS Princeton][3])
* さらに「任意方向の傾きが ‘勾配との内積’ になる」ことは、最適化（勾配法）の標準的な見方です。([cs231n.stanford.edu][4])

> これを Transformer の各層が residual stream に書き込む差分に適用し、“その差分が損失を下げる方向に働いたか”を測ります。

---

## 1. 手法4：Marginal Utility Regularization（MUR）の定義

### 1.1 記法（ブロック単位）

Transformer block（層）(\ell) の入力・出力を

* (h_\ell)：block (\ell) 入力（residual stream）
* (h_{\ell+1})：block (\ell) 出力

とし、層が書き込んだ差分（residual update）を
[
\delta_\ell := h_{\ell+1}-h_\ell
]
と定義します。

### 1.2 限界効用（Marginal Utility）の一次近似

ミニバッチ損失を (\mathcal{L}) とし、block 出力に対する勾配を
[
g_{\ell+1} := \frac{\partial \mathcal{L}}{\partial h_{\ell+1}}
]
とすると、一次近似より
[
\mathcal{L}(h_\ell+\delta_\ell) \approx \mathcal{L}(h_\ell) + \langle g_{\ell+1},\delta_\ell\rangle
]
なので、**損失を下げた量（近似）**は
[
u_\ell := -\langle g_{\ell+1},\delta_\ell\rangle
]
で測れます（(u_\ell>0) なら “その層の更新は一次近似で損失を下げる方向”、(u_\ell<0) なら逆）。([CS Princeton][3])

> これは「層が ‘損失を下げる方向’ にどれだけ動いたか」を、**目的関数に直結した形**で測る点が、BI（表現類似）との大きな違いです。([arXiv][1])

### 1.3 スケール依存を避ける（推奨：正規化版の効用）

生の (u_\ell) はスケールに依存するため、まずは **方向整合**（alignment）を目的にするのが安定です。

* **cos版（方向のみ）**：
  [
  a_\ell := -\frac{\langle g_{\ell+1},\delta_\ell\rangle}{|g_{\ell+1}||\delta_\ell|+\epsilon}\in[-1,1]
  ]

  * (a_\ell>0)：少なくとも一次近似では損失を下げる方向
  * (a_\ell\approx 1)：(\delta_\ell) が (-g) にほぼ平行（理想）
* **投影係数版（“-g方向へどれだけ進んだか”）**：
  [
  p_\ell := -\frac{\langle g_{\ell+1},\delta_\ell\rangle}{|g_{\ell+1}|^2+\epsilon}
  ]
  これは (\delta_\ell) を (-g) に射影したときの係数（“効果的ステップサイズ”）に相当します。([CS Princeton][3])

---

## 2. 正則化（“罰則化する”）の設計

### 2.1 中間層集合（最初は固定）

全層数 (L) に対して

* `mid_start = floor(L/3)`
* `mid_end   = ceil(2L/3)`
* (\mathcal{M}={mid_start,\dots,mid_end-1})

（後でアブレーションで動かす）

### 2.2 “フロア”型の罰則（推奨：Hinge）

中間層の限界効用が一定値 (\tau) を下回ると罰する：

* cos版（推奨の初期形）：
  [
  \mathcal{L}*{MUR}=\frac{1}{|\mathcal{M}|}\sum*{\ell\in\mathcal{M}}\max(0,\tau-a_\ell)
  ]
* 生の効用版（後で）：
  [
  \mathcal{L}*{MUR}=\frac{1}{|\mathcal{M}|}\sum*{\ell\in\mathcal{M}}\max(0,\tau-u_\ell)
  ]

総損失：
[
\mathcal{L}*{total}=\mathcal{L}*{LM}+\lambda\mathcal{L}_{MUR}
]

> 直感：**“損失を下げる方向に動けていない層”だけ起こす**（冗長層を減らす）ことを狙う。

### 2.3 重要：二階微分を避けるための「stop-gradient」

(g_{\ell+1}) は損失の勾配なので、これをそのまま (\mathcal{L}_{MUR}) に入れると **二階微分**が絡みやすくなります。実装では基本的に

* (g_{\ell+1}) は **detach**（定数扱い）
* (\delta_\ell) だけに勾配を流す

という “first-order credit assignment” として設計します。一次近似の根拠（テイラー/方向微分）とも整合します。([CS Princeton][3])

---

## 3. 先行研究との関係 & 新規性（論文での位置づけ）

### 3.1 ShortGPT（BI）との差

* ShortGPT：層入出力の類似から BI を定義し **冗長層を削除**する（事後圧縮）。([arXiv][1])
* MUR：**削除ではなく学習中に「損失を下げる寄与」を各層に割り当てる**。BIは表現類似だが、MURは **目的関数（損失）に直結**した層寄与を扱う点が新しい。

### 3.2 Curse of Depth（深部が効きにくい）との補完

* Curse of Depth は Pre-LN による深さ方向の学習不全を整理しており、MURはそれに対して

  * 「中間〜深部で ‘実際に損失を下げる方向に動けているか’ を監視」
  * 「動けていない層に学習圧力を与える」
    という **訓練ダイナミクス側の処方**として位置づけられます。([arXiv][2])

### 3.3 ORU（直交残差）との違い（幾何の違い）

* ORU は「入力stream方向への偏り」を抑えて新規方向の表現を促す（(\delta \perp h)）。([arXiv][5])
* MUR は「損失勾配方向への整合（(\delta) を (-g) に寄せる）」で、**目的関数に沿った“有用な更新”**を促す（(\delta) と (g) の関係）。
  → 幾何として **別軸**なので、後で組み合わせ可能（論文の展望に書ける）。

---

## 4. 研究仮説（この手法で何が起こるべきか）

### H1：中間層の「限界効用」が底上げされる

* ベースラインでは中間層で (u_\ell) や (a_\ell) が低い（または負）層が多い
* MURで (\tau) 未満の層が減り、層ごとの (a_\ell) 分布が右に寄る

### H2：冗長性指標が改善する

* BI（ShortGPT）で中間層の低BI層が減る、または BI の分布が改善する（相関の観点）。([arXiv][1])
* 層抜き（layer skip）感度で、中間層を抜いたときの劣化が増える（= “抜ける層”が減る）

### H3：言語モデリング性能（PPL等）を維持（できれば改善）

* 過剰制約にしなければ、val PPL は維持される（改善したら強い）

---

## 5. 実験計画（段階的：診断→小規模→スケール）

### 5.1 Phase 0：ベースライン診断（“中間層は本当に効用が低いのか？”）

**目的**

* MUR導入の前に、ベースラインGPTで **層別の限界効用プロファイル**を取る
* 「中間層の効用が低い／負になりやすい」ことを実証し、問題設定を強くする

**やること**

* 通常学習のチェックポイントで、以下を測る：

  * 層別 (a_\ell)（-cos）平均/分位点（p10/p50/p90）
  * 層別 (u_\ell) の分布（スケールが荒れるので参考扱い）
  * 層別 BI（比較用）([arXiv][1])
  * 層抜き感度（(\Delta)PPL）

**期待する図**

* `layer index` vs `mean(a_l)`：中間で谷が出る（仮説）

---

### 5.2 Phase 1：小規模でMURを回す（sanity + 主結果の骨格）

**モデル/データ**

* まずは GPT-2 small級・小データ（実装の確実性優先）

**比較**

* BL-0：通常学習
* MUR-cos：(a_\ell) に hinge
* （任意）MUR-proj：(p_\ell) に hinge
* （任意）MUR-raw：(u_\ell) に hinge（後回し推奨）

**主要評価**

* 冗長性：

  * BI曲線/分布（中間層）([arXiv][1])
  * 層抜き感度（中間層）
* 効用：

  * (a_\ell) の分布（中間層で (\tau) 未満が減るか）
* 性能：

  * val PPL

---

### 5.3 Phase 2：スケールアップ（一般性）

* モデルサイズを増やし、同様に

  * “中間層の効用が底上げされる”
  * “冗長性が減る”
  * “PPL維持”
    を確認する
* Curse of Depth が示す “深い層が効きにくい現象が広範に存在” と整合する議論に繋げる。([arXiv][2])

---

## 6. ベースライン & アブレーション（論文の説得力を作る）

### 6.1 必須ベースライン

* **BL-0**：正則化なし
* **BL-1**：MURを全層に適用（中間限定の意義を示す）
* **BL-2**：計算対照（同じ統計を計算するが損失に入れない）

### 6.2 重要アブレーション（“効く理由”を分解）

1. **層範囲**：中間のみ vs 後半のみ vs 全層
2. **粒度**：block境界のδ vs Attention後δ/MLP後δ
3. **効用定義**：cos版 (a_\ell) vs proj版 (p_\ell) vs raw (u_\ell)
4. **(\tau) と (\lambda)**：固定 vs warmup→ramp（後述）
5. **tokenの取り方**：全token平均 vs `sample_k` vs “高loss tokenのみ”
6. **stop-gradient戦略**：

   * `detach(g)=True`（推奨）
   * `detach(delta)=True`（効かないはず＝対照）

---

## 7. 実装計画（必要コード：詳細・階層）

> ここが一番重要です。**「まず計測→次にMUR」**の順で詰まらないように設計しています。

### 7.1 リポジトリ構成案（例）

```
mur-llm/
  configs/
    gpt2_baseline.yaml
    gpt2_mur_cos.yaml
    gpt2_mur_proj.yaml
  src/
    models/
      wrap_capture.py          # Δとh_outを捕まえる
      gpt_with_mur.py          # モデル差分（最小）
    mur/
      utility.py              # u_l / a_l / p_l 計算
      regularizer.py          # hinge / schedules
      hooks.py                # backward hook 管理
      optimizer_scaling.py    # (任意) 1-pass版のLR/gradスケール
    train/
      train_clm.py
      schedules.py
    eval/
      ppl.py
      mur_stats.py            # 層別効用の可視化用
      bi_metric.py            # ShortGPT BI算出
      layer_drop.py
    utils/
      logging.py
      seed.py
  tests/
    test_utility_math.py
    test_hook_capture.py
```

---

## 7.2 コア実装①：forwardで δ を捕まえる（Δキャプチャ）

### 7.2.1 取るべきテンソル

中間層 (\ell\in\mathcal{M}) について

* `h_in[ℓ]`：block入力（shape `[B,T,D]`）
* `h_out[ℓ]`：block出力（shape `[B,T,D]`）
* `delta[ℓ] = h_out - h_in`（shape `[B,T,D]`）

### 7.2.2 メモリ節約の推奨

* **正則化に必要なのは δ と g の内積**なので、学習の都合上は

  * `delta` は **detachして保持**（`delta_detached = (h_out - h_in).detach()`）
  * `h_out` は **hook用にテンソル参照を保持**（勾配を捕まえるため）
* これで “グラフ丸ごと保持” を避けやすい

### 7.2.3 実装パターン

* **パターンA（最短）**：HFならブロックを継承して `forward` の冒頭と末尾で `h_in/h_out` を取る
* **パターンB（差分最小）**：ブロックに `forward_hook` をつけて `input[0]` と `output[0]` を取る

  * ※checkpointing有効時の挙動に注意（最初はOFF推奨）

---

## 7.3 コア実装②：backwardで (g_{\ell+1}=\partial\mathcal{L}/\partial h_{out}) を捕まえる

### 7.3.1 hook（推奨）

`h_out.register_hook(hook_fn)` を使い、backward時に `grad` を受け取る：

```python
def save_grad_hook(layer_idx):
    def _hook(grad):
        grads[layer_idx] = grad.detach()
        return grad
    return _hook
```

* mixed precisionでも **内積計算はfp32**推奨：

  * `g = grad.detach().float()`
  * `d = delta[layer_idx].float()`

### 7.3.2 勾配蓄積（gradient accumulation）対応

* `accum_steps>1` の場合：

  * `u_layer_sum += u_layer_batch`
  * `count += 1`
  * optimizer step直前に平均を取る

---

## 7.4 コア実装③：限界効用の計算（`src/mur/utility.py`）

### 7.4.1 主要関数

* `utility_raw(g, delta)` → (u=-\langle g,\delta\rangle)
* `utility_cos(g, delta)` → (a=-cos(g,delta))
* `utility_proj(g, delta)` → (p=-<g,delta>/(||g||^2+\epsilon))

#### 疑似コード（cos）

```python
def utility_cos(g, d, eps=1e-6):
    # g,d: [B,T,D] fp32
    dot = (g * d).sum(dim=-1)                 # [B,T]
    ng  = (g * g).sum(dim=-1).sqrt()          # [B,T]
    nd  = (d * d).sum(dim=-1).sqrt()          # [B,T]
    cos = dot / (ng * nd + eps)
    a = -cos
    return a  # [B,T]
```

### 7.4.2 tokenの減らし方（計算コスト対策）

* `mean`：全token平均（まずこれ）
* `sample_k`：各系列からkトークンだけ抽出
* `topk_loss_tokens`：高loss token（hard token）だけ（後で）

---

## 7.5 正則化の組み込み方：2つの実装モード

### 7.5.1 モード1：**MUR-Loss（明示的な正則化項）** — 小規模向け（分かりやすい）

**課題**：(g) を得るには勾配が必要 → そのままだと循環
**解**：`g` は **LM損失の勾配から取り、detachして使う**（二階微分回避）。

#### 実装案（実用的）

1. forwardして `lm_loss` を作る
2. **`torch.autograd.grad` で、選んだ中間層の `h_out` に対する勾配 g を取る**（create_graph=False, retain_graph=True）
3. gをdetachし、δと内積して `mur_loss` を作る
4. `total_loss = lm_loss + λ*mur_loss` を `backward()`

> これは backward が実質2回走るので高コスト。
> しかし “論文の定義通りの loss 正則化” を小モデルで検証するには最も明快です。一次テイラー（directional derivative）に基づく定義とも整合します。([CS Princeton][3])

### 7.5.2 モード2：**MUR-Update（optimizer側で近似正則化）** — スケール向け（1 backward）

**狙い**：hookで得た (a_\ell) を使い、次ステップ以降の更新を調整（“遅い/無効な層”を起こす）

#### 代表的な2案

* **(A) 層別 LR multiplier**

  * (\bar a_\ell)（EMA）を持ち、低い層ほどLR倍率↑：
    [
    m_\ell = 1 + \alpha \cdot \max(0,\tau-\bar a_\ell)
    ]
  * optimizerのparam groupを層ごとに分け、`lr = base_lr * m_l`
* **(B) 層別 grad scaling**（より簡単）

  * optimizer.step前に、その層のパラメータ勾配に `m_l` を掛ける

> これは “lossに足す” のではなく “更新則に正則化を入れる” 形ですが、1 backwardで回せて現実的です。論文では **MUR-Loss（小規模で定義検証）→ MUR-Update（実用版）**という整理がしやすいです。

---

## 7.6 設定（config）とスケジュール（`src/train/schedules.py`）

### 7.6.1 必須パラメータ

* `mur.enabled`
* `mur.mode = loss | update`
* `mur.metric = cos | proj | raw`
* `mur.mid_start, mur.mid_end`
* `mur.tau`
* `mur.lambda_max`（lossモード）
* `mur.alpha`（updateモード）
* `mur.warmup_steps`, `mur.ramp_steps`
* `mur.sample_k`, `mur.eps`
* `mur.fp32_dot = True`

### 7.6.2 推奨スケジュール

* warmup中は MURを弱く/ゼロに（学習初期は勾配が荒い）
* `step>warmup` で線形に強める（λまたはα）

---

## 7.7 ロギング（論文の図になるログ）

最低限、**層別**で以下を保存：

* `mean(a_l)` / `p10(a_l)` / `frac(a_l<tau)`（中間層）
* `||delta||`（δがゼロに潰れていないか）
* `||grad||`（勾配の健康状態）
* `val_ppl`
* （任意）BIと相関を見る：`corr(a_l, BI_l)` ([arXiv][1])

---

## 7.8 評価コード（`src/eval/`）

### 7.8.1 perplexity

* `ppl.py`：通常のCausal LM評価

### 7.8.2 MUR統計

* `mur_stats.py`：

  * 層別の (a_\ell) 分布
  * (\tau) 未満の層比率
  * 進行に伴うプロファイル変化（学習ステップ別）

### 7.8.3 冗長性評価（必須）

* `bi_metric.py`：ShortGPT BI（層入出力cosベース）([arXiv][1])
* `layer_drop.py`：層スキップで (\Delta)PPL を計測（“抜ける層”が減ったか）

---

## 7.9 テスト（`tests/`）

* `test_utility_math.py`

  * (a_\ell\in[-1,1])
  * gやδがゼロに近いときNaNが出ない（eps）
* `test_hook_capture.py`

  * forwardでδが取れている
  * backwardでgradが取れている
  * accumulationでも平均が合う

---

## 8. 想定される失敗モード & 対策

### 8.1 “効用だけ上げる”ための無意味な回転が起きる

* (a_\ell) は方向だけなので、表現を回転させて整合させる抜け道があり得る
* 対策：

  * (\tau) を控えめに（「負を潰す」程度）
  * 中間層のみに限定
  * `proj版`（(p_\ell)）も比較し、方向だけでなく“進んだ量”も見る

### 8.2 逆に δ が小さくなりすぎて層が死ぬ

* hingeで「(\tau) を満たせばOK」にする（過剰に押さない）
* `||delta||` をログし、異常検知

### 8.3 計算コストが増える

* MUR-Loss（autograd.grad）は小規模のみに限定し、主スケールは MUR-Updateへ
* token sampling（sample_k）を導入

---

## 9. 期待する図・表（論文で強い）

1. **Layer index vs mean (a_\ell)**（baselineとMUR）
2. **Layer index vs BI**（baselineとMUR、あるいは相関）([arXiv][1])
3. **Layer drop sensitivity**（中間層を抜いた時の(\Delta)PPLが増える）
4. **学習曲線（val PPL）**：性能維持の確認
5. **コスト**：step time / メモリ増（MUR-Loss vs MUR-Update）

---

## 10. 論文用「貢献（Contributions）」テンプレ

1. **層の限界効用の定義**：一次テイラー（方向微分）に基づき、
   [
   u_\ell=-\langle \partial\mathcal{L}/\partial h_{\ell+1},; h_{\ell+1}-h_\ell\rangle
   ]
   を層の“損失低下寄与”の近似として導入。([CS Princeton][3])
2. **MUR（限界効用フロア正則化）**：中間層の (a_\ell)（または (u_\ell)）が閾値を下回る場合に罰則を与え、冗長層（寄与の小さい層）を減らす学習法を提案。
3. **冗長性削減の実証**：ShortGPTのBIや層抜き感度により、中間層冗長性の低下を定量的に示す。([arXiv][1])
4. **スケーラブルな実装**：明示loss版（MUR-Loss）に加え、1 backwardで動く更新則版（MUR-Update）を提示し、規模を上げても適用可能な設計を示す。
5. **既存説明との整合**：Curse of Depth が示す “深い層が効きにくい訓練ダイナミクス” を、層別効用プロファイルとして観測・改善する観点を提供。([arXiv][2])

---

## 11. 最短ToDo（まず何からやるべきか）

1. **Phase 0計測**：baselineで層別 (a_\ell) をログし、“中間層の効用不足”が本当に出るか確認
2. hookで `delta` と `grad` を取る基盤を実装（MURなしでOK）
3. `mur_stats.py` で層別分布の可視化（図の雛形を先に作る）
4. MUR-Update（1 backward）を先に入れて回す（効けばスケールへ）
5. 小モデルで MUR-Loss（定義に忠実）も検証し、両者の関係を整理して論文化

---


[1]: https://arxiv.org/abs/2403.03853?utm_source=chatgpt.com "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect"
[2]: https://arxiv.org/abs/2502.05795?utm_source=chatgpt.com "The Curse of Depth in Large Language Models"
[3]: https://www.cs.princeton.edu/courses/archive/fall18/cos597G/lecnotes/lecture3.pdf?utm_source=chatgpt.com "20 September 2018 3.1 Taylor series approximation"
[4]: https://cs231n.stanford.edu/2024/slides/2024/lecture_3.pdf?utm_source=chatgpt.com "Lecture 3: Regularization and Optimization"
[5]: https://arxiv.org/abs/2505.11881?utm_source=chatgpt.com "Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks"
