# Byteifying Apertus via Task Arithmetic

This note discusses applying the **Post-Training via Task Arithmetic** flow of this repo ([§Post-Training via Task Arithmetic](README.md)) to the **Apertus** model suite
(see [website](https://apertus-ai.org) and [tech report](https://arxiv.org/abs/2509.14233)).

The instructify mechanism (`src/examples/bolmo/instructify.py`) byteifies an existing post-trained checkpoint
**without any additional training**: the post-training delta is computed in parameter space and
added to the byteified backbone.

The byteification *recipe* itself (Stage 1 + Stage 2) is identical in structure to the [Qwen3-8B run](QWEN3_8B_TRAINING.md); this note focuses on what changes for Apertus and provides a
step-by-step implementation plan.

## Getting started

Apertus is a *fully* open suite: weights for
`swiss-ai/Apertus-8B-2509` (base) and `swiss-ai/Apertus-8B-Instruct-2509` (post-trained) plus the
70B pair, final and intermediate checkpoints, the post-training pipeline
([swiss-ai/posttraining](https://github.com/swiss-ai/posttraining)), the chat format
([swiss-ai/apertus-format](https://github.com/swiss-ai/apertus-format)), the SFT/alignment data,
and the eval suites. Task arithmetic needs exactly three checkpoints — byteified base, original
base, post-trained model — all obtainable.

The one-time cost is byteifying the *base* once (Stage 1 + Stage 2, ≈1% of pretraining budget). Every post-trained variant is then free: with a plain
parameter-space conversion of `Apertus-8B-Instruct-2509`, `instructify.py` produces the byteified
instruct model in minutes on CPU. If the intermediate SFT-only checkpoint (the report's
"Apertus-SFT", used to generate off-policy completions, §4.1.4) is released, the SFT delta
(ΔSFT = θ_SFT − θ_base) and the alignment delta (ΔQRPO = θ_instruct − θ_SFT) can be isolated and
applied separately — a unique research handle: byteified models with and without QRPO alignment,
or with a scaled alignment vector (α_QRPO < 1), at zero training cost.

**Multilingual fairness is Apertus's core motivation — byte-level modeling is its limit case.**
Apertus selected the Mistral-Nemo v3 *tekken* tokenizer precisely for its low Gini coefficient
(tokenization-cost inequality across languages, §2.2, Appendix H.3). Subword segmentation is
removed entirely in a byteified model, so the fairness metrics Apertus already computed
(fertility, compression ratio, vocabulary utilization, Gini) provide a ready-made framework to
quantify the byte-level gain. The post-training mixture deliberately includes low-resource
material (Romansh in six written varieties, Swiss-German dialects, African languages; §4.1.3) —
exactly the regimes where subword fertility is worst and byte-level operation should help most.

**Recipe synergy.** Apertus validated its architecture and recipe (xIELU, QK-Norm, AdEMAMix,
WSD) against OLMo2 in a matched setup (§2.4), i.e. the two ecosystems were designed to be
comparable. Apertus's pretraining-time Goldfish loss (2% token masking, k=50, h=50) suppresses
verbatim memorization in the backbone we would inherit; byte-level continuation is the classic
elicitation surface for memorized content, and [swiss-ai/apertus-memorization](https://github.com/swiss-ai/apertus-memorization)
provides the probes to test whether suppression survives byteification (§5.4, Appendix F).

**Convenient scale coincidence.** Apertus-8B has `d_model = 4096` — the same as Qwen3-8B — so the
local encoder/decoder hyperparameters from `QWEN3_8B_TRAINING.md` (local `d_model=4096`, local
FFN hidden 5504, 1 encoder / 4 decoder layers, xLSTM local models) transfer directly.

## Task-arithmetic byteification

`src/examples/bolmo/instructify.py` loads three olmo-core checkpoints:

| CLI flag | Contents |
|---|---|
| `--checkpoint-dir` | The **Bolmo** checkpoint (Stage 1 + Stage 2 output over the base model) |
| `--base-checkpoint-dir` | The original **base subword** model, converted from HF to olmo-core format |
| `--instruct-checkpoint-dir` | The **post-trained subword** model, converted from HF to olmo-core format |

It then merges, per parameter of the Bolmo model:

- **`teacher.*`** — copied verbatim from the instruct checkpoint (the post-trained subword model is
  stored alongside for olmo-core distillation/continued-training flows; it is *not* exported to the
  HF format, and generation uses the boundary predictor, not the teacher);
- **shared backbone keys** (`blocks.*`, norms) — `θ ← θ_bolmo + α · (θ_instruct − θ_base)`, the
  standard task-vector addition with `--alpha` (default 1.0). This is the entire "byteification
  without training": the post-trained delta lives in the backbone, which is parameter-identical
  between the subword and byteified model;
- **Bolmo-only keys** (local encoder/decoder, boundary predictor) — left untouched;
- **shape-mismatched keys** (e.g. the byte-level `lm_head` vs. the 131k-vocab subword head) —
  skipped with a warning. Instruction following must therefore be carried by the backbone, which is
  also why the head cannot absorb the subword `lm_head` delta;
- with `--instructify-embeddings`, the subword input-embedding task vector
  (`embeddings.weight`) is added into `local_encoder.expanded_embeddings.weight` — the local
  encoder's subword-vocab-sized embedding (same 131k shape, so the transfer is exact).

**Hard requirement:** the base and instruct subword checkpoints must be converted to olmo-core
format with the *identical* architecture config and weight mapping, so the task vector is
well-defined (same keys, same shapes). For Apertus this holds — base and Instruct share one
architecture — but it must be built first (Phase 1–2 below).

## Implementation plan

### Phase 0 — Prerequisites

- Checkpoints: `swiss-ai/Apertus-8B-2509`, `swiss-ai/Apertus-8B-Instruct-2509` (start with 8B; 70B
  is ~9× the compute and this repo has so far only demonstrated 8B-scale runs).
- Environment: as in `QWEN3_8B_TRAINING.md` ("Setting Up"; `uv sync --frozen --extra xlstm
  --extra wandb --extra eval`, flash-attention wheel, `datasets==4.8.5`, `huggingface-hub==0.34.0`).
- Training data: `allenai/bolmo_mix` + the `dolma` tool (same as the Qwen3 run).
- Verify the vendored transformers (4.57.3 per `uv.lock`) loads `ApertusForCausalLM` natively
  (Apertus support landed in transformers ≥ 4.56). If not, fall back to `trust_remote_code` with
  the HF repo's modeling files, and cross-check parameter names against
  [swiss-ai/hfconverter](https://github.com/swiss-ai/hfconverter).

### Phase 1 — Apertus architecture support in olmo-core

Target: a plain `TransformerConfig.apertus_8B` classmethod that builds an Apertus block, mirroring
`qwen3_8B`. From the tech report §2.1 (8B: 32 layers, d=4096, MLP 21504, 32 Q heads / 8 KV heads):

| Apertus feature | olmo-core status | Action |
|---|---|---|
| GQA (32/8 heads) | supported (`n_kv_heads`) | config only |
| QK-Norm, per-head RMSNorm over `head_dim` | supported (`qk_norm` + `use_head_qk_norm`; Qwen3-style) | config only |
| Pre-norm RMSNorm, no biases | supported | config only |
| RoPE Θ=500k + NTK-aware (LLaMA-3-style) scaling | supported (`StepwiseRoPEScalingConfig` ↔ HF `rope_type="llama3"`) | copy the released model's `rope_scaling` |
| **xIELU** activation (trainable αp, αn per layer) | **not implemented** — `FeedForward` only supports `swiglu` or `torch.nn.functional` activations | **new module** |
| Untied embeddings | supported | config only |

The xIELU feed-forward is the main code task:

- Implement an xIELU `FeedForward` variant in `src/olmo_core/nn/feed_forward.py`
  (`FeedForwardType.xielu`): `αp·x² + 0.5x` for x>0, `αn·(e^x − 1) − αn·x + 0.5x` otherwise, with
  αp/αn as *trainable per-layer scalars* (per the report they are trained; the released checkpoints
  carry them as weights). It must not assume a gated structure (`gate_proj`/`up_proj`): Apertus MLP
  is a plain up/down projection with the activation in between.
- Ensure the Bolmo local encoder/decoder work with it: their block configs are derived from the
  teacher block config via `replace(...)` in `train_stage1.py`/`train_stage2.py`, so xIELU
  propagates automatically once supported — verify this path (local models run their own
  `feed_forward.hidden_size`, e.g. 5504).
- Add a `TransformerConfig.apertus_8B` (and later `apertus_70B`) classmethod.

### Phase 2 — HF ↔ olmo-core conversion mappings

Add `"apertus"` entries to the model-type-specific templates in `src/olmo_core/nn/hf/convert.py`
(see the existing `bolmo` / `bolmo_prenorm` blocks for the pattern):

- Weight-name mapping for the Apertus HF naming (embed tokens, q/k/v/o projections, q_norm/k_norm,
  MLP up/down, **and the xIELU αp/αn scalars**, lm_head — untied, so both embedding and head
  transfer). Verify exact HF parameter names from the `ApertusForCausalLM` implementation; the
  scalar names for αp/αn are the likeliest place to get this wrong.
- Apertus is pre-norm like Llama/Qwen → the `bolmo_prenorm` norm-placement convention applies to
  the byteified export later (`attention_norm` = `input_layernorm`, `feed_forward_norm` =
  `post_attention_layernorm`).
- Validate with the script's built-in logits check (`validate_conversion`).

### Phase 3 — Tokenizer support

- Add a `TokenizerConfig.apertus()` classmethod and register it in `_get_tokenizer_config` of
  `src/examples/huggingface/convert_checkpoint_from_hf.py`: vocab 131,072 (2¹⁷), `eos_token_id = 2`
  (tekken `</s>` = the EoD token Apertus appends to every document), pad token — tekken has no
  dedicated pad, so pick a reserved special token following the llama3 convention
  (`pad_token_id = 128002`-equivalent; verify against `tokenizer_config.json`). ~47 tekken special
  tokens (code/math) ride along via `AutoTokenizer`.
- Byte tokenizer: reuse `ByteTokenizerConfig.blt()` with `original_identifier` pointing at the
  Apertus tokenizer. `from_tokenizer_config` lifts all special/added tokens (chat tokens included)
  into the byte vocab's special-token block — sanity-check that the Apertus chat tokens are captured.
- **`original_model_kind == "apertus"`** (currently hardcoded for `qwen3`/`llama3`/olmo in
  `BolmoTransformer.__init__`, `src/olmo_core/nn/transformer/model.py`):
  - `get_apertus_space_mask()`: tekken is a *tiktoken-style byte-level BPE* — vocab entries are raw
    byte strings, so the `Ġ`/`Ċ`/`ĉ` prefix check used for GPT-2-style tokenizers does **not**
    apply. Check tokens starting with the literal space/newline bytes (and verify how the
    HF-converted tokenizer renders token strings).
  - `eos_token_original`: the token id used as the sequence-start anchor in the original space
    (for Qwen3 it is `Ċ` = 198; for Llama3 `<|begin_of_text|>`). For Apertus the analogue is the
    tekken `<s>`/BoD token (id 1) — confirm against how expanded embeddings/BOS alignment are used,
    and document the choice.

### Phase 4 — Convert the base and instruct checkpoints

Identical config + mapping for both (this is what makes the task vector well-defined):

```bash
for ckpt in Apertus-8B-2509 Apertus-8B-Instruct-2509; do
HF_CHECKPOINT=swiss-ai/$ckpt
OUTPUT_DIR=/data/checkpoints/${ckpt}_olmo_core
python3 src/examples/huggingface/convert_checkpoint_from_hf.py \
    --checkpoint-input-path $HF_CHECKPOINT \
    --output-dir $OUTPUT_DIR \
    --model-arch apertus_8b \
    --tokenizer apertus
done
```

Both released checkpoints are *post-long-context* (65k; RoPE Θ extended per stage up to 12M for
8B, §2.5) — since base and Instruct share this, the delta isolates SFT + QRPO, which is what we
want. The RoPE scaling is config-only, so it does not affect the merge.

### Phase 5 — Byteify the base once (Stage 1 + Stage 2)

Follow `QWEN3_8B_TRAINING.md` steps 4–5 and the Stage 1/Stage 2 commands verbatim, with:

- `OLMO_ARCH=apertus_8B` (picks up the new config + tokenizer wiring; verify the `startswith`
  checks in `build_config` handle `apertus_*` the way `qwen3*` does);
- `data_sources_apertus8.txt` built from `bolmo_mix` tokenized with the tekken tokenizer:

```bash
dolma tokens \
    --documents "/path/to/bolmo_mix/**/*.zst" \
    --tokenizer.name_or_path swiss-ai/Apertus-8B-2509 \
    --destination /path/to/bolmo_mix_apertus \
    --tokenizer.eos_token_id 2 \
    --tokenizer.pad_token_id <reserved-pad-id> \
    --dtype uint32 \
    --processes 32
```

- local model sizes identical to the Qwen3-8B run (`d_model=4096` matches; local FFN hidden 5504);
- expected throughput ~30k TPS/device (Stage 1) and ~25k (Stage 2) on H100, i.e. the same order as
  the Qwen3-8B byteification.
- Sanity gate before proceeding: Stage-1 eval with teacher forcing (cf. `QWEN3_06B_RESULTS.md`) —
  the byteified base should track the subword base on core tasks.

*Variant worth considering:* applying task arithmetic on top of a **Stage-1-only** checkpoint (the
backbone is then bit-identical to the base, so the merge reduces to the pure
`(1−α)·base + α·instruct` combination) vs. the full Stage-2 checkpoint (the released pipeline's
choice; the backbone has drifted via end-to-end training and the delta rides on top). Evaluate both
if budget allows.

### Phase 6 — Instructify (the actual task arithmetic)

```bash
python3 src/examples/bolmo/instructify.py \
    --output=/data/checkpoints/bapertus_8b_instruct \
    --checkpoint-dir=/data/trained/stage2_bapertus_8b/step150000 \
    --base-checkpoint-dir=/data/checkpoints/Apertus-8B-2509_olmo_core \
    --instruct-checkpoint-dir=/data/checkpoints/Apertus-8B-Instruct-2509_olmo_core \
    --alpha=1.0 \
    --instructify-embeddings
```

- **α sweep**: {0.5, 0.75, 1.0, 1.25} evaluated on the multilingual + IFEval + safety slices
  (Phase 8) before picking the release value; the released byteified instruct models used
  α=1.0 by default.
- `--instructify-embeddings` transfers the 131k embedding task vector into the local encoder's
  expanded embeddings (shape-exact). The subword `lm_head` delta is structurally unused.
- Run `count_parameters.py` on the output to confirm the teacher is excluded from counts, and
  inspect the merge log: expected warnings only for shape-mismatched keys (lm_head) and
  Bolmo-only keys (local enc/dec, boundary predictor).
- Apertus-specific ablations (if the SFT-only checkpoint is obtainable):
  1. apply ΔSFT only (byteified SFT model);
  2. apply ΔSFT + α·ΔQRPO (scaled alignment);
  3. compare against the full Δ(SFT+QRPO) at α=1.0.
- Ops note: the merge materializes three 8B models — use `--device cpu` with sufficient host RAM
  (≥ ~60–90 GB depending on checkpoint dtype), or run distributed.

### Phase 7 — Export to HF

Extend the HF Bolmo modeling code (`src/olmo_core/nn/bolmo/hf/modeling_bolmo.py`) with an Apertus
block type: an xIELU MLP variant (the current `BolmoMLP` is SwiGLU gate/up/down) and per-head
QK-norm (already supported via `use_head_qk_norm`), plus the llama3-style RoPE scaling (already
supported). Then:

```bash
python3 src/examples/huggingface/convert_checkpoint_to_hf.py \
    -i /data/checkpoints/bapertus_8b_instruct \
    -o /data/checkpoints/Bapertus-8B-Instruct-hf \
    -s 65536 \
    --dtype bfloat16 \
    --skip-validation   # re-enable validation once the Apertus block type passes it
```

Suggested naming follows the repo convention (Bolmo/Bwen/Blama → **Bapertus-8B(-Instruct)**).

### Phase 8 — Evaluation

- **Loglikelihood evals**: the OLMES fork (bminixhofer/olmes) with the boundary-logit folding
  (README "Reproducing Evaluation"): fold the boundary+byte logits (second vocab half) into the
  byte logits (first half). Define an `apertus8b` task suite analogous to `bolmo1b`/`bolmo7b`.
- **Multilingual (the headline claim)**: reuse Apertus's own suites — `swiss-ai/evals` /
  `swiss-ai/lm-evaluation-harness` (94 languages), Global-MMLU, multilingual HellaSwag/TruthQA,
  FLORES+ translation (§5.3), and the Romansh tasks (Appendix J.1). Compare byteified base vs.
  byteified instruct on the deltas that post-training moved in the subword model.
- **Tokenizer fairness**: recompute fertility / compression ratio / vocabulary utilization / Gini
  (§2.2, Appendix H.3 methodology) — in byte space these become uniform by construction; report the
  quantified equity gain vs. tekken as a headline result.
- **Post-training behavior**: IFEval (`src/examples/bolmo/generate_ifeval.py`); safety suite from
  §5.5 (BBQ, HarmBench, RealToxicityPrompts, ToxiGen, LinguaSafe); constitutional/charter behavior
  on controversial topics (§4.3.2).
- **Memorization**: run the `swiss-ai/apertus-memorization` probes on the byteified models — does
  the Goldfish-trained backbone still suppress verbatim recall when the model can continue at byte
  level? (Byte-level decoding is the natural elicitation surface, so this is both a safety check
  and a paper-able result.)
- **Long context**: the checkpoints carry the 65k config, but Stage 1/2 here ran at 4k tokens
  (24k bytes at expansion 6). Validate the long-context evals (§5.2 long-context table) separately;
  otherwise ship a 4k-token/24k-byte context model and extend later.
- **Success criteria**: the byteified instruct model retains ≈ the multilingual/instruction gains
  that Apertus-8B-Instruct shows over Apertus-8B, with safety behavior intact; reference points are
  the published instruct scores (§5.2, e.g. Knowledge avg 58.8 for Apertus-8B-Instruct) and, if
  budget allows, an upper-bound byteified instruct obtained by byteifying the instruct model
  directly.

---

## Challenges

1. **xIELU is the main engineering gap.** It is absent from olmo-core (only SwiGLU/`F` activations
   in `FeedForward`), and it is not a GLU — plain up/down projection plus two trainable per-layer
   scalars. It must be implemented once, wired into both conversion directions (including the αp/αn
   weight mappings — easy to get the HF names wrong), and verified inside the Bolmo local models
   (which inherit the block config from the teacher config).
2. **Alignment fragility under weight averaging.** Apertus's own report emphasizes that post-hoc
   alignment is fragile (it cites Nasr et al.'s finetuning attacks, Appendix F). Task arithmetic is
   a linear intervention on exactly those weights: the QRPO-derived behavior — especially the
   constitutional/charter handling of controversial topics, trained at very low LR (5e-7) — may
   partially wash out or interact nonlinearly. Mitigation: re-run the full safety suite, sweep α,
   and (if available) apply ΔSFT and ΔQRPO separately to detect asymmetric transfer of helpfulness
   vs. safety.
3. **Optimizer/LR mismatch between the delta and the backbone.** The instruct delta was produced by
   AdEMAMix (β₃=0.99, α=8.0) at SFT LR 5e-6 / QRPO LR 5e-7, while the byteified backbone was
   continued end-to-end with AdamW. How much of the delta survives at α=1.0 depends on the delta's
   magnitude relative to the byteification drift — this is empirical and is the main reason for the
   α sweep. (AdEMAMix support in olmo-core would only matter if we later *train* the merged model;
   task arithmetic itself needs none.)
4. **Patching/boundary calibration shift.** The boundary predictor and local models are trained on
   the base model's byte distribution; the instructified model shifts register (chat turns, longer
   responses, role markers). Generation-time patching uses the boundary predictor, so monitor
   patch-length statistics on chat data and use the available knobs
   (`gen_boundary_bias`, `force_boundary_every`, `max_patch_length_decode`) if patching degrades.
5. **Chat template in byte space.** Apertus's format (§4.2.1) uses dedicated special tokens for
   user/assistant/**developer** roles (developer carries tool definitions — agent-oriented). The
   byte tokenizer lifts special/added tokens via `from_tokenizer_config`, but the 47 tekken special
   tokens + chat tokens must be verified to survive that lift, and the chat template must be
   byte-ified (literal token strings or mapped byte sequences) when formatting prompts at
   evaluation time.
6. **Tokenizer integration details.** tekken is tiktoken-style byte-level BPE: the
   space-mask/eos-anchor helpers assume GPT-2-style `Ġ`/`Ċ` representations and per-family
   hardcoding (`original_model_kind`), so a tekken-specific space mask and eos anchor must be added
   (Phase 3). Also verify `dolma tokens` handles the Apertus tokenizer deterministically.
7. **Long context.** Both released checkpoints are 65k-context; the byteification here runs at 4k
   tokens. Byte-level context of 65k×6 ≈ 393k bytes needs the same context-parallel treatment the
   report used for its long-context phase (§2.5). Treat long-context validation as a separate work
   package; don't block the 4k-context release on it.
8. **Compute.** One-time cost is the two byteification runs (≈ the Qwen3-8B schedule: Stage 1
   75k steps, Stage 2 150k steps on 8×H100). The merge itself is minutes-to-hours. The 70B pair is
   ~9× and undemonstrated in this repo — sequence: 8B first.
9. **Data-prep convention.** Apertus prepends BoD `<s>` / appends EoD `</s>` per document and masks
   the EoD loss (§2.1). Mirror that convention in the byteified data prep and document it, so the
   byteified model's document framing matches what the backbone was pretrained on.

---

## Open questions

1. The SFT-only checkpoint (pre-QRPO) enables the ΔSFT vs. ΔQRPO isolation, which is an interesting Apertus-specific ablation. Could it be used here?
2. For the released byteified instruct models (Olmo/Qwen/Llama): was `--instructify-embeddings`
   used, and was α=1.0 the final choice after a sweep?
3. Was any light post-merge training applied to the released byteified instruct models, or are they
   pure task-arithmetic outputs?
4. Should we target 8B only first, or use smaller madels for verification, or plan the 70B from the start (affects local-model sizing
   and parallelism choices)?
5. Preferred eval target: the Apertus multilingual suite as primary with OLMES secondary, or the
   other way around? Any specific languages to prioritize (Romansh? the 94-language set?).

---

## References

- Apertus 1.0 technical report (arXiv:2509.14233)
- This repo (Post-Training via Task Arithmetic and Byteification recipe)
- ...
