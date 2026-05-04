# gcm_qualitative — interactive interp harness

Loads the model + SAE once, drops you into a REPL, and serves a live
dashboard that updates as you run commands.

## Quick start

```bash
qrsh -l h_rt=4:00:00 -l gpus=1 -l gpu_c=8.9 -l gpu_memory=32G
module load miniconda
conda activate <env>

cd /projectnb/mcnet/jbrin/lang-probing
python experiments/gcm_qualitative/repl.py \
    --config experiments/gcm_qualitative/configs/samples.yaml \
    --port 8765
```

Then in another local terminal (or via VSCode's Ports panel) forward the
port and open  `http://127.0.0.1:8765/dashboard.html`.

In the REPL:

```
(qualitative) > observe                  # all samples
(qualitative) > observe eng_spa_short    # one sample
(qualitative) > attn-modes i,ii,iii      # show all three aggregations
(qualitative) > observe eng_spa_short
(qualitative) > intervene eng_spa_short  # baseline vs each op in YAML
(qualitative) > clear
(qualitative) > reload                   # re-read YAML after editing
(qualitative) > quit
```

## What gets shown

**observe** — for each sample, one panel per requested SAE feature and
per requested attention head:

- Per-token activation strip (color-coded). For heads, three available
  aggregation modes:
  - **(i)** mean attention paid by this head, averaged over query positions
  - **(ii)** attention to last_src_token from each query
  - **(iii)** attention from last_src_token as query (DEFAULT — matches
    where the GCM run patches)
- Logit lens at configured anchor positions (last_src_token / first_tgt_token):
  top-K vocab from the head's residual contribution, shown both with and
  without the final layer norm.

**intervene** — for each sample × each op in `intervene.ops`:

- Baseline generation
- Generation with the op applied (head zero'd, SAE feature zero'd, or SAE
  feature scaled / activated)
- Δ log p(gold target | prompt) under the intervention

Op kinds:
- `ablate_head` (layer, head, positions)
- `ablate_feature` (feature_idx, positions)
- `steer_feature` (feature_idx, scale, positions)

`positions: all` applies during every step of decoding; `positions:
last_src_only` only modifies the last source token.

## Files

- `repl.py` — REPL + arg parsing (named `repl` to avoid shadowing stdlib `inspect`)
- `core.py` — observe / intervene model functions
- `render.py` — manifest.json + HTML fragment writers
- `server.py` — threaded HTTP server
- `templates/dashboard.html` — auto-polling dashboard (stepper + grid views)
- `configs/samples.yaml` — sample set + components + intervene ops
- `out/` — written by the REPL; served by the HTTP server (gitignored)
