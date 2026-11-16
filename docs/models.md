# Models

Which checkpoints the server can run, what differs between them, and what each one needs. The server itself, its environment variables, and the memory and cache behaviour that apply to every model, are in [server.md](server.md).

One server process runs one model, chosen at startup with `MFLUXIBLE_MODEL`:

| `MFLUXIBLE_MODEL` | Weights | RAM q8 / q4 | Default steps | Guidance | Negative prompt | Fractional start |
|---|---|---|---|---|---|---|
| `z-image-turbo` (default) | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | 11.0 / 5.9 GB | 9 | — | — | yes |
| `z-image` | [Tongyi-MAI/Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | 11.0 / 5.9 GB | 50 | default 4.0 | yes | — |
| `flux-schnell` | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) † | 18.0 / 9.6 GB | 4 | — | — | yes |
| `flux-dev` | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) † | 18.0 / 9.6 GB | 25 | default 3.5 | — | yes |
| `krea-dev` | [black-forest-labs/FLUX.1-Krea-dev](https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev) † | 18.0 / 9.6 GB | 25 | default 3.5 | — | yes |
| `qwen-image` | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | 30.8 / 16.4 GB | 20 | default 3.5 | yes | yes |
| `qwen-image-2.1` | [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) | not measured | 40 | default 1.0 | above guidance 1.0 | yes |
| `qwen-image-2.1-turbo` | [Qwen/Qwen-Image-2.1-Turbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo) | not measured | 8 | — | — | — |
| `krea-2` | [krea/Krea-2-Turbo](https://huggingface.co/krea/Krea-2-Turbo) † | 18.8 / 10.2 GB | 8 | default 1.0 | above guidance 1.0 | — |
| `krea-2-raw` | [krea/Krea-2-Raw](https://huggingface.co/krea/Krea-2-Raw) † | 18.8 / 10.2 GB | 25 | default 1.0 | above guidance 1.0 | — |
| `ernie-image-turbo` | [baidu/ERNIE-Image-Turbo](https://huggingface.co/baidu/ERNIE-Image-Turbo) | 12.8 / 6.9 GB | 8 | — | — | yes |
| `ernie-image` | [baidu/ERNIE-Image](https://huggingface.co/baidu/ERNIE-Image) | 12.8 / 6.9 GB | 50 | default 4.0 | yes | yes |
| `anima-turbo` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | not measured | 10 | default 1.0 | above guidance 1.0 | — |
| `anima-aesthetic` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | not measured | 30 | default 4.0 | yes | — |
| `anima-base` | [circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima) | not measured | 30 | default 4.0 | yes | — |
| `fibo2` | [briaai/Fibo-2.0-dev](https://huggingface.co/briaai/Fibo-2.0-dev) † | not measured | 50 | default 5.0 | yes | — |
| `flux3-image-klein` | [black-forest-labs/flux-3-image-klein](https://huggingface.co/black-forest-labs/flux-3-image-klein) † | not measured | 16 | — | — | — |
| `flux3-image-klein-base` | [black-forest-labs/flux-3-image-klein-base](https://huggingface.co/black-forest-labs/flux-3-image-klein-base) † | not measured | 50 | default 3.5 | — | — |
| `fibo` | [briaai/FIBO](https://huggingface.co/briaai/FIBO) † | not measured | 50 | default 4.0 | — | — |
| `iris-3b` | [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) | not measured | 4 | — | — | — |
| `iris-3b-base` | [speridlabs/iris-3b](https://huggingface.co/speridlabs/iris-3b) | not measured | 100 | default 3.0 | yes | — |
| `boogu-image-turbo` | [Boogu/Boogu-Image-0.1-Turbo](https://huggingface.co/Boogu/Boogu-Image-0.1-Turbo) | not measured | 4 | — | — | — |
| `ming-image-design` | [inclusionAI/Ming-Image-0.1-Design](https://huggingface.co/inclusionAI/Ming-Image-0.1-Design) | not measured | 12 | default 1.0 | — | — |
| `flux2-klein-4b` | [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | 8.6 / 4.6 GB | 4 | — | — | — |
| `flux2-klein-9b` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) † | 18.5 / 9.9 GB | 4 | — | — | — |
| `flux2-klein-9b-kv` | [black-forest-labs/FLUX.2-klein-9b-kv](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv) † | 18.5 / 9.9 GB | 4 | — | — | — |
| `flux2-klein-base-4b` | [black-forest-labs/FLUX.2-klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | 8.6 / 4.6 GB | 50 | default 1.5 | — | — |
| `flux2-klein-base-9b` | [black-forest-labs/FLUX.2-klein-base-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) † | 18.5 / 9.9 GB | 50 | default 1.5 | — | — |

† Gated on Hugging Face — needs an account, an accepted licence and a token before the weights will download. See [Gated weights](#gated-weights-and-hf_token).

**RAM q8 / q4** is the quantized weights at `MFLUXIBLE_QUANTIZE=8` (the default) and at `4`, summed across the directories mflux downloads (`transformer/`, `text_encoder*/`, `vae/`). MLX quantizes in groups of 64 weights and stores an fp16 scale and bias per group, so a weight costs `bits/8 + 0.0625` bytes, and the VAE is not quantized at all — which is why **q4 lands at ~53% of q8 rather than half**: that 0.0625 is fixed, so it is a larger share of a 4-bit weight than an 8-bit one. `none` gives full precision, roughly twice the q8 figure.

Expect the process to sit roughly 2–3 GB above the figure while generating: MLX's reusable buffer cache (capped at 1 GB by default, see [Memory](server.md#memory)) plus activations, which scale with resolution rather than step count.

These are derived from each checkpoint's published parameter count rather than measured one by one, but the formula agrees with all three quantized caches measured locally to within 2% — `flux2-klein-4b` caches at 8.59 GB at q8 and 4.63 GB at q4, `z-image-turbo` at 10.74 GB, which runs at 12–13 GB resident.

Base and turbo variants of the same family share an architecture, so they share a footprint: `z-image` and `z-image-turbo` are both 6.2B parameters, as are `ernie-image` and its turbo, and the three 9B Klein checkpoints. What separates them is step count and guidance behaviour, not size.

Disk costs more than RAM on the first run, and it's the download that dominates: every checkpoint here publishes bf16 weights, about 1.9× the q8 figure, *except* `z-image-turbo`, whose transformer ships fp32 — 33 GB downloaded to cache 10 GB. The quantized copy is then written separately, so peak disk is roughly 3× the q8 figure (4× for `z-image-turbo`). Once that quantized cache exists the Hugging Face download under `~/.cache/huggingface/hub/` can be deleted — it's only needed again if you change `MFLUXIBLE_QUANTIZE` or the LoRA configuration, both of which quantize afresh into their own cache directory.

mflux's own aliases work too (`schnell`, `dev`, `qwen`, `zimage`, `klein-4b`, `krea2`, …), and an unrecognised name fails at startup with the list of valid ones — before anything is downloaded.

**Only the model you select is ever fetched.** All twenty-seven are named in `mfluxible/models.py`, but an entry there is inert data: its mflux imports are deferred into a loader function that runs at load time, and mflux downloads weights inside the model's constructor, not at import. The twenty-six you aren't running cost nothing beyond their row in that table.

Switching models means restarting the server, the same way `MFLUXIBLE_QUANTIZE` and LoRAs do. Each model + quantization + LoRA combination keeps its own quantized cache directory, so switching back doesn't re-quantize.

What differs per model, beyond the step count:

- **Guidance.** The distilled models — Z-Image-Turbo, FLUX.1-schnell, ERNIE-Image-Turbo, Qwen-Image-2.1-Turbo, FLUX.3-Klein and the three non-`base` FLUX.2 Klein checkpoints — have nowhere for a guidance value to go: mflux forces it to 0, builds no guidance embedder, or hard-errors on any value but 1.0. Sending `guidance` to those is a 400 rather than a field quietly dropped. FLUX.1-dev and FLUX.1-Krea-dev take distilled guidance; Z-Image, Qwen-Image, Qwen-Image-2.1 (base), ERNIE-Image, Krea-2, Anima (Aesthetic and Base, and Turbo above 1.0), FIBO-2 and the FLUX.2 `base` checkpoints run classifier-free guidance; FLUX.3-Klein-Base takes a guidance value with no negative branch. Qwen-Image-2.1 base defaults to guidance 1.0 and its own recommended sampling is 40 steps without CFG.
- **Negative prompts.** A negative prompt needs CFG to have any effect, and every model here builds its unconditional branch only *above* guidance 1.0. So Z-Image, Qwen-Image, Qwen-Image-2.1 (base), ERNIE-Image, Krea-2, Anima and FIBO-2 accept one and the rest are a 400 — and on Krea-2, Qwen-Image-2.1 base and Anima-Turbo, whose default guidance is 1.0, a negative prompt is also a 400 unless you raise `guidance` alongside it. FLUX.1 and FLUX.2 have no negative branch at all. That branch is also why CFG steps are expensive: the transformer runs twice per step, conditional and unconditional, whether or not you send a `negative_prompt`.
- **Fractional start.** [`fractional_start`](api.md#fractional-start) works by extending mflux's *linear* schedule, so it's only offered on models that would have run one. Z-Image (base), Krea-2, Qwen-Image-2.1-Turbo (its own fixed 8-node schedule) and every FLUX.2 Klein pick a different sampler for themselves — flow-match, Krea-2's `er_sde`, or Turbo's schedule — and asking for a fractional start there is a 400. Plain `image_strength` still works on all of them, quantized to `1/steps` as usual.
- **Step previews.** Iris and Boogu cannot decode a step preview: Iris is pixel-space with no VAE, and Boogu has no latent creator in mflux. `preview_every` (and OpenAI's `partial_images`) is a 400 on them, and the harness hides the control.
- **Input images.** Anima, FIBO-2, FLUX.3 Klein, Ming-Image-Design, Iris and Boogu are text-to-image only here: their mflux `generate_image()` takes no `image_path`, so an `image` (and any `mask`) is a 400 on them. The other models take one as described under [Image-to-image](api.md#image-to-image).
- **Size.** Z-Image-Turbo is among the smallest here and Qwen-Image much the largest (a ~20B transformer alongside a multimodal text encoder), with Krea-2 (12B) and the 9B FLUX.2/ERNIE checkpoints in between. For scale, Z-Image-Turbo alone occupies 10GB of quantized weights at `MFLUXIBLE_QUANTIZE=8` and 5.5GB at `4`. On a 32GB machine, expect to want `4` or lower for anything above ~9B, and read [Memory](server.md#memory) first — running out of headroom doesn't fail loudly, it just makes every generation slow.

All the bundled clients leave `steps` to the server unless you set it, so they follow whichever model is loaded without reconfiguration. The MCP tool and the browser harness go further and read [`/health`](api.md#get-health): the harness only shows Guidance, Negative prompt and Fractional start when the model accepts them, and the MCP tool refuses those arguments up front rather than spending a round trip to be told no.

### Gated weights and `HF_TOKEN`

Twelve of the twenty-seven repos above are gated († in the table). Getting at those takes an account, an accepted licence *and* a token — the token on its own is not enough:

1. Open the model page signed in and click **Agree and access repository**. All eight are `gated: auto`, so access lands the moment you accept — there's no queue waiting on a human.
2. Create a **Read** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A fine-grained token works too, but it needs *"Read access to contents of all public gated repos you can access"* ticked, or it will 401 on exactly these models.
3. Store it once with `hf auth login` (which writes `~/.cache/huggingface/token`), or export `HF_TOKEN` in the shell that starts the server. Either way it's `huggingface_hub` that reads it, not mfluxible — which is why there's no `MFLUXIBLE_*` variable for it, and why a dedicated machine needs its own copy (see [Running on a dedicated machine](server.md#running-on-a-dedicated-machine)).

Gating is a per-repo switch with no particular relationship to the licence — FLUX.1-schnell is Apache 2.0 and still answers an anonymous request with a 401 — and it's per-checkpoint rather than per-family: `flux2-klein-4b` and `flux2-klein-base-4b` are open while all three 9B Kleins are gated. Check the row rather than generalising from the sibling you ran last.

Missing access fails at **startup**, not at request time. `engine.load()` runs in the FastAPI lifespan and mflux downloads weights inside the model's constructor, so a 401 surfaces while the server is still booting and nothing is listening on the port yet. The default `z-image-turbo` is ungated, so the [quickstart](../README.md) needs no token at all.

One thing to know before killing a download to add a token: **it will not resume.** `huggingface_hub` 1.x writes each file to a process-unique `<etag>.<uuid>.incomplete` and only renames it into place once complete, so the restart starts that file from zero and the dead run's partials are orphaned. That's deliberate — a name unique per process means a filesystem whose `flock` silently lets two writers through costs duplicated bandwidth instead of a corrupted blob. `hf cache prune` reclaims the orphans, but run it only when nothing is downloading: it globs every `*.incomplete` in the cache with no check for whether something is still writing to one.

Adding a model mfluxible doesn't already run is a [CONTRIBUTING](../CONTRIBUTING.md#adding-a-model) topic.
