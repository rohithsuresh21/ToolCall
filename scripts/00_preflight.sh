#!/usr/bin/env bash
# Quick readiness gate for the tomorrow run. Run ON THE NODE before starting
# the full pipeline to confirm the model + CUDA + vllm are ready, so we never
# burn GPU reservation time discovering a missing dependency.
#
# Usage (on the node, from repo root):  bash scripts/00_preflight.sh [MODEL]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-${MODEL:-Qwen/Qwen3-1.7B}}"

echo "======================================================================"
echo "  PREFLIGHT  model=$MODEL"
echo "======================================================================"

python3 - "$MODEL" <<'PY'
import importlib, os, sys, time
model = sys.argv[1]
mods = ["torch", "transformers", "peft", "vllm", "datasets"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK  {m:<12} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  FAIL {m:<12} {type(e).__name__}: {e}")
        sys.exit(1)
import torch
print(f"  CUDA available: {torch.cuda.is_available()}  devices: {torch.cuda.device_count()}")
if not torch.cuda.is_available():
    sys.exit("FATAL: no CUDA on node - nothing to train on")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
try:
    t0 = time.time()
    from huggingface_hub import scan_cache_dir
    hits = [r for r in scan_cache_dir().repos if r.repo_id == model and r.repo_type == "model"]
    if not hits or not hits[0].refs:
        raise SystemExit(f"{model} not found in HF cache")
    rev = next(iter(hits[0].revisions))
    print(f"  snapshot OK: {rev.snapshot_path} ({time.time()-t0:.0f}s)")
    import transformers
    transformers.AutoTokenizer.from_pretrained(model)
    print(f"  tokenizer OK for {model}")
except SystemExit:
    raise
except Exception as e:
    print(f"  FAIL model not cached offline: {type(e).__name__}: {e}")
    sys.exit(1)
PY

echo ""
echo "  PREFLIGHT PASSED - model cached, CUDA ready, vllm importable."
echo "  If any FAIL above: pull the model with network ON, e.g."
echo "    HF_HUB_OFFLINE=0 python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('$MODEL')\""
echo "  then re-run this gate."
echo "======================================================================"
