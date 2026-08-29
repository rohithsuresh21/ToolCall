"""
Modal deployment: SFT, GRPO, batch eval, and an inference endpoint.

    modal run scripts/modal_app.py::sft
    modal run scripts/modal_app.py::grpo
    modal run scripts/modal_app.py::eval_dev --adapter /vol/sft-1p7b
    modal serve scripts/modal_app.py            # OpenAI-compatible vLLM endpoint

The final submission model is Qwen3-4B, but every experiment should be run on
1.7B first: same code path, roughly a third of the cost, and the ordering of
data-recipe decisions transfers almost perfectly. Only the last two or three
runs need to be 4B.
"""
from __future__ import annotations

import modal

APP = "atr-tool-reasoning"
VOL = modal.Volume.from_name("atr-artifacts", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("atr-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4", "transformers>=4.51", "accelerate>=0.34", "peft>=0.13",
        "datasets>=2.20", "vllm>=0.6.3", "openai>=1.40", "bitsandbytes>=0.43",
    )
    .add_local_dir("atr", remote_path="/root/atr")
)

app = modal.App(APP, image=image)
GPU_DEV = "A100-40GB"     # plenty for 1.7B LoRA
GPU_FINAL = "H100"        # 4B full fine-tune / 4B GRPO rollouts
VOLUMES = {"/vol": VOL, "/root/.cache/huggingface": HF_CACHE}


@app.function(gpu=GPU_DEV, volumes=VOLUMES, timeout=60 * 60 * 6)
def sft(model_id: str = "Qwen/Qwen3-1.7B", data: str = "/vol/sft.jsonl",
        out_dir: str = "/vol/sft-1p7b", epochs: float = 2.0, lora: bool = True):
    import sys
    sys.path.insert(0, "/root")
    from atr.train.sft import SFTConfig, main
    main(SFTConfig(model_id=model_id, data=data, out_dir=out_dir, epochs=epochs, lora=lora))
    VOL.commit()


@app.function(gpu=GPU_FINAL, volumes=VOLUMES, timeout=60 * 60 * 12)
def grpo(model_id: str = "Qwen/Qwen3-1.7B", adapter: str = "/vol/sft-1p7b",
         out_dir: str = "/vol/grpo-1p7b", steps: int = 300):
    import sys
    sys.path.insert(0, "/root")
    from atr.train.grpo import GRPOConfig, GRPOTrainer
    GRPOTrainer(GRPOConfig(model_id=model_id, adapter=adapter, out_dir=out_dir,
                           steps=steps, lr=2e-5)).train()
    VOL.commit()


@app.function(gpu=GPU_DEV, volumes=VOLUMES, timeout=60 * 60 * 3)
def eval_dev(model_id: str = "Qwen/Qwen3-1.7B", adapter: str | None = None,
             n_per_type: int | None = None, out: str = "/vol/eval"):
    import sys
    sys.path.insert(0, "/root")
    from atr.agent.backends import SamplingParams, VLLMBackend
    from atr.agent.loop import LoopConfig
    from atr.eval.harness import aggregate, evaluate, format_report, load_eval_config, save_run
    from atr.tasks.generator import dev_set

    ecfg = load_eval_config()   # F10: frozen conditions, same as the GRPO canary
    n_per_type = n_per_type or ecfg["n_per_type"]
    tasks = dev_set(n_per_type)
    be = VLLMBackend(model_id, adapter=adapter)
    cards, trajs = evaluate(tasks, be,
                            cfg=LoopConfig(max_steps=ecfg["max_steps"],
                                           repeat_guard=ecfg["repeat_guard"]),
                            sp=SamplingParams(temperature=ecfg["temperature"],
                                              max_tokens=ecfg["max_new_tokens"]))
    rep = aggregate(cards)
    print(format_report(rep, f"{model_id} + {adapter or 'base'}"))
    save_run(out, cards, trajs, rep, {"title": model_id, "adapter": adapter})
    VOL.commit()
    return rep["overall"]


@app.function(gpu=GPU_FINAL, volumes=VOLUMES, timeout=60 * 60, scaledown_window=300)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    """OpenAI-compatible endpoint for the submitted model."""
    import subprocess
    model = "/vol/final-4b"   # merged checkpoint
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--served-model-name", "atr-qwen3-4b",
        "--host", "0.0.0.0", "--port", "8000",
        "--max-model-len", "8192", "--gpu-memory-utilization", "0.90",
    ])


@app.local_entrypoint()
def main():
    print("targets: sft | grpo | eval_dev | serve  (see module docstring)")
