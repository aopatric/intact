"""Run in its own process (vLLM holds the GPU for its lifetime): greedy ids and top logprobs on a few prompts,
and the top_k −1 vs 0 equivalence. Writes JSON to argv[1]."""

import json
import sys

from testbed import config, io, sample, weights

cfg = config.load()
prompts = io.read_parquet(cfg.artifacts_root / "prompts" / "hint.parquet")
prompts = prompts[prompts.split == "test"].head(5)
llm, _ = weights.make_llm(cfg, "rh-s1")
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

reqs = [TokensPrompt(prompt_token_ids=list(map(int, ids))) for ids in prompts.prompt_token_ids]
greedy = llm.generate(reqs, SamplingParams(temperature=0.0, max_tokens=16, logprobs=2), use_tqdm=False)
out = {
    "problem_ids": prompts.problem_id.tolist(),
    "greedy_ids": [list(o.outputs[0].token_ids) for o in greedy],
    "greedy_logprobs": [
        [{str(t): lp.logprob for t, lp in step.items()} for step in o.outputs[0].logprobs] for o in greedy
    ],
}
# top_k -1 and 0 both map to "no top-k" in the engine (v1/worker/gpu_input_batch.py); here we check whether a
# repeated seeded request reproduces (the second call hits the prefix cache).
runs = []
for top_k in (-1, -1, 0):
    p = sample.sampling_params(cfg, "train-cfg", seed=1234, n=4)
    p.top_k, p.max_tokens = top_k, 64
    runs.append([[list(c.token_ids) for c in o.outputs] for o in llm.generate(reqs[:2], p, use_tqdm=False)])
out["repeat_identical"] = runs[0] == runs[1]
out["topk_minus1_equals_0"] = runs[1] == runs[2]
out["default_sampling_params"] = repr(llm.get_default_sampling_params())
json.dump(out, open(sys.argv[1], "w"))
