"""Run in its own process (vLLM holds the GPU for its lifetime): greedy ids and top logprobs on a few prompts,
and the engine's default sampling parameters. Writes JSON to argv[1]."""

import json
import sys

from intact import config, io, weights

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
out["default_sampling_params"] = repr(llm.get_default_sampling_params())
json.dump(out, open(sys.argv[1], "w"))
