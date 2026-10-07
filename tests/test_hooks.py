"""Hook capture on a tiny random Qwen3 (CPU): keys, sites, lifecycle (ported from `claude-docs/reference/test_model.py`)."""

import pytest
import torch

from intact.hooks import capture_activations, find_decoder, parse_layers


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
    )
    return Qwen3ForCausalLM(cfg).eval()


IDS = torch.tensor([[1, 5, 9, 2, 7, 3, 11, 4]])


@pytest.mark.parametrize(
    "spec, keys",
    [
        ("34", ("34",)),
        ("36pre,34,34", ("34", "36pre")),
        ([36, "36pre", 0], ("0", "36pre", "36")),
        ("all", tuple(str(i) for i in range(37))),
        ("all,36pre", tuple(str(i) for i in range(36)) + ("36pre", "36")),
        (" 07 ", ("7",)),
    ],
)
def test_parse_layers(spec, keys):
    assert parse_layers(spec) == keys


@pytest.mark.parametrize("bad", ["37", "-1", "35pre", "pre", "x", "", []])
def test_parse_layers_rejects(bad):
    with pytest.raises(ValueError):
        parse_layers(bad)


def test_capture_equals_hidden_states_exactly(tiny):
    """Every index 0..n is the very tensor HF puts in `hidden_states`; `<n>pre` is the last block's output, and the
    final norm of it is `hidden_states[n]`."""
    with torch.no_grad(), capture_activations(tiny, "all,4pre") as store:
        hs = tiny(IDS, output_hidden_states=True).hidden_states
    assert len(hs) == 5
    for i in range(5):
        assert len(store[str(i)]) == 1
        assert torch.equal(store[str(i)][0], hs[i]), i
    pre = store["4pre"][0]
    assert not torch.equal(pre, hs[4])
    assert torch.equal(find_decoder(tiny).norm(pre), hs[4])


def test_find_decoder_through_peft(tiny):
    from peft import LoraConfig, get_peft_model

    wrapped = get_peft_model(tiny, LoraConfig(r=2, target_modules=["q_proj", "v_proj"]), adapter_name="t")
    assert find_decoder(wrapped) is find_decoder(tiny)
    wrapped.unload()  # restore the shared fixture's modules


def test_hooks_removed_even_on_exception(tiny):
    decoder = find_decoder(tiny)
    sites = [decoder.embed_tokens, decoder.layers[1], decoder.layers[3], decoder.norm]
    with pytest.raises(RuntimeError):
        with capture_activations(tiny, "0,2,4pre,4"):
            assert all(len(m._forward_hooks) == 1 for m in sites)
            raise RuntimeError("boom")
    assert all(len(m._forward_hooks) == 0 for m in sites)


def test_one_entry_per_forward_call(tiny):
    """Under `generate` with a KV cache the hook fires per step: the prompt first, then one token each."""
    with torch.no_grad(), capture_activations(tiny, [2]) as store:
        tiny.generate(IDS, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
    assert [t.shape[1] for t in store["2"]] == [IDS.shape[1], 1, 1]


def test_readonly_hooks_leave_outputs_unchanged(tiny):
    with torch.no_grad():
        plain = tiny(IDS).logits
        with capture_activations(tiny, "all,4pre"):
            hooked = tiny(IDS).logits
    assert torch.equal(plain, hooked)
