import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tiktoken
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, mask=None, cache=None):
        b, t, c = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        q = q.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, c // self.n_head).transpose(1, 2)

        if cache is not None:
            key_cache, value_cache = cache
            k = torch.cat([key_cache, k], dim=2)
            v = torch.cat([value_cache, v], dim=2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if mask is not None:
            mask = mask.bool()
            att = att.masked_fill(~mask, float("-inf"))

        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(b, t, c)

        out = self.resid_dropout(self.c_proj(out))
        return out, (k, v)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            config.n_embd, eps=1e-5, elementwise_affine=config.bias
        )
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(
            config.n_embd, eps=1e-5, elementwise_affine=config.bias
        )
        self.mlp = MLP(config)

    def forward(self, x, mask=None, cache=None):
        norm = self.ln_1(x)
        att, cache = self.attn(norm, mask=mask, cache=cache)
        x = x + att
        norm = self.ln_2(x)
        mlp = self.mlp(norm)
        x = x + mlp
        return x, cache


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(
            config.n_embd, eps=1e-5, elementwise_affine=config.bias
        )

    def _forward_transformer_blocks(self, x, mask=None, cache=None, build_cache=False):
        x = self.drop(x)
        kv_cache = []

        if cache is not None:
            for i in range(len(cache)):
                x, cache[i] = self.h[i](x, mask=None, cache=cache[i])
        else:
            for block in self.h:
                x, curr_cache = block(x, mask=mask)
                if build_cache:
                    kv_cache.append(curr_cache)

        x = self.ln_f(x)
        return x, kv_cache if build_cache else cache

    def _create_causal_mask(self, length: int):
        mask = torch.tril(torch.ones((length, length), dtype=torch.float32))
        mask = mask.bool()
        return mask.view(1, 1, length, length).to(self.wte.weight.device)

    def _sample_next_token(self, x, temperature):
        logits = x[:, -1:] @ self.wte.weight.T
        y = logits[:, -1, :]
        y = torch.multinomial(
            torch.softmax(y * (1 / temperature), dim=-1), num_samples=1
        )
        return y

    def generate(self, x, max_new_tokens=256, temperature=0.8):
        text_embeds = self.wte(x)
        batch_size, text_len, _ = text_embeds.size()
        pos_ids = torch.arange(0, text_len, dtype=torch.long, device=text_embeds.device)
        pos_emb = self.wpe(pos_ids).unsqueeze(0).expand(batch_size, text_len, -1)
        text_embeds = text_embeds + pos_emb

        mask = self._create_causal_mask(text_len)

        text_embeds, cache = self._forward_transformer_blocks(
            text_embeds, mask=mask, build_cache=True
        )
        text_embeds = text_embeds.detach()

        tokens = []
        for _ in range(max_new_tokens):
            y = self._sample_next_token(text_embeds[:, -1:, :], temperature)

            yield y

            tokens.append(y)
            new_embed = self.wte(y)

            pos_emb = self.wpe(
                torch.tensor([text_len], dtype=torch.long, device=y.device)
            ).unsqueeze(0)
            new_embed = new_embed + pos_emb

            text_embeds = torch.cat([text_embeds, new_embed], dim=1)

            text_embeds[:, -1:, :], cache = self._forward_transformer_blocks(
                text_embeds[:, -1:, :], cache=cache
            )
            text_embeds = text_embeds.detach()
            text_len += 1

        del cache
        torch.cuda.empty_cache()

        return tokens

    def forward(
        self,
        x,
        targets=None,
        attention_mask=None,
    ):
        text_embeds = self.wte(x)

        batch_size, text_len, _ = text_embeds.size()
        pos_ids = torch.arange(0, text_len, dtype=torch.long, device=text_embeds.device)
        pos_emb = self.wpe(pos_ids).unsqueeze(0).expand(batch_size, text_len, -1)
        text_embeds = text_embeds + pos_emb

        batch_size, seq_length, _ = text_embeds.shape
        causal_mask = self._create_causal_mask(seq_length)

        assert (
            seq_length <= self.config.block_size
        ), f"Cannot forward sequence of length {seq_length}, block size is only {self.config.block_size}"

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        if attention_mask is not None:
            combined_mask = causal_mask & attention_mask
        else:
            combined_mask = causal_mask

        x, _ = self._forward_transformer_blocks(text_embeds, mask=combined_mask)
        logits = x @ self.wte.weight.T

        if targets is not None:
            vocab_size = logits.size(-1)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets, ignore_index=-100)
            return logits, loss

        return logits, None


def transpose_specific_layers(state_dict):
    layers_to_transpose = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for key in state_dict.keys():
        if any(key.endswith(suffix) for suffix in layers_to_transpose):
            state_dict[key] = state_dict[key].T
    return state_dict


def generate_text(
    model: GPT, tokenizer, initial_text="", temperature=0.8, max_new_tokens=256
):
    if initial_text:
        x = torch.tensor(
            [tokenizer.encode(initial_text)], device=model.wte.weight.device
        )
    else:
        x = torch.tensor([[tokenizer.bos_token_id]], device=model.wte.weight.device)

    tokens = []
    start = time.time()
    for token in model.generate(
        x, max_new_tokens=max_new_tokens, temperature=temperature
    ):
        tok = token.item()
        tokens.append(tok)
        print(tokenizer.decode([tok]), end="", flush=True)
    end = time.time()
    print("---------------")
    print(
        f"time: {end - start:.3f} s, tokens per second: {len(tokens) / (end - start)}"
    )
    print("---------------")
    returned_text = []
    for token in tokens:
        decoded = tokenizer.decode([token])
        if token == tokenizer.eot_token:
            break
        if decoded == "\n":
            returned_text.append(" ")
        elif decoded != "":
            returned_text.append(decoded)

    return "".join(returned_text)


if __name__ == "__main__":
    config = GPTConfig()
    model = GPT(config)

    state_dict = torch.load("gpt2_sft_3_15000.pt", map_location="cpu")
    # state_dict = torch.load("gpt2.pt", map_location="cpu")
    # state_dict = transpose_specific_layers(state_dict)

    model.load_state_dict(state_dict, strict=False)

    tokenizer = tiktoken.get_encoding("gpt2")

    # prompt = "Tell me your favorite fact about the Milky Way galaxy"
    prompt = """<|im_start|>user
Tell me your favorite fact about the Milky Way galaxy
<|im_end|>"""
    print(prompt, end="")
    generate_text(model, tokenizer, initial_text=prompt, max_new_tokens=512)
