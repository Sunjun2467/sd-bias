"""노트북과 스크립트가 함께 쓰는 함수."""
from pathlib import Path

import numpy as np
import torch

from src.prompts import (fill, DIRECTION_PAIRS, TEMPLATES, GEN_TEMPLATE,
                         RANDOM_DIR_SEED, NUM_INFERENCE_STEPS)

ROOT = Path(__file__).resolve().parent.parent      # C:\sd-bias
SD_PATH = ROOT / "models" / "sd15"
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
DEVICE = "cuda"


# ---------- 모델 로드 ----------
def load_pipeline():
    """이미지 생성용. SD 1.5 전체를 fp16으로 GPU에 올린다."""
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_PATH, torch_dtype=torch.float16, variant="fp16").to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_text_encoder():
    """분석용. 텍스트 인코더와 토크나이저만 올린다 (빠르고 VRAM 적게 씀).
    분석은 정밀도를 위해 fp32로 계산한다."""
    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(SD_PATH, subfolder="tokenizer")
    enc = CLIPTextModel.from_pretrained(
        SD_PATH, subfolder="text_encoder", variant="fp16",
        torch_dtype=torch.float16).float().to(DEVICE).eval()
    return tok, enc


# ---------- 인코딩 ----------
@torch.no_grad()
def encode(prompts, tok, enc, batch_size=64):
    """프롬프트들의 EOS 위치 벡터를 돌려준다.

    반환 dict:
      layers : (n, 13, 768)  임베딩층 + 12개 층의 hidden state (분석용)
      final  : (n, 768)      최종 출력(last_hidden_state, final layer norm 이후).
                             SD가 실제로 받는 공간이며 개입용 방향은 여기서 구한다.
    참고: hidden_states[-1]은 final layer norm 이전이라 final과 다르다.
    """
    layers, final = [], []
    eos_id = tok.eos_token_id
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        ids = tok(batch, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(DEVICE)
        out = enc(ids, output_hidden_states=True)
        # SD 1.5 토크나이저는 패딩도 EOS 토큰이라, 첫 번째 EOS 위치를 쓴다
        eos_pos = (ids == eos_id).int().argmax(dim=1)
        rows = torch.arange(len(batch), device=DEVICE)
        hs = torch.stack(out.hidden_states, dim=1)          # (b, 13, 77, 768)
        layers.append(hs[rows, :, eos_pos].float().cpu())   # (b, 13, 768)
        final.append(out.last_hidden_state[rows, eos_pos].float().cpu())
    return {"layers": torch.cat(layers).numpy(),
            "final": torch.cat(final).numpy()}


def encode_words(words, templates, tok, enc):
    """단어마다 템플릿 평균 벡터. 반환은 encode와 같은 dict이고 n = 단어 수."""
    prompts = [fill(t, w) for w in words for t in templates]
    out = encode(prompts, tok, enc)
    k = len(templates)
    return {key: v.reshape(len(words), k, *v.shape[1:]).mean(axis=1)
            for key, v in out.items()}


# ---------- 성별 방향 ----------
def gender_direction(pairs, templates, tok, enc, space="layers"):
    """각 쌍의 (남성 − 여성) 차이를 템플릿·쌍에 걸쳐 평균 내고 정규화한다.

    space="layers": (13, 768), 층마다 따로 구한 방향 (분석·시각화용)
    space="final" : (768,), 최종 출력 공간의 방향 (개입용)
    차이 벡터를 쓰므로 평균 중심화를 해도 결과는 같다.

    반환: (d, diffs)  diffs는 쌍별 차이 벡터로 scree plot(검증 2)에 쓴다.
    """
    male = encode_words([m for m, _ in pairs], templates, tok, enc)[space]
    female = encode_words([f for _, f in pairs], templates, tok, enc)[space]
    diffs = male - female                       # (쌍 수, [13,] 768)
    d = diffs.mean(axis=0)
    norm = np.linalg.norm(d, axis=-1, keepdims=True)
    # 0번 층(임베딩층)의 EOS 벡터에는 앞 단어 정보가 없어 차이가 0일 수 있다.
    # 그런 층은 NaN 대신 0 벡터를 돌려준다 (투영값이 전부 0 = "성별 정보 없음").
    d = np.where(norm > 1e-6, d / np.maximum(norm, 1e-12), 0.0)
    return d, diffs


def gender_anchors(pairs, templates, tok, enc, d):
    """최종 출력 공간에서 성별 축의 기준점.
    c: 남성 단어 평균 투영값과 여성 단어 평균 투영값의 가운데 (제거 목표)
    s: 두 평균 차이의 절반 (밀기 크기)"""
    m = encode_words([a for a, _ in pairs], templates, tok, enc)["final"] @ d
    f = encode_words([b for _, b in pairs], templates, tok, enc)["final"] @ d
    return (m.mean() + f.mean()) / 2, (m.mean() - f.mean()) / 2


# ---------- 개입 (5단계) ----------
@torch.no_grad()
def prompt_embeds(prompts, tok, enc):
    """SD에 들어가는 최종 출력 (b, 77, 768)."""
    ids = tok(prompts, padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids.to(enc.device)
    return enc(ids).last_hidden_state


def job_token_start(template, job, tok):
    """프롬프트에서 직업 단어의 첫 토큰 위치. 그 앞 토큰에는 직업 정보가 없다."""
    prefix = fill(template, job).rsplit(job, 1)[0].strip()      # 예: "a photo of a"
    return len(tok(prefix).input_ids) - 1                        # BOS 포함, EOS 제외


def intervention_delta(cond, p, c, s):
    """조건별로 투영값을 얼마나 옮길지. p는 그 프롬프트 EOS의 투영값.
    original: 0, remove: c로, flip: 원래 기운 반대쪽 c ∓ s로,
    random: 무작위 방향으로 remove와 같은 거리 (방향은 intervene에서 바꿔 끼운다)."""
    if cond == "original":
        return 0.0
    if cond == "remove":
        return c - p
    if cond == "flip":
        return (c - np.sign(p - c) * s) - p
    if cond == "random":
        return abs(c - p)
    raise ValueError(cond)


def intervene(embeds, direction, delta, start_token):
    """직업 토큰부터 끝까지 모든 위치를 direction 방향으로 delta만큼 옮긴다."""
    out = embeds.clone()
    v = torch.as_tensor(direction, dtype=out.dtype, device=out.device)
    out[:, start_token:] += float(delta) * v
    return out


def load_intervention_setup(tok, enc):
    """개입에 필요한 것: 성별 방향 d, 무작위 방향 r, 중간점 c, 크기 s."""
    d = np.load(DATA / "gender_dir_final.npy")
    r = random_direction(seed=RANDOM_DIR_SEED)
    c, s = gender_anchors(DIRECTION_PAIRS, TEMPLATES, tok, enc, d)
    return {"d": d, "r": r, "c": float(c), "s": float(s)}


def job_embeds(job, cond, tok, enc, setup):
    """직업 프롬프트의 개입된 임베딩.
    cond: "original", "random", 또는 숫자 lam (투영값을 c + lam*s 로 옮김).
    random은 무작위 방향으로 lam=0(제거)과 같은 거리만큼 옮긴다.
    반환: (embeds, p_lam)  p_lam은 원래 위치를 lam 단위로 나타낸 값."""
    d, c, s = setup["d"], setup["c"], setup["s"]
    prompt = fill(GEN_TEMPLATE, job)
    e = prompt_embeds([prompt], tok, enc)
    p = float(encode([prompt], tok, enc)["final"][0] @ d)
    start = job_token_start(GEN_TEMPLATE, job, tok)
    if cond == "original":
        out = e
    elif cond == "random":
        out = intervene(e, setup["r"], abs(c - p), start)
    else:
        out = intervene(e, d, (c + float(cond) * s) - p, start)
    return out, (p - c) / s


def generate_image(pipe, embeds, seed):
    g = torch.Generator(DEVICE).manual_seed(seed)
    return pipe(prompt_embeds=embeds, num_inference_steps=NUM_INFERENCE_STEPS,
                generator=g).images[0]


def random_direction(dim=768, seed=0, n_layers=None):
    """기준선용 무작위 단위 방향."""
    rng = np.random.default_rng(seed)
    shape = (n_layers, dim) if n_layers else (dim,)
    r = rng.standard_normal(shape)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)