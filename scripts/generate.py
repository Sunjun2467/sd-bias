"""이미지 일괄 생성 (6단계).

사용 예 (C:\\sd-bias 에서, 가상환경 켠 상태):
  python scripts\\generate.py                       # 전체: 6직업 x 7조건 x 50장
  python scripts\\generate.py --jobs CEO --n 4      # 빠른 확인
이미 있는 파일은 건너뛰므로, 중간에 멈춰도 같은 명령으로 이어서 실행된다.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.prompts import GEN_JOBS, LAMS, SEEDS    # noqa: E402
from src.utils import load_pipeline, load_intervention_setup, job_embeds, generate_image  # noqa: E402

OUT = ROOT / "outputs" / "images"
META = ROOT / "data" / "generation_meta.csv"
LOG = ROOT / "outputs" / "generate.log"


def tag(cond):
    return cond if isinstance(cond, str) else f"lam{cond:+.1f}"


def main():
    all_jobs = [j for js in GEN_JOBS.values() for j in js]
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", default=all_jobs)
    ap.add_argument("--conds", nargs="+", default=None,
                    help='예: original random 0.0 -1.0  (기본: original random + LAMS 전부)')
    ap.add_argument("--n", type=int, default=len(SEEDS), help="시드 0..n-1")
    args = ap.parse_args()

    conds = ["original", "random"] + LAMS if args.conds is None else \
        [c if c in ("original", "random") else float(c) for c in args.conds]

    OUT.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG, encoding="utf-8"),
                                  logging.StreamHandler()])

    pipe = load_pipeline()
    tok, enc = pipe.tokenizer, pipe.text_encoder
    setup = load_intervention_setup(tok, enc)
    logging.info(f"c={setup['c']:.3f} s={setup['s']:.3f} jobs={args.jobs} "
                 f"conds={[tag(c) for c in conds]} n={args.n}")

    # 임베딩은 직업 x 조건마다 한 번만 계산
    embeds = {}
    for job in args.jobs:
        for cond in conds:
            embeds[job, tag(cond)] = job_embeds(job, cond, tok, enc, setup)

    new_meta = not META.exists()
    total = len(args.jobs) * len(conds) * args.n
    done = made = 0
    t0 = time.time()
    with open(META, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_meta:
            w.writerow(["file", "job", "cond", "seed", "p_lam"])
        # 시드를 바깥 루프로: 중간에 멈춰도 모든 조건이 고르게 채워진다
        for seed in range(args.n):
            for job in args.jobs:
                for cond in conds:
                    done += 1
                    name = f"{job}_{tag(cond)}_s{seed:02d}.png"
                    path = OUT / job / name
                    if path.exists():
                        continue
                    path.parent.mkdir(exist_ok=True)
                    e, p_lam = embeds[job, tag(cond)]
                    generate_image(pipe, e, seed).save(path)
                    w.writerow([f"{job}/{name}", job, tag(cond), seed, f"{p_lam:.4f}"])
                    f.flush()
                    made += 1
                    if made % 20 == 0:
                        el = time.time() - t0
                        logging.info(f"{done}/{total}  ({el / made:.2f}s/img, "
                                     f"남은 시간 약 {(total - done) * el / made / 60:.0f}분)")
    logging.info(f"완료: 새로 {made}장, 전체 {total}장")


if __name__ == "__main__":
    main()
