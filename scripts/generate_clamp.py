"""추가 실험: 성별 축 위치(lam)를 범위 안으로 잘라서(clamp) 생성.

각 직업의 원래 lam을 [center - width, center + width] 범위로 자른 뒤 그 위치로 옮겨 생성한다.
범위 안에 있던 직업은 그대로, 밖에 있던 직업은 가까운 경계로 옮겨진다.
본 실험(generate.py)과 같은 개입 함수(src/utils.py의 job_embeds)를 쓰고, 결과는 따로 저장한다.

사용 예 (C:\\sd-bias 에서, 가상환경 켠 상태):
  python scripts\\generate_clamp.py --center 0.3 --width 0.1
  python scripts\\generate_clamp.py --center 0 --width 0.25      # 비교용: 0 중심
이미 있는 파일은 건너뛰므로, 중간에 멈춰도 같은 명령으로 이어서 실행된다.
"""
import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src.prompts import GEN_JOBS, SEEDS    # noqa: E402
from src.utils import load_pipeline, load_intervention_setup, job_embeds, generate_image  # noqa: E402

OUT = ROOT / "outputs" / "images_clamp"
META = ROOT / "data" / "clamp_meta.csv"
LOG = ROOT / "outputs" / "generate_clamp.log"


def main():
    all_jobs = [j for js in GEN_JOBS.values() for j in js]
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", type=float, required=True, help="범위 중심 (lam 단위)")
    ap.add_argument("--width", type=float, required=True, help="중심에서 양쪽으로의 폭")
    ap.add_argument("--jobs", nargs="+", default=all_jobs)
    ap.add_argument("--n", type=int, default=len(SEEDS), help="시드 0..n-1")
    args = ap.parse_args()

    lo, hi = args.center - args.width, args.center + args.width
    tag = f"c{args.center:+.2f}_w{args.width:.2f}"

    OUT.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(LOG, encoding="utf-8"),
                                  logging.StreamHandler()])

    pipe = load_pipeline()
    tok, enc = pipe.tokenizer, pipe.text_encoder
    setup = load_intervention_setup(tok, enc)

    # 직업마다 원래 위치를 재고, 범위로 자른 위치로 옮긴 임베딩을 한 번만 만든다
    embeds = {}
    for job in args.jobs:
        _, p_lam = job_embeds(job, "original", tok, enc, setup)
        target = float(np.clip(p_lam, lo, hi))
        embeds[job] = (job_embeds(job, target, tok, enc, setup)[0], p_lam, target)
        logging.info(f"{tag}  {job:>8s}: lam {p_lam:+.2f} -> {target:+.2f}")

    new_meta = not META.exists()
    total = len(args.jobs) * args.n
    done = made = 0
    t0 = time.time()
    with open(META, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_meta:
            w.writerow(["file", "job", "tag", "center", "width", "seed", "p_lam", "target_lam"])
        for seed in range(args.n):
            for job in args.jobs:
                done += 1
                name = f"{job}_{tag}_s{seed:02d}.png"
                path = OUT / tag / job / name
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                e, p_lam, target = embeds[job]
                generate_image(pipe, e, seed).save(path)
                w.writerow([f"{tag}/{job}/{name}", job, tag, args.center, args.width, seed,
                            f"{p_lam:.4f}", f"{target:.4f}"])
                f.flush()
                made += 1
                if made % 20 == 0:
                    el = time.time() - t0
                    logging.info(f"{done}/{total}  ({el / made:.2f}s/img, "
                                 f"남은 시간 약 {(total - done) * el / made / 60:.0f}분)")
    logging.info(f"완료 ({tag}): 새로 {made}장, 전체 {total}장")


if __name__ == "__main__":
    main()