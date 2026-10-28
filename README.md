# AI는 왜 CEO를 남자로 그릴까?

Stable Diffusion 1.5의 텍스트 인코더 안에서 **성별 방향**을 찾아 시각화하고,
그 방향을 직접 움직여 생성 이미지의 성별이 어떻게 바뀌는지 확인한 프로젝트.

## 핵심 결과

| 질문 | 결과 |
|---|---|
| 성별 방향이 있나? | 있다. 방향 계산에 쓰지 않은 3쌍(brother/sister 등)도 3층부터 모두 남성 쪽으로 갈린다. 최종 출력에서 대표 방향 하나가 차이의 56.8%를 설명한다 |
| 몇 번째 층에서 생기나? | 성별 인식은 1~3층(probe 정확도 0.8~0.98), 직업이 성별 쪽으로 기우는 것은 5층 이후 강해진다 |
| 방향을 옮기면 이미지가 바뀌나? | 바뀐다. lam −1에서 남성 비율 0~3%, +1에서 81~100%. 무작위 방향은 원본과 차이 3%p 이하 |
| 가운데로 옮기면 반반이 되나? | 아니다. lam 0에서 6개 직업 모두 여성 다수이고, 반반이 되는 지점은 CEO +0.18 ~ nanny +0.45로 직업마다 다르다 |
| 클리핑(lam을 +0.2~+0.4로 제한) | 남성 비율 범위 0.00~0.98 → 0.19~0.82, 50%까지의 평균 거리 0.41 → 0.25. 6개 중 5개가 생성 전 예측과 ±0.04 이내 |

lam: 성별 축 위의 위치. −1 = 여성 단어 평균, 0 = 가운데, +1 = 남성 단어 평균.

## 구조

```
src/          prompts.py (실험 설정), utils.py (인코딩, 성별 방향, 개입)
scripts/      download_models.py, generate.py (본 실험), generate_clamp.py (클리핑 실험)
notebooks/    01 동작 확인 → 02 성별 방향 → 03 시각화 → 04 개입 테스트 → 05 판정·집계 → 06 클리핑
data/         성별 방향(.npy), 생성 목록과 CLIP 판정(.csv)
figures/      결과 그림, layer_slider.html (브라우저로 연다)
```

모델 가중치(`models/`)와 생성 이미지(`outputs/images*/`)는 용량 때문에 올리지 않았다.

## 재현

환경: Windows 11, Python 3.11.9, RTX 5060 (8GB), PyTorch + CUDA 12.8

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python scripts/download_model.py                         # SD 1.5, CLIP ViT-L/14
python scripts/generate.py                               # 2,100장, 약 1.5시간
python scripts/generate_clamp.py --center 0.3 --width 0.1   # 300장, 약 13분
```

그다음 `notebooks/`를 01부터 순서대로 실행한다. 02~04는 이미지 없이도 돌아간다.

- 생성 모델: Stable Diffusion 1.5 (fp16), 25 steps, 512×512, 시드 0~49
- 판정 모델: CLIP ViT-L/14, zero-shot ("a photo of a man / woman / with no person")

## 한계

- 판정기의 편향: 생성 모델과 판정기가 같은 CLIP 계열이라 편향을 공유할 수 있다. 사람 검증 없이 확신도 분석으로 대신했고, 10%대의 "사람 없음" 판정은 제외했다.
- 선형 방향 하나: 성별을 방향 하나로만 다뤘다. 이미지가 반반이 되는 지점이 직업마다 다른 것은 다른 방향, 비선형 정보, 또는 U-Net 쪽에 편향이 남아 있음을 시사하지만 본 연구에서는 다루지 않았다.
- "공정함"의 기준: 50:50을 목표로 삼았지만, 무엇이 공정한 비율인지는 가치 판단이다(e.g, 실제 성비 분포 고려 필요)


## 참고문헌

1. Bolukbasi et al. Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings. NeurIPS 2016.
2. Radford et al. Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
3. Rombach et al. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
4. Luccioni et al. Stable Bias: Evaluating Societal Representations in Diffusion Models. NeurIPS 2023.
5. Chuang et al. Debiasing Vision-Language Models via Biased Prompts. arXiv:2302.00070, 2023.
