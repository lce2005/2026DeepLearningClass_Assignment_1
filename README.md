# ResNet18 CIFAR-10 Classification with Two-Neuron Bottleneck Analysis

이 프로젝트는 CIFAR-10 이미지 분류를 위해 파인튜닝된 사전 학습된 ResNet18 모델에 두 뉴런 병목 계층이 미치는 영향을 탐구합니다. 표준 분류 헤드와 특수화된 두 뉴런 병목 헤드의 성능을 비교하고, 학습된 특징 표현과 병목 가중치의 최적화 환경을 시각화합니다.

## Table of Contents

- [Project Overview](#project-overview)
- [Setup and Installation](#setup-and-installation)
- [Data](#data)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Results](#results)
- [Generated Files](#generated-files)
- [Conclusion](#conclusion)

## Project Overview

이 리포지토리는 ResNet18 모델을 사용하여 CIFAR-10 이미지를 분류하기 위한 코드와 분석을 포함합니다. 주요 초점은 최종 분류 계층 (즉, '병목') 이전에 특징의 차원을 줄이는 것이 성능과 해석 가능성에 어떤 영향을 미치는지 이해하는 것입니다. 우리는 고정된 ResNet18 백본 위에 두 가지 유형의 분류 '헤드'를 훈련합니다:

1.  **기준 헤드(Baseline Head)**: 단일 `Linear` 계층 (512 특징 -> 10 클래스).
2.  **두 뉴런 헤드(Two-Neuron Head)**: 두 개의 `Linear` 계층 (512 특징 -> 2 특징 -> 10 클래스)으로 구성된 병목 구조.

이 프로젝트는 또한 두 뉴런 병목에 의해 학습된 2D 특징 공간의 시각화와 경사 하강 중 한 쌍의 가중치에 대한 손실 환경 검사를 포함합니다.

## Setup and Installation

이 프로젝트를 실행하려면 Python과 PyTorch가 필요합니다. 다른 필수 라이브러리는 아래에 나열되어 있으며 `pip`을 통해 설치할 수 있습니다:

```bash
pip install torch torchvision matplotlib numpy tqdm pandas-gbq koreanize-matplotlib
```

## Data

[CIFAR-10 데이터셋](https://www.cs.toronto.edu/~kriz/cifar.html)이 사용되며, 10개 클래스에 걸쳐 60,000개의 32x32 컬러 이미지로 구성되어 있으며, 각 클래스당 6,000개의 이미지가 있습니다. 50,000개의 훈련 이미지와 10,000개의 테스트 이미지가 있습니다.

이미지는 64x64 픽셀로 변환되고, 텐서로 변환되며, ImageNet의 평균 및 표준 편차인 `MEAN=(0.485, 0.456, 0.406)` 및 `STD=(0.229, 0.224, 0.225)`를 사용하여 정규화됩니다.

## Model Architecture

모델의 핵심은 ImageNet에서 사전 학습된 `resnet18` 백본입니다. ResNet18의 최종 분류 계층은 CIFAR-10의 10개 클래스에 맞게 교체됩니다.

*고정된* ResNet18 백본의 512차원 특징 위에 두 가지 다른 분류 '헤드'가 훈련됩니다.

### Task 1. 고정 백본과 두 뉴런 병목 구조

![Model Architecture](images/task1_architecture.png)

CIFAR-10으로 미세조정한 ResNet18 백본을 고정하고, 기준 헤드 512→10과 두 뉴런 헤드 512→2→10을 각각 학습하였습니다.

## Training

훈련 과정은 크게 두 단계로 나뉩니다:

1.  **ResNet18 백본 파인튜닝**: 전체 `resnet18` 모델(백본과 새로운 10클래스 분류 계층 포함)은 `BACKBONE_EPOCHS` (3) 에포크 동안 학습률 `1e-4`로 Adam 옵티마이저를 사용하여 CIFAR-10 훈련 세트에서 파인튜닝됩니다. 이 단계는 사전 학습된 특징을 CIFAR-10 데이터셋에 맞게 조정합니다.

2.  **분류 헤드 훈련**: 파인튜닝 후, ResNet18 백본은 고정되고, 그 출력 특징(512차원)은 전체 CIFAR-10 훈련 및 테스트 세트에 대해 추출됩니다. 그런 다음, 두 가지 다른 분류 헤드(기준 및 두 뉴런)는 `HEAD_EPOCHS` (40) 에포크 동안 학습률 `1e-3`로 Adam 옵티마이저를 사용하여 이러한 추출된 특징에 대해 별도로 훈련됩니다.

## Results

### Task 2. 두 뉴런 병목의 정확도 비용

![Accuracy Comparison](images/task2_accuracy.png)

두 뉴런 모델의 테스트 정확도는 79.97%로, 기준 모델의 89.52%보다 9.55%p 낮았습니다. 이는 높은 제약이 있는 병목을 사용할 때 성능 저하가 발생함을 나타냅니다.

### Task 3. 테스트 이미지 10,000장의 2차원 표현

![2D Feature Representation](images/task3_representation.png)

테스트 이미지 10,000장의 병목 출력을 동일한 좌표 범위의 클래스별 산점도로 나타내어, 각 클래스의 위치와 분포 및 다른 클래스와 겹치는 영역을 비교하였습니다. 2D 특징 `Z` (두 뉴런 병목의 출력)의 시각화는 이 압축된 공간에서 다른 클래스가 어떻게 표현되는지를 보여줍니다. 이를 통해 병목에 의해 어떤 정보가 보존되고 손실되는지에 대한 통찰력을 얻을 수 있습니다.

### Task 4. 두 가중치의 손실 단면과 학습률별 경사하강

![Loss Landscape and Optimization](images/task4_optimization.png)

같은 시작점의 손실 0.349에서 두 가중치만 경사하강으로 갱신한 결과, 각 실행의 마지막 손실은 η=0.00282: 0.341; η=0.0282: 0.277; η=0.0846: 0.227; η=0.282: 0.202였으며 표시 범위를 이탈한 실행은 조기 중단하였습니다. 이 섹션은 병목 계층에서 두 특정 가중치의 손실 환경을 시각화하고, 다양한 학습률에서 경사 하강의 동작을 보여줍니다. 이는 다른 학습률이 수렴과 이 두 가중치에 대한 손실 표면을 가로지르는 경로에 어떻게 영향을 미치는지 보여주며, 일부 학습률은 조기 종료 또는 발산으로 이어질 수 있음을 나타냅니다.

## Generated Files

실행 시, 노트북은 지정된 `SAVE_DIR` (`/content/drive/MyDrive/Assignment1_TwoNeurons`)에 다음 파일을 생성합니다:

-   `Assignment1_figures.pdf`: 모든 생성된 플롯을 컴파일한 PDF 문서.
-   `baseline_head.pt`: 훈련된 기준 분류 헤드의 상태 딕셔너리.
-   `experiment_results.json`: 모든 실험 설정 및 정량적 결과를 포함하는 JSON 파일.
-   `figure_captions.txt`: 그림에 대한 모든 캡션을 포함하는 텍스트 파일.
-   `frozen_backbone.pt`: 고정된 ResNet18 백본의 상태 딕셔너리.
-   `resnet18_cifar10_checkpoint.pt`: 백본 파인튜닝 과정에서 저장된 체크포인트.
-   `task1_architecture.png`: 모델 아키텍처 다이어그램의 PNG 이미지.
-   `task2_accuracy.png`: 정확도 비교 막대 차트의 PNG 이미지.
-   `task3_representation.png`: 2D 특징 표현 산점도 플롯의 PNG 이미지.
-   `task4_optimization.png`: 손실 환경 및 경사 하강 경로의 PNG 이미지.
-   `two_neuron_head.pt`: 훈련된 두 뉴런 분류 헤드의 상태 딕셔너리.
-   `visualization_data.npz`: 시각화에 사용된 원시 데이터(예: 2D 특징, 손실 그리드)를 포함하는 압축된 NumPy 아카이브.

## Conclusion

이 프로젝트를 통해 ResNet18 백본 위에 얕은 신경망 헤드를 사용한 CIFAR-10 분류에서 두 뉴런 병목 계층의 영향을 분석했습니다. 주요 발견 사항은 다음과 같습니다:

-   **정확도 저하**: 두 뉴런 병목 모델은 기준 모델보다 약 9.55%p 낮은 테스트 정확도를 보였습니다. 이는 특징 공간의 극심한 차원 축소가 정보 손실과 분류 성능 저하로 이어진다는 것을 시사합니다.
-   **특징 표현**: 2D 특징 공간 시각화는 일부 클래스들이 겹치면서도 어느 정도 분리 가능한 영역을 형성함을 보여주었습니다. 이는 병목 계층이 중요한 특징을 압축하면서도 클래스 간의 차이를 어느 정도 유지함을 나타냅니다.
-   **손실 환경**: 다양한 학습률을 사용한 경사 하강 시뮬레이션은 최적화 경로가 시작점과 학습률에 민감하게 반응함을 확인했습니다. 특정 학습률은 더 빠르고 안정적인 수렴을 보였지만, 일부는 발산하거나 표시 범위를 벗어났습니다.

결론적으로, 두 뉴런 병목은 모델의 복잡성을 줄이고 해석 가능성을 높이는 데 기여할 수 있지만, 이는 상당한 성능 저하를 감수해야 할 수 있습니다. 이 연구는 모델 압축과 성능 사이의 트레이드오프를 이해하는 데 중요한 통찰력을 제공합니다.
