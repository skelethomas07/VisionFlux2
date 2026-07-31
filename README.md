# VisionFlux

## Surface to Volume

VisionFlux는 SEM 이미지에서 fiber의 두께와 방향을 정리하고, 3D 구조 생성에 사용할 분포를 만드는 Streamlit 앱입니다.

여러 이미지를 한 번에 올리면 한 장씩 순서대로 분석합니다. 화면에는 전체 진행률, 현재 처리 중인 파일, 경과 시간이 표시됩니다. 모든 분석이 끝난 뒤에는 입력한 이메일 주소로 완료 알림을 보낼 수 있습니다.

자동 분석 뒤에는 원본 SEM 위에서 결과를 바로 고칠 수 있습니다. 자동으로 잡지 못한 fiber는 양쪽 edge를 차례로 클릭해 추가하고, 잘못 잡힌 선은 지우개로 제거합니다. 여러 수정은 브라우저 안에서 임시로 보관되며 **전체 반영**을 눌렀을 때 한 번에 분포에 적용됩니다.

## 핵심 기능

- 여러 SEM 이미지 동시 업로드 및 순차 분석
- 전체 진행률 `%`, 현재 파일 번호, 처리 단계 표시
- 분석 중 경과 시간과 완료 후 총 소요 시간 표시
- 분석 완료 이메일 알림
- Edge, bright ridge, OrientationJ 방향 정보와 원본 SEM 일치도를 결합한 두께 검출
- 같은 fiber 영역에서 반복 측정된 값을 대표 두께로 정리
- 확대·축소·이동이 가능한 측정 캔버스
- 누락 fiber 수동 추가: 한쪽 edge와 반대쪽 edge를 차례로 클릭
- 지우개, 실행 취소, 임시 측정 초기화, 일괄 반영
- 두께 분포와 방향 분포를 별도 탭으로 확인
- 주방향, coherency 기반 정렬도 `S`, 방향 색상 지도 제공
- 수정된 두께 분포 CSV와 검토 세션 ZIP 저장
- 로컬 CUDA/CuPy 환경에서 OrientationJ 구조 tensor Gaussian filtering GPU 가속

## GitHub에 올리기

이 폴더의 **내용 전체**를 GitHub 저장소 최상단에 올립니다. 업로드 후 저장소 첫 화면에 다음 파일이 바로 보여야 합니다.

```text
app.py
requirements.txt
requirements-gpu.txt
README.md
pipeline/
services/
ui/
.streamlit/
```

## Streamlit Community Cloud 배포

1. Streamlit Community Cloud에서 **Create app**을 선택합니다.
2. GitHub 저장소와 `main` 브랜치를 선택합니다.
3. **Main file path**에 `app.py`를 입력합니다.
4. 이메일 알림을 사용할 경우 **Advanced settings → Secrets**에 아래 설정을 입력합니다.
5. Deploy를 누릅니다.

이 프로젝트는 Streamlit Custom Components v2를 사용하므로 `requirements.txt`에서 Streamlit 1.60.0을 고정합니다.

## 이메일 완료 알림 설정

사용자는 앱 화면에서 받는 주소만 입력합니다.

```text
skelethomas07@gmail.com
```

발신 Gmail 정보는 앱 관리자가 한 번만 Streamlit Secrets에 등록합니다. 일반 Gmail 비밀번호가 아니라 Google 계정에서 발급한 **앱 비밀번호**를 사용합니다.

Streamlit Community Cloud의 **App settings → Secrets**에 다음 내용을 입력합니다.

```toml
[email]
sender = "발신용계정@gmail.com"
app_password = "xxxx xxxx xxxx xxxx"
smtp_host = "smtp.gmail.com"
smtp_port = 465

[app]
url = "https://배포주소.streamlit.app"
```

로컬에서는 `.streamlit/secrets.toml.example`을 `.streamlit/secrets.toml`로 복사한 뒤 값을 입력합니다.

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

`.streamlit/secrets.toml`은 `.gitignore`에 포함되어 있으므로 GitHub에 올리지 않습니다.

이메일 발송에 실패해도 완료된 이미지 분석 결과는 유지됩니다. 화면에 발송 실패 원인만 표시됩니다.

## 로컬 실행

Windows에서는 `run_local.bat`을 실행하거나 다음 명령을 사용합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

## 로컬 GPU 사용

기본 `requirements.txt`는 Streamlit Community Cloud에서도 설치되도록 CUDA 패키지를 포함하지 않습니다. NVIDIA CUDA 12.x가 설치된 로컬 컴퓨터에서는 다음 명령으로 GPU 선택 패키지를 추가합니다.

```bash
python -m pip install -r requirements-gpu.txt
streamlit run app.py
```

앱에서 **가능하면 GPU 사용**을 켜면 CUDA와 CuPy를 확인합니다.

- GPU를 찾으면 방향 구조 tensor의 Gaussian derivative와 tensor averaging을 GPU에서 계산합니다.
- GPU 또는 CuPy가 없거나 실행 중 오류가 발생하면 자동으로 CPU로 전환합니다.
- 기존 fiber edge·ridge 추적 알고리즘은 SciPy/scikit-image 기반이므로 CPU에서 실행됩니다.
- Streamlit Community Cloud에서는 일반적으로 CPU 경로를 사용합니다.

CUDA 버전이 12.x가 아니라면 CuPy 공식 설치표에 맞는 패키지로 `requirements-gpu.txt`의 패키지명을 변경해야 합니다.

## 사용 순서

1. 왼쪽에서 SEM 이미지 한 장 또는 여러 장을 올립니다.
2. 완료 알림이 필요하면 이메일 주소를 입력합니다.
3. 물리 단위가 필요하면 `nm 단위 사용`을 켜고 원본 이미지의 `nm/px`를 입력합니다.
4. 로컬 GPU 환경이면 `가능하면 GPU 사용`을 켭니다.
5. `분석 시작`을 누릅니다.
6. 진행률, 현재 파일과 경과 시간을 확인합니다.
7. 분석 후 `검토할 이미지`에서 한 장을 선택합니다.
8. 두께 탭에서 누락된 fiber를 추가하거나 잘못된 선을 지웁니다.
9. 수정이 끝나면 `전체 반영`을 누릅니다.
10. 두께 분포 CSV 또는 검토 결과 ZIP을 저장합니다.

## 진행률의 의미

전체 진행률은 다음 두 값을 결합합니다.

- 전체 이미지 중 몇 번째 파일인지
- 현재 파일의 분석 단계가 얼마나 진행됐는지

기존 핵심 fiber 추적 함수 하나가 오래 실행되는 동안에는 진행률이 같은 값에 잠시 머물 수 있지만, 브라우저의 경과 시간은 계속 증가합니다. 여러 이미지는 메모리 사용량을 줄이기 위해 동시에 돌리지 않고 순서대로 처리합니다.

## 캔버스 조작

- 마우스 휠: 확대·축소
- `이동`: 드래그해서 화면 이동
- `두께 추가`: 두 edge를 순서대로 클릭
- `지우개`: 삭제할 대표 두께선을 클릭
- `실행 취소`: 직전 추가·삭제 복원
- `Esc`: 첫 번째 edge 선택 취소
- 숫자 `1`, `2`, `3`: 이동, 두께 추가, 지우개 모드 전환

## 폴더 구조

```text
VisionFlux/
├─ app.py
├─ requirements.txt
├─ requirements-gpu.txt
├─ pipeline/
│  ├─ analyzer.py
│  ├─ batch.py
│  ├─ compute.py
│  ├─ legacy_pipeline.py
│  ├─ orientation.py
│  ├─ review.py
│  └─ review_state.py
├─ services/
│  └─ notifications.py
├─ ui/
│  ├─ figures.py
│  ├─ live_timer.py
│  └─ measurement_canvas.py
├─ tests/
├─ .github/workflows/tests.yml
└─ .streamlit/
   ├─ config.toml
   └─ secrets.toml.example
```

## 성능 참고

Streamlit Community Cloud에서는 `빠름 · 최대 1200 px` 설정을 권장합니다. 분석 결과는 축소된 분석 이미지에서 계산되더라도 원본 픽셀 크기로 환산됩니다. 동일한 이미지와 설정은 Streamlit 캐시를 사용해 불필요한 재계산을 줄입니다.

## Fast direction-graph detector

The default detector is now `fast_direction_graph_v1`.

- Computes the structure tensor only once.
- Computes a six-scale bright-ridge map once and reuses it.
- Builds a conservative dark pore-core mask before centerline tracing.
- Converts the ridge skeleton into local paths instead of running the previous beam search.
- Measures all path-normal edge profiles in vectorized batches.
- Keeps one curved path while splitting its local direction into persistent direction segments.
- Weights the fiber-direction chart by detected segment length, so a curved fiber is not forced into one angle.

`pipeline/legacy_pipeline.py` is still included for emergency compatibility, but the Streamlit app does not call it during normal analysis. The new detector runs on CPU in Streamlit Community Cloud. On a local CUDA/CuPy environment, both the structure tensor and multiscale Hessian ridge calculations use the GPU.

The detector is intentionally high-recall. Low-confidence edge or pore-crossing candidates are excluded, and the remaining false positives can still be removed with the existing eraser tool.
