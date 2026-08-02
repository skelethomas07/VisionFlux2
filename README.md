# VisionFlux — Surface to Volume

SEM 이미지에서 fiber 두께와 국소 방향을 자동 분석하고, 사람이 확대·수정한 결과를 ImageJ 형식 CSV와 표시 이미지로 내보내는 Streamlit 앱입니다.

## 이번 버전의 핵심 기능

- 하단 배율·전압·날짜 정보 영역 자동 제외
- 스케일바/FOV 기반 실제 길이 보정
- 여러 이미지 순차 분석, 진행률·경과 시간·완료 이메일
- 방향 graph 기반 고속 fiber 중심선/두께 검출
- 전체 또는 6·9·12·16개 섹터 검토
- 마우스를 따라다니는 돋보기
- 돋보기 안의 자동·수동 두께선, 클릭점, 법선, 수정 가이드 표시
- `두께 추가`: 모델이 실제 경로를 검출한 fiber 위에서 1.5초 머물면 그 경로만 강조
- 첫 edge 클릭 후 검출 경로의 접선으로부터 법선 표시
- 경로가 없는 fiber는 자동 강조/법선 추정하지 않고 두 점을 자유롭게 클릭
- `두께 수정`: 기존 두께선 끝점 선택 후 기존 방향 가이드에 맞춰 교체
- 라벨 표시 ON/OFF
- 5분 브라우저 임시저장, Supabase 사용 시 서버 자동저장
- 두께·방향·개수 3D surface와 2D heatmap
- 라벨 포함 PNG와 라벨 없는 PNG 별도 출력
- ImageJ 형식 CSV: `label, Area, Mean, Min, Max, Angle, Length`
- 선택 사항: Supabase 기반 5인 공동 작업, 이미지 잠금, 결과 공유

## 로컬 실행

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## GitHub/Streamlit 배포

ZIP의 바깥 폴더를 저장소 안에 다시 넣지 말고, 압축을 푼 폴더 **안의 내용 전체**를 저장소 최상단에 올립니다. 저장소 첫 화면에 `app.py`, `pipeline/`, `ui/`, `services/`가 바로 보여야 합니다.

Streamlit Main file path:

```text
app.py
```

## 이메일 알림

Streamlit Community Cloud의 App settings → Secrets에 `.streamlit/secrets.toml.example`의 `[email]` 내용을 입력합니다. Gmail 일반 비밀번호가 아니라 16자리 앱 비밀번호를 사용합니다.

## Supabase 공동 작업 설정

### 1. Supabase 프로젝트 만들기

Supabase Dashboard에서 새 프로젝트를 만든 뒤 SQL Editor에서 다음 파일 전체를 실행합니다.

```text
supabase/setup.sql
```

이 SQL은 private Storage bucket 두 개, 공유 이미지/스냅샷/결과 테이블, 30분 만료 잠금 RPC를 만듭니다.

### 2. Streamlit Secrets 입력

Supabase Dashboard → Project Settings → API에서 Project URL과 `service_role` key를 확인합니다. `service_role` key는 RLS를 우회하므로 GitHub나 브라우저 코드에 절대 넣지 말고 Streamlit Secrets에만 저장합니다.

```toml
[supabase]
url = "https://YOUR_PROJECT.supabase.co"
service_role_key = "YOUR_SERVICE_ROLE_KEY"
project_id = "visionflux-shared"
```

Supabase 설정이 없으면 앱은 기존 개인 업로드/검토 모드로 그대로 작동합니다.

### 3. 공동 작업 흐름

1. 사이드바 `Supabase 공동 작업`에서 작업자 이름 입력
2. 34개 이미지를 공동 프로젝트에 업로드
3. 이미지 선택 후 `공유 이미지 열기`
4. 잠금 획득 시 편집 가능, 다른 사람이 작업 중이면 읽기 전용
5. 브라우저는 변경 직후 로컬 임시저장, 5분마다 Supabase에 서버 스냅샷 저장
6. `공유 프로젝트에 지금 저장` 또는 `작업 완료 및 잠금 해제`

현재 구조는 같은 이미지의 동시 자동 병합 대신 **이미지별 잠금**을 사용합니다. 5명이 서로 다른 이미지를 나눠 작업하는 방식이 가장 안전합니다.

## CSV 의미

- `label`: 수정 후 1부터 다시 부여한 연속 라벨
- `Area`: ImageJ의 1px 폭 line ROI 표본 면적. 보정 시 nm², 미보정 시 px²
- `Mean`, `Min`, `Max`: 두께선 위 8-bit grayscale intensity
- `Angle`: ImageJ 방식의 두께선 각도
- `Length`: 두께선 길이, 즉 fiber 두께

Fiber 접선 방향은 별도 `<name>_fiber_directions.csv`에 같은 `label`로 저장됩니다.

## GPU

Streamlit Community Cloud는 일반적으로 CPU로 실행됩니다. 로컬 NVIDIA CUDA/CuPy 환경에서는 `requirements-gpu.txt`를 추가 설치하면 structure tensor와 Hessian ridge 계산에 GPU를 사용할 수 있습니다. GPU가 없으면 자동으로 CPU로 전환됩니다.

## 테스트

```bash
python -m pytest -q
python -m compileall -q app.py pipeline services ui
```
