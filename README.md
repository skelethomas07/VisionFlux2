# VisionFlux — Surface to Volume

VisionFlux는 2D SEM 이미지에서 fiber의 두께와 국소 방향을 자동 검출하고, 사용자가 결과를 직접 검토·수정한 뒤 ImageJ 형식의 측정표와 라벨 이미지를 내보내는 Streamlit 앱입니다.

## 이번 버전의 핵심 흐름

1. SEM 이미지 한 장 또는 여러 장 업로드
2. 하단의 배율·전압·날짜·스케일바 정보 영역 자동 분리
3. 실제 SEM 본문만 대상으로 고속 방향 graph 검출 수행
4. 스케일바 또는 `FOV:12.8x9.6um` 같은 하단 정보를 OCR로 읽어 nm/px 계산
5. 자동 두께선과 연속 라벨 표시
6. 전체 보기 또는 6·9·12·16개 섹터로 나누어 검토
7. 누락 fiber 추가, 오검출 삭제, 클릭한 fiber 강조
8. 마우스 주변 돋보기 ON/OFF
9. 브라우저에 5분마다 임시저장
10. ImageJ 형식 CSV, 방향 매칭 CSV, 라벨·두께 표시 PNG, 전체 검토 ZIP 출력

## 권장 이미지 품질

- 분석 영역의 긴 변: **1200px 이상 권장**
- 가장 얇은 fiber: **최소 6px, 가능하면 8px 이상**
- 초점이 선명하고 fiber와 배경의 명암 대비가 충분한 원본 사용
- 화면 캡처보다 장비에서 저장한 원본 TIFF/JPG/PNG 권장

앱은 업로드 후 해상도, 선명도, 대비, 포화 픽셀과 추정 최소 구조 폭을 확인해 `양호/보통/주의` 안내를 표시합니다. 이 값은 측정 합격·불합격 판정이 아니라 검토 참고값입니다.

## 하단 정보 영역과 길이 보정

VisionFlux는 이미지 하단의 검은 정보 띠를 자동으로 감지합니다.

- fiber 검출, 방향 계산, 두께 측정: **하단 정보 영역 제외**
- 스케일바, `100 nm`, `1 µm`, FOV 정보: **길이 보정에만 사용**

자동 보정 우선순위:

1. `FOV:12.8x9.6um`처럼 가로·세로 시야 크기를 읽어 이미지 크기와 비교
2. 스케일바의 픽셀 길이와 옆 단위 값을 조합
3. 자동 인식이 실패하거나 틀리면 이미지별 `nm/px`를 사용자가 수정
4. 보정하지 않을 경우 px/px² 단위 유지

OCR은 Tesseract를 사용합니다. Streamlit Community Cloud 배포 시 `packages.txt`가 `tesseract-ocr`를 설치합니다.

## 검토 캔버스

### 이동·선택

- 드래그: 화면 이동
- 마우스 휠: 확대·축소
- 두께선 클릭: 해당 fiber 중심선과 두께선을 강조
- 선택 정보: 라벨, 두께, 해당 위치의 fiber 방향 표시

### 누락 fiber 추가

1. `두께 추가` 선택
2. fiber의 한쪽 edge 클릭
3. 반대쪽 edge 클릭
4. 여러 선을 계속 추가
5. `전체 반영`을 눌러 한 번에 결과에 적용

수동 두께선의 방향에 수직인 축을 fiber 방향으로 저장합니다.

### 지우개

`지우개`를 선택한 뒤 자동 또는 수동 두께선을 클릭합니다. `전체 반영` 전에는 실행 취소할 수 있습니다. 반영 후에는 활성 결과를 다시 정리하고 화면 라벨을 1부터 연속으로 재부여합니다.

### 섹터 검토

검토 보기에서 다음 중 하나를 선택할 수 있습니다.

- 전체
- 6개: 3열 × 2행
- 9개: 3열 × 3행
- 12개: 4열 × 3행
- 16개: 4열 × 4행

`완료·다음`을 누르면 현재 섹터를 완료 처리하고 다음 섹터로 이동합니다. 섹터 모드를 사용하지 않고 전체 화면에서 줌과 돋보기만 사용해도 됩니다.

### 돋보기

`돋보기 ON/OFF` 버튼으로 커서 주변 확대창을 켜거나 끕니다. 돋보기 중앙의 십자선을 보고 fiber edge를 정확히 선택할 수 있습니다.

### 5분 임시저장

캔버스 상태는 브라우저 `localStorage`에 5분마다 저장됩니다.

저장 항목:

- 아직 반영하지 않은 수동 두께선
- 삭제 예정 선
- 섹터 위치와 완료 상태
- 돋보기 설정

같은 이미지와 같은 분석 revision에서 페이지를 다시 열면 저장 상태를 복원합니다. 원본 이미지 자체는 브라우저 저장소에 저장하지 않습니다.

## ImageJ 형식 CSV

다운로드 파일명:

```text
<image>_ImageJ_results.csv
```

열 순서:

```text
label,Area,Mean,Min,Max,Angle,Length
```

| 열 | 의미 |
|---|---|
| `label` | 수정 완료 후 이미지에 표시되는 연속 VisionFlux 라벨 번호 |
| `Area` | ImageJ의 selection area 의미. 1px 폭 straight-line ROI가 차지하는 표본 픽셀 면적. 보정 시 nm², 미보정 시 px² |
| `Mean` | 두께선 위 8-bit grayscale intensity 평균 |
| `Min` | 두께선 위 최소 grayscale intensity |
| `Max` | 두께선 위 최대 grayscale intensity |
| `Angle` | ImageJ straight-line ROI와 같은 두께선 각도. fiber 방향이 아니라 두께 측정선 방향 |
| `Length` | 두께선 길이, 즉 해당 위치의 fiber 두께. 보정 시 nm, 미보정 시 px |

ImageJ의 line selection에서는 Length와 Angle을 기록할 수 있고 Mean/Min/Max는 선 위 픽셀의 gray value 통계입니다. VisionFlux는 이 의미를 유지합니다. `label`은 결과 이미지의 번호와 CSV 행을 직접 대응시키기 위해 연속 번호로 사용합니다.

## 방향 매칭 CSV

ZIP 안의 별도 파일:

```text
<image>_fiber_directions.csv
```

주요 열:

- `label`
- `fiber_direction_deg`: 두께선 위치의 fiber 접선 방향
- `thickness_line_angle_deg`: ImageJ CSV의 Angle과 같은 두께선 방향
- `thickness`
- `length_unit`
- `fiber_region_id`
- `source`

따라서 두께와 해당 위치의 방향을 같은 label로 연결할 수 있습니다.

## 결과 파일

### 개별 다운로드

- ImageJ 형식 CSV
- 최종 라벨·두께 표시 PNG
- 전체 검토 ZIP

### 전체 ZIP 내용

- corrected measurements
- region representatives
- ImageJ results CSV
- fiber direction CSV
- 라벨·두께 표시 PNG
- feedback/history JSON
- analysis summary JSON
- measurement units JSON

## 로컬 실행

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

로컬에서 자동 OCR을 사용하려면 Tesseract도 설치해야 합니다. OCR이 없어도 앱은 실행되며 nm/px를 직접 입력할 수 있습니다.

## NVIDIA GPU 로컬 실행

```bash
pip install -r requirements-gpu.txt
streamlit run app.py
```

GPU는 structure tensor와 다중 scale Hessian ridge 계산에 사용됩니다. Streamlit Community Cloud는 일반적으로 CPU로 실행되며, GPU가 없으면 자동으로 CPU 경로를 사용합니다.

## Streamlit Community Cloud 배포

1. 이 프로젝트의 내부 파일 전체를 GitHub 저장소 최상단에 업로드
2. Streamlit Community Cloud에서 저장소 선택
3. Main file path: `app.py`
4. 필요하면 App settings → Secrets에 Gmail 발신 설정 입력

```toml
[email]
sender = "sender@gmail.com"
app_password = "16자리 Google 앱 비밀번호"
smtp_host = "smtp.gmail.com"
smtp_port = 465

[app]
url = "https://your-app.streamlit.app"
```

실제 `.streamlit/secrets.toml`은 GitHub에 올리지 않습니다.
