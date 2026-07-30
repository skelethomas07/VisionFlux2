# VisionFlux

## Surface to Volume

VisionFlux는 SEM 이미지에서 fiber의 두께와 방향을 정리하고, 3D 구조 생성에 사용할 분포를 만드는 Streamlit 앱입니다.

자동 분석 뒤에는 원본 SEM 위에서 결과를 바로 고칠 수 있습니다. 자동으로 잡지 못한 fiber는 양쪽 edge를 차례로 클릭해 추가하고, 잘못 잡힌 선은 지우개로 제거합니다. 여러 수정은 브라우저 안에서 임시로 보관되며 **전체 반영**을 눌렀을 때 한 번에 분포에 적용됩니다.

## 핵심 기능

- Edge, bright ridge, OrientationJ 방향 정보와 원본 SEM 일치도를 결합한 두께 검출
- 같은 fiber 영역에서 반복 측정된 값을 대표 두께로 정리
- 확대·축소·이동이 가능한 측정 캔버스
- 누락 fiber 수동 추가: 한쪽 edge와 반대쪽 edge를 차례로 클릭
- 지우개, 실행 취소, 임시 측정 초기화, 일괄 반영
- 두께 분포와 방향 분포를 별도 탭으로 확인
- 주방향, coherency 기반 정렬도 `S`, 방향 색상 지도 제공
- 수정된 두께 분포 CSV와 검토 세션 ZIP 저장

## GitHub에 올리기

이 폴더의 **내용 전체**를 GitHub 저장소 최상단에 올립니다. 업로드 후 저장소 첫 화면에 다음 파일이 바로 보여야 합니다.

```text
app.py
requirements.txt
README.md
pipeline/
ui/
.streamlit/
```

## Streamlit Community Cloud 배포

1. Streamlit Community Cloud에서 **Create app**을 선택합니다.
2. GitHub 저장소와 `main` 브랜치를 선택합니다.
3. **Main file path**에 `app.py`를 입력합니다.
4. Deploy를 누릅니다.

이 프로젝트는 Streamlit Custom Components v2를 사용하므로 `requirements.txt`에서 Streamlit 1.60.0을 고정합니다.

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

## 사용 순서

1. 왼쪽에서 SEM 이미지를 올립니다.
2. 물리 단위가 필요하면 `nm 단위 사용`을 켜고 원본 이미지의 `nm/px`를 입력합니다.
3. `분석 시작`을 누릅니다.
4. 두께 탭에서 결과를 확인합니다.
5. 누락된 fiber는 `두께 추가`를 선택하고 양쪽 edge를 차례로 클릭합니다.
6. 잘못된 선은 `지우개`로 클릭합니다.
7. 수정이 끝나면 `전체 반영`을 누릅니다.
8. 두께 분포 CSV 또는 검토 결과 ZIP을 저장합니다.

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
├─ pipeline/
│  ├─ analyzer.py
│  ├─ legacy_pipeline.py
│  ├─ orientation.py
│  └─ review.py
├─ ui/
│  ├─ figures.py
│  └─ measurement_canvas.py
├─ tests/
├─ .github/workflows/tests.yml
└─ .streamlit/config.toml
```

## 성능 참고

Streamlit Community Cloud에서는 `빠름 · 최대 1200 px` 설정을 권장합니다. 분석 결과는 축소된 분석 이미지에서 계산되더라도 원본 픽셀 크기로 환산됩니다. 분석 결과는 캐시되므로 같은 이미지와 같은 설정으로 다시 실행할 때 불필요한 계산을 줄입니다.
