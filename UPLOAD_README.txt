VisionFlux 자동 측정 전체 지우기 핫픽스

변경 파일
1) ui/measurement_canvas.py
2) pipeline/review.py

동작
- '자동 측정 전체 지우기' 버튼 추가
- 확인창 후 모델 자동 측정만 화면에서 즉시 숨김
- 기존 수동 측정과 아직 반영하지 않은 수동 측정은 유지
- '실행 취소'로 전체 지우기 취소 가능
- '전체 반영'을 누르면 서버 데이터에서 source != 'manual'인 활성 자동 측정만 rejected 처리
- Supabase/브라우저 임시저장 흐름은 기존 delete_ids를 그대로 사용

GitHub 업로드
- GitHub의 ui 폴더에 measurement_canvas.py를 업로드하여 기존 파일 교체
- GitHub의 pipeline 폴더에 review.py를 업로드하여 기존 파일 교체
- tests/test_auto_clear.py는 선택 사항(검증용)
- 이후 Streamlit Manage app -> Reboot app
