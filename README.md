# 강의 분석기

VTT 파일과 커리큘럼 파일을 분석하여 강의 내용을 요약하고 분석하는 도구입니다.

## 설치 및 실행 방법

### 백엔드 설정

1. Python 가상환경 생성 및 활성화:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. 필요한 패키지 설치:
```bash
cd backend
pip install -r requirements.txt
```

3. OpenAI API 키 설정:
`.env` 파일을 backend 디렉토리에 생성하고 다음 내용을 추가:
```
OPENAI_API_KEY=your_api_key_here
```

4. 백엔드 서버 실행:
```bash
uvicorn app.main:app --reload
```

### 프론트엔드 설정

1. 필요한 패키지 설치:
```bash
cd frontend
npm install
```

2. 개발 서버 실행:
```bash
npm run dev
```

## 사용 방법

1. 브라우저에서 http://localhost:5173 접속
2. VTT 파일과 커리큘럼 파일 업로드
3. "분석하기" 버튼 클릭
4. 분석 결과 확인

## 기능

- VTT 파일 분석
  - 강의 내용 요약
  - 어려웠던 점 추출
  - 위험한 표현 식별

- 커리큘럼 분석
  - 매칭된 단항 추출
  - 게임 제작 신화 이론 관련성 분석 # zoom_analysis
