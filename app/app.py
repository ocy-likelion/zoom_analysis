from flask import Flask, render_template, request, jsonify
import openai
import webvtt
import os
import traceback
import time
import json
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    print("경고: OpenAI API 키가 설정되지 않았습니다!")

# 교과목 매핑 데이터
SUBJECTS = {
    "게임 기획": ["게임 디자인", "레벨 디자인", "기획", "시스템 기획", "컨텐츠 기획"],
    "프로그래밍": ["코딩", "개발", "프로그래밍", "엔진", "알고리즘"],
    "그래픽": ["아트", "애니메이션", "모델링", "이펙트", "UI/UX"],
    "사운드": ["음향", "음악", "효과음", "사운드 디자인"],
    "프로젝트 관리": ["프로젝트", "관리", "협업", "버전 관리", "품질 관리"]
}

def split_text(text, max_chunk_size=4000):
    """텍스트를 더 큰 청크로 나눕니다."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1
        if current_size > max_chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def analyze_text_chunk(chunk, system_prompt):
    """텍스트 청크를 분석하고 재시도 로직을 포함합니다."""
    max_retries = 3
    base_delay = 10  # 대기 시간 감소
    
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": chunk
                    }
                ]
            )
            return response.choices[0].message['content']
        except Exception as e:
            if "Rate limit" in str(e):
                wait_time = base_delay * (attempt + 1)
                print(f"Rate limit에 도달했습니다. {wait_time}초 대기 후 재시도합니다.")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception("최대 재시도 횟수를 초과했습니다.")

def combine_analyses(analyses):
    """청크별 분석 결과를 통합합니다."""
    combined = {
        "summary": [],
        "difficulties": [],
        "risks": []
    }
    
    for analysis in analyses:
        parts = analysis.split("\n")
        current_section = None
        
        for part in parts:
            if "1. 강의 내용 요약" in part:
                current_section = "summary"
            elif "2. 어려웠던 점" in part:
                current_section = "difficulties"
            elif "3. 발언 중 위험한 표현" in part:
                current_section = "risks"
            elif part.strip() and current_section:
                combined[current_section].append(part.strip())
    
    result = "=== 통합 분석 결과 ===\n\n"
    result += "1. 강의 내용 요약\n" + "\n".join(combined["summary"]) + "\n\n"
    result += "2. 어려웠던 점\n" + "\n".join(list(set(combined["difficulties"]))) + "\n\n"
    result += "3. 위험한 표현\n" + "\n".join(list(set(combined["risks"])))
    
    return result

def analyze_curriculum_match(content):
    """커리큘럼 내용과 교과목 매칭 분석"""
    matches = {}
    content_lower = content.lower()
    
    for subject, keywords in SUBJECTS.items():
        count = 0
        for keyword in keywords:
            count += content_lower.count(keyword.lower())
        matches[subject] = count
    
    total = sum(matches.values())
    if total == 0:
        return {}
    
    percentages = {subject: (count / total) * 100 for subject, count in matches.items()}
    return {k: round(v, 2) for k, v in percentages.items()}

def analyze_vtt(vtt_content):
    try:
        captions = []
        vtt_file = StringIO(vtt_content)
        for caption in webvtt.read_buffer(vtt_file):
            captions.append(caption.text)
        
        full_text = " ".join(captions)
        
        if not full_text.strip():
            raise ValueError("VTT 파일에서 텍스트를 추출할 수 없습니다.")
        
        chunks = split_text(full_text)
        print(f"총 {len(chunks)}개의 청크로 나누어졌습니다.")
        
        system_prompt = """
        강의 내용을 분석하여 다음 형식으로 정리해주세요:
        1. 주석 형식 및 수강생 소통: 강의 중 사용된 주석과 학생들과의 소통 방식
        2. 오늘 수업 목표: 강의에서 다룬 주요 학습 목표
        3. 예제 코드 설명: 사용된 예제 코드와 설명 방식
        4. 학습자의 질문 응답: 학습자들의 질문과 그에 대한 응답 내용

        각 섹션은 명확히 구분되어야 하며, 불릿 포인트(•)를 사용하여 정리해주세요.
        """
        
        analyses = []
        for i, chunk in enumerate(chunks, 1):
            print(f"청크 {i}/{len(chunks)} 분석 중...")
            analysis = analyze_text_chunk(chunk, system_prompt)
            if analysis:
                analyses.append(analysis)
            if i < len(chunks):
                time.sleep(10)
        
        if not analyses:
            raise ValueError("텍스트 분석에 실패했습니다.")
        
        return "\n\n".join(analyses)
        
    except Exception as e:
        print(f"VTT 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise

def analyze_curriculum(curriculum_content):
    try:
        if not curriculum_content.strip():
            raise ValueError("커리큘럼 파일이 비어있습니다.")
        
        chunks = split_text(curriculum_content)
        print(f"커리큘럼이 {len(chunks)}개의 청크로 나누어졌습니다.")
        
        system_prompt = """
        커리큘럼 내용을 분석하여 다음 형식의 JSON으로 반환해주세요:
        {
            "matched_unit": "매칭된 단원명",
            "match_percentage": 정수값(0-100),
            "criteria_matches": {
                "소프트웨어 구조 이해": true/false,
                "동시성/멀티스레드 프로그래밍": true/false,
                "게임 엔진 활용": true/false,
                "컴퓨터 그래픽스 관련": true/false
            }
        }
        """
        
        analyses = []
        for i, chunk in enumerate(chunks, 1):
            print(f"커리큘럼 청크 {i}/{len(chunks)} 분석 중...")
            analysis = analyze_text_chunk(chunk, system_prompt)
            if analysis:
                try:
                    analysis_json = json.loads(analysis)
                    analyses.append(analysis_json)
                except json.JSONDecodeError:
                    print(f"JSON 파싱 실패: {analysis}")
            if i < len(chunks):
                time.sleep(10)
        
        if not analyses:
            raise ValueError("커리큘럼 분석에 실패했습니다.")
        
        # 교과목 매칭 분석
        subject_matches = analyze_curriculum_match(curriculum_content)
        
        # 가장 높은 매칭률을 가진 분석 결과 선택
        best_match = max(analyses, key=lambda x: x.get('match_percentage', 0))
        
        # 결과 통합
        result = {
            "matched_unit": best_match.get('matched_unit', ''),
            "overall_match_percentage": best_match.get('match_percentage', 0),
            "criteria_matches": best_match.get('criteria_matches', {}),
            "subject_matches": subject_matches
        }
        
        return result
        
    except Exception as e:
        print(f"커리큘럼 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'vtt_file' not in request.files or 'curriculum_file' not in request.files:
            return jsonify({'error': '파일이 누락되었습니다.'}), 400
        
        vtt_file = request.files['vtt_file']
        curriculum_file = request.files['curriculum_file']
        
        if vtt_file.filename == '' or curriculum_file.filename == '':
            return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
            
        if not vtt_file.filename.endswith('.vtt'):
            return jsonify({'error': 'VTT 파일만 업로드 가능합니다.'}), 400
        
        try:
            vtt_content = vtt_file.read().decode('utf-8')
            curriculum_content = curriculum_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return jsonify({'error': '파일 인코딩이 올바르지 않습니다. UTF-8 형식의 파일을 업로드해주세요.'}), 400
        
        print("VTT 파일 분석 시작...")
        vtt_analysis = analyze_vtt(vtt_content)
        print("VTT 파일 분석 완료")
        
        print("커리큘럼 파일 분석 시작...")
        curriculum_analysis = analyze_curriculum(curriculum_content)
        print("커리큘럼 파일 분석 완료")
        
        return jsonify({
            'vtt_analysis': vtt_analysis,
            'curriculum_analysis': curriculum_analysis
        })
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'서버 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True) 