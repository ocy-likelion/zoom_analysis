from flask import Flask, render_template, request, jsonify, Response
import openai
import webvtt
import os
import traceback
import time
import json
import pandas as pd
from dotenv import load_dotenv
from io import StringIO, BytesIO
from datetime import datetime

load_dotenv()

app = Flask(__name__)

# OpenAI 클라이언트 초기화
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("경고: OpenAI API 키가 설정되지 않았습니다!")
openai.api_key = openai_api_key

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
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": chunk}
                ],
                temperature=0.3
            )
            return response.choices[0].message['content'].strip()
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

def extract_curriculum_topics(curriculum_content):
    """커리큘럼에서 교과목명과 세부내용을 추출합니다."""
    try:
        # JSON 문자열을 파싱
        curriculum_data = json.loads(curriculum_content)
        
        # 교과목명과 세부내용 추출
        subjects = {}
        if isinstance(curriculum_data, dict) and 'units' in curriculum_data:
            for unit in curriculum_data['units']:
                if 'subject_name' in unit and unit['subject_name'] and 'details' in unit:
                    subjects[unit['subject_name']] = unit['details']
        
        if not subjects:
            raise ValueError("커리큘럼에서 교과목명과 세부내용을 찾을 수 없습니다.")
        
        system_prompt = f"""
        다음 커리큘럼의 교과목별 키워드를 추출해주세요.
        각 교과목의 세부내용을 바탕으로 핵심 키워드를 추출합니다.

        결과는 다음과 같은 JSON 형식으로 반환해주세요:
        {{
            "subject_keywords": {{
                "교과목명1": ["키워드1", "키워드2", ...],
                "교과목명2": ["키워드1", "키워드2", ...],
                ...
            }}
        }}
        """
        
        # 교과목 정보를 JSON 형식으로 변환
        subjects_json = {
            "subjects": [
                {"name": name, "details": details}
                for name, details in subjects.items()
            ]
        }
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(subjects_json, ensure_ascii=False)}
            ],
            temperature=0.3
        )
        
        topics = json.loads(response.choices[0].message['content'].strip())
        topics['subjects_details'] = subjects  # 세부내용 저장
        return topics
    except Exception as e:
        print(f"주제 추출 중 오류 발생: {str(e)}")
        return None

def analyze_curriculum_match(vtt_content, topics):
    """VTT 내용과 교과목 매칭 분석"""
    if not topics or 'subject_keywords' not in topics:
        return {}, {}
    
    # 교과목별 매칭 분석
    subject_matches = {}
    vtt_lower = vtt_content.lower()
    
    for subject, keywords in topics['subject_keywords'].items():
        count = 0
        for keyword in keywords:
            count += vtt_lower.count(keyword.lower())
        if count > 0:  # 매칭된 키워드가 있는 경우만 포함
            subject_matches[subject] = count
    
    # 세부내용 달성도 분석
    details_matches = {}
    if 'subjects_details' in topics:
        for subject, details in topics['subjects_details'].items():
            # 세부내용을 개별 항목으로 분리
            detail_items = [item.strip() for item in details.split('\n') if item.strip()]
            matches = []
            
            for item in detail_items:
                # 각 세부내용 항목에 대한 매칭 여부 확인
                system_prompt = f"""
                다음 강의 내용이 세부내용 항목을 달성했는지 판단해주세요.
                세부내용: {item}

                답변은 true/false로만 해주세요.
                """
                
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": vtt_content}
                        ],
                        temperature=0.3
                    )
                    
                    result = response.choices[0].message['content'].strip().lower()
                    matches.append(result == 'true')
                    
                except Exception as e:
                    print(f"세부내용 분석 중 오류 발생: {str(e)}")
                    matches.append(False)
            
            if matches:  # 매칭된 항목이 있는 경우만 포함
                details_matches[subject] = {
                    'matches': matches,
                    'total': len(detail_items),
                    'achieved': sum(matches)
                }
    
    return subject_matches, details_matches

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
            print(f"VTT 청크 {i}/{len(chunks)} 분석 중...")
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

def analyze_curriculum(curriculum_content, vtt_content):
    try:
        if not curriculum_content.strip():
            raise ValueError("커리큘럼 파일이 비어있습니다.")
        
        # 커리큘럼에서 교과목 정보 추출
        topics = extract_curriculum_topics(curriculum_content)
        if not topics:
            raise ValueError("커리큘럼 분석에 실패했습니다.")
        
        # 교과목 매칭 분석
        subject_matches, details_matches = analyze_curriculum_match(vtt_content, topics)
        
        # 매칭된 교과목과 달성도 계산
        matched_subjects = []
        for subject, match_info in details_matches.items():
            achievement_rate = (match_info['achieved'] / match_info['total']) * 100
            matched_subjects.append({
                'name': subject,
                'achievement_rate': round(achievement_rate, 2)
            })
        
        # 달성도 기준으로 정렬
        matched_subjects.sort(key=lambda x: x['achievement_rate'], reverse=True)
        
        # 결과 통합
        result = {
            "matched_subjects": matched_subjects,
            "subject_matches": subject_matches,
            "details_matches": details_matches
        }
        
        return result
        
    except Exception as e:
        print(f"커리큘럼 분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        raise

def excel_to_json(excel_content):
    """엑셀 파일을 JSON 형식으로 변환"""
    try:
        # BytesIO 객체로 엑셀 데이터 읽기
        excel_file = BytesIO(excel_content)
        
        # 엑셀 파일 읽기
        df = pd.read_excel(excel_file)
        
        # NaN 값을 None으로 변환
        df = df.where(pd.notnull(df), None)
        
        # 필수 컬럼 확인
        required_columns = ['교과목명', '세부내용']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"필수 컬럼 '{col}'이(가) 엑셀 파일에 없습니다.")
        
        # 데이터 전처리
        units = []
        for _, row in df.iterrows():
            if pd.notna(row['교과목명']) and pd.notna(row['세부내용']):
                unit = {
                    'subject_name': str(row['교과목명']).strip(),
                    'details': str(row['세부내용']).strip()
                }
                units.append(unit)
        
        if not units:
            raise ValueError("유효한 교과목 데이터를 찾을 수 없습니다.")
        
        # 커리큘럼 형식에 맞게 JSON 구조화
        curriculum_json = {
            "units": units
        }
        
        return json.dumps(curriculum_json, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"엑셀 변환 중 오류 발생: {str(e)}")
        raise ValueError(f"엑셀 파일 변환 실패: {str(e)}")

def send_progress(message):
    """진행 상황을 클라이언트에 전송"""
    return json.dumps({
        "status": "progress",
        "message": message
    }) + "\n"

def send_complete(data):
    """최종 결과를 클라이언트에 전송"""
    return json.dumps({
        "status": "complete",
        **data
    }) + "\n"

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
            curriculum_content = curriculum_file.read()
            
            # 파일 확장자 확인
            if curriculum_file.filename.endswith(('.xlsx', '.xls')):
                # 엑셀 파일을 JSON으로 변환
                print("Excel 파일을 JSON으로 변환 중...")
                curriculum_content = excel_to_json(curriculum_content)
            else:
                # JSON 파일인 경우 문자열로 디코딩
                curriculum_content = curriculum_content.decode('utf-8')
                # JSON 유효성 검사
                try:
                    json.loads(curriculum_content)
                except json.JSONDecodeError:
                    return jsonify({'error': '올바른 JSON 형식이 아닙니다.'}), 400
                
        except UnicodeDecodeError:
            return jsonify({'error': '파일 인코딩이 올바르지 않습니다. UTF-8 형식의 파일을 업로드해주세요.'}), 400
        
        print("VTT 파일 분석 시작...")
        vtt_analysis = analyze_vtt(vtt_content)
        print("VTT 파일 분석 완료")
        
        print("커리큘럼 파일 분석 시작...")
        curriculum_analysis = analyze_curriculum(curriculum_content, vtt_content)
        print("커리큘럼 파일 분석 완료")
        
        return jsonify({
            'progress': '분석이 완료되었습니다!',
            'vtt_analysis': vtt_analysis,
            'curriculum_analysis': curriculum_analysis
        })
        
    except Exception as e:
        print(f"분석 중 오류 발생: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': f'서버 오류가 발생했습니다: {str(e)}',
            'progress': '분석 중 오류가 발생했습니다.'
        }), 500

if __name__ == '__main__':
    app.run(debug=True) 