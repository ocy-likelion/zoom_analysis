document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analysisForm');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const results = document.getElementById('results');
    const vttAnalysis = document.getElementById('vttAnalysis');
    const curriculumAnalysis = document.getElementById('curriculumAnalysis');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const formData = new FormData(form);
        
        // 파일 유효성 검사
        const vttFile = formData.get('vtt_file');
        const curriculumFile = formData.get('curriculum_file');

        if (!vttFile || !curriculumFile) {
            alert('모든 파일을 선택해주세요.');
            return;
        }

        // UI 상태 업데이트
        loadingSpinner.classList.remove('hidden');
        results.classList.add('hidden');
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('서버 오류가 발생했습니다.');
            }

            const data = await response.json();
            
            // 결과 표시
            vttAnalysis.textContent = data.vtt_analysis;
            curriculumAnalysis.textContent = data.curriculum_analysis;
            
            results.classList.remove('hidden');
        } catch (error) {
            alert('오류가 발생했습니다: ' + error.message);
        } finally {
            loadingSpinner.classList.add('hidden');
        }
    });
}); 