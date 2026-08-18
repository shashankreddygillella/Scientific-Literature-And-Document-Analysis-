/* ============================================================
   main.js — only runs on pages that have the summarizer UI
   (summary.html). Guards prevent crashes on home/chat pages.
   ============================================================ */

// Chart variables — available globally for summary.html
let limeChart = null;
let shapChart = null;

// ── Explanation tab switching ─────────────────────────────────
function showExplanationTab(tab) {
    document.querySelectorAll('.tab-button').forEach((btn) => {
        btn.classList.remove('active');
        const text = btn.textContent.trim().toLowerCase();
        if ((tab === 'lime' && text.includes('lime')) || (tab === 'shap' && text.includes('shap'))) {
            btn.classList.add('active');
        }
    });
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    const target = document.getElementById(tab + '-content');
    if (target) target.classList.add('active');
}

// ── Bar chart ─────────────────────────────────────────────────
function createBarChart(canvasId, data, labels, title) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    if (canvasId === 'lime-chart' && limeChart) { limeChart.destroy(); limeChart = null; }
    if (canvasId === 'shap-chart' && shapChart) { shapChart.destroy(); shapChart = null; }

    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: title,
                data,
                backgroundColor: data.map(v => v >= 0.7 ? 'rgba(22,163,74,.7)' : v >= 0.4 ? 'rgba(217,119,6,.7)' : 'rgba(148,163,184,.7)'),
                borderColor:     data.map(v => v >= 0.7 ? 'rgba(22,163,74,1)'  : v >= 0.4 ? 'rgba(217,119,6,1)'  : 'rgba(148,163,184,1)'),
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: { beginAtZero: true, max: 1.0, title: { display: true, text: 'Importance Score' } },
                x: { title: { display: true, text: 'Sentence Index' } }
            },
            plugins: { legend: { display: false }, title: { display: true, text: title } }
        }
    });

    if (canvasId === 'lime-chart') limeChart = chart;
    else if (canvasId === 'shap-chart') shapChart = chart;
    return chart;
}

// ── Sentence list ─────────────────────────────────────────────
function displaySentenceList(containerId, explanations, scoreType) {
    const container = document.getElementById(containerId);
    if (!container) return;
    container.innerHTML = '';

    const sorted = [...explanations].sort((a, b) => {
        const sA = scoreType === 'lime' ? a.lime_score : a.shap_value;
        const sB = scoreType === 'lime' ? b.lime_score : b.shap_value;
        return sB - sA;
    });

    sorted.forEach(item => {
        const score = scoreType === 'lime' ? item.lime_score : item.shap_value;
        let importanceClass = 'low-importance', scoreClass = 'score-low';
        if (score >= 0.7) { importanceClass = 'high-importance';   scoreClass = 'score-high'; }
        else if (score >= 0.4) { importanceClass = 'medium-importance'; scoreClass = 'score-medium'; }

        const div = document.createElement('div');
        div.className = `sentence-item ${importanceClass}`;
        div.innerHTML = `
            <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                <div style="flex:1;">
                    <strong>Sentence ${item.index + 1}:</strong>
                    <span class="sentence-score ${scoreClass}">${(score * 100).toFixed(1)}%</span>
                    <p style="margin-top:5px;margin-bottom:0;">${item.sentence}</p>
                </div>
            </div>`;
        container.appendChild(div);
    });
}

// ── Display LIME + SHAP explanations ─────────────────────────
function displayExplanations(explanations) {
    if (!explanations || !explanations.lime_scores) return;
    const section = document.getElementById('explanation-section');
    if (!section) return;
    section.style.display = 'block';

    const limeData = explanations.lime_scores;
    const shapData = explanations.shap_values;

    const topLime = [...limeData].sort((a, b) => b.lime_score  - a.lime_score).slice(0, 20);
    const topShap = [...shapData].sort((a, b) => b.shap_value  - a.shap_value).slice(0, 20);

    createBarChart('lime-chart', topLime.map(i => i.lime_score),  topLime.map(i => `S${i.index + 1}`), 'LIME: Top 20 Most Important Sentences');
    createBarChart('shap-chart', topShap.map(i => i.shap_value), topShap.map(i => `S${i.index + 1}`), 'SHAP: Top 20 Most Contributing Sentences');

    displaySentenceList('lime-sentences', limeData, 'lime');
    displaySentenceList('shap-sentences', shapData, 'shap');

    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ── DOM-dependent setup — only runs when elements exist ───────
document.addEventListener('DOMContentLoaded', function () {
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    const pdfUpload = document.getElementById('pdf-upload');
    const fileInfo  = document.getElementById('file-info');
    const fileNameEl = document.getElementById('file-name');

    // Only attach input/file listeners if these elements are present (home.html)
    if (textInput && charCount) {
        textInput.addEventListener('input', function () {
            charCount.textContent = this.value.length;
            if (this.value.trim() && pdfUpload) {
                pdfUpload.value = '';
                if (fileInfo) fileInfo.style.display = 'none';
            }
        });

        // Ctrl+Enter shortcut — only relevant on summary.html where summarize() is defined
        textInput.addEventListener('keydown', function (e) {
            if (e.ctrlKey && e.key === 'Enter') {
                if (typeof summarize === 'function') summarize();
            }
        });
    }

    if (pdfUpload) {
        pdfUpload.addEventListener('change', function (e) {
            const file = e.target.files[0];
            if (file) {
                if (file.type !== 'application/pdf') {
                    if (typeof showError === 'function') showError('Please select a valid PDF file.');
                    pdfUpload.value = '';
                    if (fileInfo) fileInfo.style.display = 'none';
                    return;
                }
                if (fileNameEl) fileNameEl.textContent = file.name + ' (' + (file.size / 1024).toFixed(2) + ' KB)';
                if (fileInfo)   fileInfo.style.display = 'block';
                if (textInput)  textInput.value = '';
                if (charCount)  charCount.textContent = '0';
            } else {
                if (fileInfo) fileInfo.style.display = 'none';
            }
        });
    }
});
