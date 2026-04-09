document.addEventListener('DOMContentLoaded', function() {
    // Tab Switching
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.getAttribute('data-tab');
            
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            
            if(tabId === 'analytics') {
                loadAnalytics();
            }
        });
    });

    // Single Prediction Form
    const predictForm = document.getElementById('predict-form');
    const resultArea = document.getElementById('prediction-result');
    const spinner = document.getElementById('predict-spinner');

    predictForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        spinner.style.display = 'block';
        resultArea.innerHTML = '';

        const formData = new FormData(predictForm);
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            spinner.style.display = 'none';

            if (data.error) {
                resultArea.innerHTML = `<div class="error-msg">${data.error}</div>`;
                return;
            }

            const statusClass = data.survived ? 'survived' : 'not-survived';
            const statusText = data.survived ? 'Survived' : 'Did Not Survive';

            resultArea.innerHTML = `
                <div class="result-badge" style="border-color: ${data.survived ? '#10b981' : '#f43f5e'}">
                    <span style="font-size: 2.5rem; font-weight: 800;">${data.probability}%</span>
                    <span style="font-size: 0.8rem; font-weight: 500;">${statusText}</span>
                </div>
                <div class="explanation-card">
                    <h4>Why this prediction?</h4>
                    <p>${data.explanation}</p>
                </div>
            `;
        } catch (error) {
            spinner.style.display = 'none';
            resultArea.innerHTML = `<div class="error-msg">Failed to connect to server.</div>`;
        }
    });

    // Batch Prediction
    const batchForm = document.getElementById('batch-form');
    const batchResult = document.getElementById('batch-result');

    batchForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById('csv-file');
        if (!fileInput.files[0]) return;

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        try {
            const response = await fetch('/batch-predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.results) {
                let html = '<table><thead><tr><th>Name</th><th>Prediction</th><th>Confidence</th></tr></thead><tbody>';
                data.results.forEach(row => {
                    html += `<tr><td>${row.Name}</td><td>${row.Survival_Prediction}</td><td>${row.Confidence}</td></tr>`;
                });
                html += '</tbody></table>';
                batchResult.innerHTML = html;
            }
        } catch (err) {
            batchResult.innerHTML = 'Error processing batch.';
        }
    });

    // Chart.js Analytics
    let chartsInit = false;
    async function loadAnalytics() {
        if (chartsInit) return;
        
        try {
            const response = await fetch('/analytics-data');
            const data = await response.json();

            // Gender Survival Chart
            new Chart(document.getElementById('genderChart'), {
                type: 'bar',
                data: {
                    labels: Object.keys(data.gender_survival),
                    datasets: [{
                        label: 'Survival Rate',
                        data: Object.values(data.gender_survival).map(v => v * 100),
                        backgroundColor: ['#f43f5e', '#6366f1'],
                        borderRadius: 8
                    }]
                },
                options: {
                    responsive: true,
                    scales: { y: { beginAtZero: true, max: 100 } }
                }
            });

            // Pclass Survival Chart
            new Chart(document.getElementById('classChart'), {
                type: 'doughnut',
                data: {
                    labels: ['First Class', 'Second Class', 'Third Class'],
                    datasets: [{
                        data: Object.values(data.pclass_survival).map(v => v * 100),
                        backgroundColor: ['#6366f1', '#a855f7', '#f43f5e']
                    }]
                }
            });

            // Age Distribution Chart
            new Chart(document.getElementById('ageChart'), {
                type: 'line',
                data: {
                    labels: Object.keys(data.age_dist),
                    datasets: [{
                        label: 'Passenger Count',
                        data: Object.values(data.age_dist),
                        borderColor: '#a855f7',
                        backgroundColor: 'rgba(168, 85, 247, 0.2)',
                        fill: true,
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true
                }
            });

            chartsInit = true;
        } catch (err) {
            console.error("Analytics failed:", err);
        }
    }

    // Chatbot Logic
    const chatToggle = document.getElementById('chat-toggle');
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    chatToggle.addEventListener('click', () => {
        chatWindow.style.display = chatWindow.style.display === 'flex' ? 'none' : 'flex';
    });

    chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const msg = chatInput.value.trim();
        if (!msg) return;

        // Add user message
        addMessage(msg, 'user');
        chatInput.value = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: msg })
            });
            const data = await response.json();
            addMessage(data.reply, 'bot');
        } catch (err) {
            addMessage("Sorry, I'm having trouble connecting right now.", 'bot');
        }
    });

    function addMessage(text, side) {
        const div = document.createElement('div');
        div.className = `msg ${side}`;
        div.innerText = text;
        chatMessages.appendChild(div);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
