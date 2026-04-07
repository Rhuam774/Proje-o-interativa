const canvas = document.getElementById('interaction-canvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('video-feed');
const tagModal = document.getElementById('tag-modal');
const tagInput = document.getElementById('tag-input');
const logOutput = document.getElementById('log-output');
const calibWizard = document.getElementById('calibration-wizard');
const wizardInstruction = document.getElementById('wizard-instruction');
const steps = document.querySelectorAll('.step');

let isSelecting = false;
let startX, startY, currentX, currentY;
let pendingROI = null;
let isCalibrating = false;
let calibPoints = [];

// Sincronizar tamanho do canvas com o vídeo
function resizeCanvas() {
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
video.addEventListener('load', resizeCanvas);
setTimeout(resizeCanvas, 1000);

// Log Helper
function addLog(msg, type = 'info') {
    const p = document.createElement('p');
    p.className = type;
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logOutput.prepend(p);
}

// Interação com o Canvas (Seleção de ROI)
canvas.addEventListener('mousedown', async (e) => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isCalibrating) {
        // Enviar clique de calibração
        const scaleX = 640 / canvas.width;
        const scaleY = 480 / canvas.height;
        const realX = Math.round(x * scaleX);
        const realY = Math.round(y * scaleY);
        
        calibPoints.push({x, y});
        const res = await fetch('/record_calib', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: realX, y: realY })
        });
        const data = await res.json();
        if (data.result === 'next') {
            addLog(`Ponto ${calibPoints.length} registrado.`, 'success');
        } else if (data.result === 'finished') {
            addLog('Calibração CONCLUÍDA!', 'success');
            calibPoints = [];
            isCalibrating = false;
        }
        return;
    }

    startX = x;
    startY = y;
    isSelecting = true;
});

canvas.addEventListener('mousemove', (e) => {
    if (!isSelecting) return;
    const rect = canvas.getBoundingClientRect();
    currentX = e.clientX - rect.left;
    currentY = e.clientY - rect.top;
    drawSelection();
});

canvas.addEventListener('mouseup', () => {
    if (!isSelecting) return;
    isSelecting = false;
    
    // Calcular ROI real (considerando redimensionamento do vídeo)
    const scaleX = 640 / canvas.width; // Assumindo 640x480 padrão ou dinâmico
    const scaleY = 480 / canvas.height;
    
    const x = Math.min(startX, currentX);
    const y = Math.min(startY, currentY);
    const w = Math.abs(startX - currentX);
    const h = Math.abs(startY - currentY);

    if (w > 10 && h > 10) {
        pendingROI = [
            Math.round(x * scaleX),
            Math.round(y * scaleY),
            Math.round(w * scaleX),
            Math.round(h * scaleY)
        ];
        showTagModal();
    }
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

function drawSelection() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (isCalibrating) {
        drawCalibProgress();
        return;
    }

    ctx.strokeStyle = '#00d2ff';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
    ctx.fillStyle = 'rgba(0, 210, 255, 0.1)';
    ctx.fillRect(startX, startY, currentX - startX, currentY - startY);
}

function drawCalibProgress() {
    if (calibPoints.length === 0) return;
    
    ctx.strokeStyle = '#ff3b3b';
    ctx.fillStyle = '#ff3b3b';
    ctx.lineWidth = 3;
    ctx.setLineDash([]);

    ctx.beginPath();
    calibPoints.forEach((pt, i) => {
        ctx.arc(pt.x, pt.y, 5, 0, Math.PI * 2);
        ctx.fill();
        if (i === 0) ctx.moveTo(pt.x, pt.y);
        else ctx.lineTo(pt.x, pt.y);
    });
    
    if (calibPoints.length === 4) ctx.closePath();
    ctx.stroke();
}

// Modal de Tag
function showTagModal() {
    tagModal.classList.remove('hidden');
    tagInput.focus();
}

document.getElementById('btn-confirm-tag').onclick = async () => {
    const label = tagInput.value || "Objeto";
    const response = await fetch('/add_tracker', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ roi: pendingROI, label: label })
    });
    
    if (response.ok) {
        addLog(`Rastreando objeti: ${label}`, 'success');
        updateStatus();
    }
    closeModal();
};

document.getElementById('btn-cancel-tag').onclick = closeModal;
function closeModal() {
    tagModal.classList.add('hidden');
    tagInput.value = '';
    pendingROI = null;
}

// Botões de Ação
document.getElementById('btn-toggle-yolo').onclick = async () => {
    const res = await fetch('/toggle_yolo', { method: 'POST' });
    const data = await res.json();
    document.getElementById('yolo-status').textContent = data.state ? 'ATIVO' : 'DESLIGADO';
    addLog(`YOLO: ${data.state ? 'Ativado' : 'Desativado'}`);
};

document.getElementById('btn-toggle-sim').onclick = async () => {
    const res = await fetch('/toggle_simulation', { method: 'POST' });
    const data = await res.json();
    document.getElementById('simulation-status').textContent = data.state ? 'SIMULAÇÃO' : 'REAL';
    addLog(`Modo: ${data.state ? 'Simulação' : 'Real'}`);
};

document.getElementById('btn-reset').onclick = async () => {
    await fetch('/reset', { method: 'POST' });
    addLog('Rastreadores resetados.', 'error');
    updateStatus();
};

// Calibração Wizard
document.getElementById('btn-start-calibration').onclick = async () => {
    await fetch('/start_calib', { method: 'POST' });
    calibWizard.classList.remove('hidden');
    addLog('Iniciando calibração assistida...', 'success');
};

document.getElementById('btn-record-point').onclick = async () => {
    // Mantém compatibilidade com marcador se o usuário preferir apertar o botão
    const res = await fetch('/record_calib', { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}) 
    });
    const data = await res.json();
    
    if (data.result === 'next') {
        addLog('Ponto registrado via marcador!', 'success');
    } else if (data.result === 'finished') {
        addLog('Calibração CONCLUÍDA!', 'success');
        calibWizard.classList.add('hidden');
        calibPoints = [];
    } else if (data.result === 'no_object') {
        addLog('Erro: Marcador não detectado. Tente clicar na cruz vermelha!', 'error');
    }
};

document.getElementById('btn-cancel-calib').onclick = () => {
    calibWizard.classList.add('hidden');
};

// Desligar
document.getElementById('btn-power-off').onclick = async () => {
    if(confirm("Deseja desligar o sistema completamente?")) {
        addLog("Desligando sistema...", "error");
        await fetch('/shutdown', { method: 'POST' });
        setTimeout(() => {
            window.close();
            // Fallback caso o navegador bloqueie window.close()
            document.body.innerHTML = "<h1 style='color:white; text-align:center; margin-top:20%;'>Sistema Desligado. Pode fechar esta aba.</h1>";
        }, 500);
    }
};

// Troca de Monitor
document.getElementById('select-monitor').onchange = async (e) => {
    const idx = parseInt(e.target.value);
    const res = await fetch('/set_monitor', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ index: idx })
    });
    if (res.ok) {
        addLog(`Monitor de projeção alterado para: ${e.target.options[idx].text}`, 'success');
        addLog('Reiniciando projetor...', 'info');
        addLog('IMPORTANTE: Nova calibração necessária!', 'error');
    }
};

// Atualização Periódica de Status
async function updateStatus() {
    try {
        const res = await fetch('/status');
        const data = await res.json();
        
        // Gerenciar Wizard Automático
        if (data.calibrating) {
            isCalibrating = true;
            calibWizard.classList.remove('hidden');
            steps.forEach((s, i) => {
                s.classList.toggle('active', i === data.calib_step);
            });
            wizardInstruction.textContent = `CLIQUE na Cruz Vermelha nº ${data.calib_step + 1} que você vê no projetor.`;
            
            // Força o desenho do progresso
            drawCalibProgress();
        } else {
            if (isCalibrating) {
                isCalibrating = false;
                calibPoints = [];
            }
            calibWizard.classList.add('hidden');
        }

        // Atualizar status do YOLO
        document.getElementById('yolo-status').textContent = data.yolo ? 'ATIVO' : 'DESLIGADO';

        const list = document.getElementById('list-trackers');
        list.innerHTML = data.trackers.length ? '' : '<li class="empty-msg">Nenhum objeto selecionado</li>';
        data.trackers.forEach(t => {
            const li = document.createElement('li');
            li.textContent = t;
            list.appendChild(li);
        });

        // Atualizar Select de Monitores se estiver vazio
        const select = document.getElementById('select-monitor');
        if (select.options.length <= 1) {
            select.innerHTML = '';
            data.monitors.forEach((m, i) => {
                const opt = document.createElement('option');
                opt.value = i;
                opt.textContent = m;
                select.appendChild(opt);
            });
            // Definir o valor atual baseado na config
            select.value = data.projector_index || 0;
        }
    } catch (e) {
        console.error("Falha ao atualizar status", e);
    }
}

setInterval(updateStatus, 3000);
updateStatus();
addLog('Painel de Controle pronto.', 'success');
