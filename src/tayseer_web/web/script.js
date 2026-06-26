document.addEventListener('DOMContentLoaded', () => {
    const ws = new WebSocket('ws://' + window.location.host + '/ws');
    const chatArea = document.getElementById('chat-area');
    const input = document.getElementById('msg-input');
    const sendBtn = document.getElementById('send-btn');
    const connDot = document.getElementById('conn-dot');
    const connText = document.getElementById('conn-text');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.getElementById('status-text');
    const statusDetail = document.getElementById('status-detail');
    
    let isExecuting = false;

    ws.onopen = () => {
        connDot.classList.add('connected');
        connText.textContent = 'Connected';
    };

    ws.onclose = () => {
        connDot.classList.remove('connected');
        connText.textContent = 'Disconnected';
    };

    ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        
        if (payload.type === 'world_state') {
            renderWorld(payload.data);
        } else if (payload.type === 'chat') {
            addBotMessage(payload.data.content, payload.data.options);
            isExecuting = false;
            updateInputState();
        } else if (payload.type === 'status') {
            updateStatus(payload.data);
            if (payload.data.status === 'thinking') {
                addThinking();
            } else if (payload.data.status === 'executing') {
                isExecuting = true;
                updateInputState();
                removeThinking();
            } else if (['completed', 'failed', 'idle'].includes(payload.data.status)) {
                isExecuting = false;
                updateInputState();
                removeThinking();
            }
        } else if (payload.type === 'plan') {
            // Optional: show plan in a subtle way
        }
    };

    function sendMessage() {
        const text = input.value.trim();
        if (!text || isExecuting) return;
        addUserMessage(text);
        ws.send(JSON.stringify({type: 'message', data: text}));
        input.value = '';
        addThinking();
    }

    function sendOption(text) {
        if (isExecuting) return;
        addUserMessage(text);
        ws.send(JSON.stringify({type: 'message', data: text}));
        addThinking();
        document.querySelectorAll('.options').forEach(el => el.remove());
    }

    function addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'msg user';
        div.textContent = text;
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function addBotMessage(text, options) {
        removeThinking();
        const div = document.createElement('div');
        div.className = 'msg bot';
        div.textContent = text;
        
        if (options && options.length > 0) {
            const optsDiv = document.createElement('div');
            optsDiv.className = 'options';
            options.forEach(opt => {
                const btn = document.createElement('button');
                btn.className = 'option-btn';
                btn.textContent = opt;
                btn.onclick = () => sendOption(opt);
                optsDiv.appendChild(btn);
            });
            div.appendChild(optsDiv);
        }
        
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function addThinking() {
        if (document.getElementById('thinking-msg')) return;
        const div = document.createElement('div');
        div.id = 'thinking-msg';
        div.className = 'msg bot thinking';
        div.textContent = 'Tayseer is thinking...';
        chatArea.appendChild(div);
        chatArea.scrollTop = chatArea.scrollHeight;
    }

    function removeThinking() {
        const el = document.getElementById('thinking-msg');
        if (el) el.remove();
    }

    function renderWorld(objects) {
        const container = document.getElementById('world-list');
        if (!objects || Object.keys(objects).length === 0) {
            container.innerHTML = '<div style="color:#147a7a; text-align:center; padding:40px 0;">No objects</div>';
            return;
        }
        container.innerHTML = Object.entries(objects).map(([name, info]) => {
            const pos = info.position || [0,0,0];
            return `<div class="object-card">
                <div class="object-name">${name}</div>
                <div class="object-coords">[${pos.map(v => v.toFixed(2)).join(', ')}]</div>
            </div>`;
        }).join('');
    }

    function updateStatus(data) {
        statusText.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
        statusDetail.textContent = data.detail || '';
        statusDot.className = 'status-dot ' + data.status;
    }

    function updateInputState() {
        input.disabled = isExecuting;
        sendBtn.disabled = isExecuting;
    }

    // Attach Event Listeners
    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});