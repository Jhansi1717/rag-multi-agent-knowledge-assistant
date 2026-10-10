/**
 * App Module
 * Orchestrates the overall application logic.
 */

document.addEventListener('DOMContentLoaded', () => {
    const ui = new window.ChatUI();
    let currentSessionId = localStorage.getItem('ka_session_id') || null;
    let currentClarificationId = null;
    let isProcessing = false;
    let sessions = [];

    // DOM Elements
    const input = document.getElementById('message-input');
    const sendBtn = document.getElementById('send-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const clearChatBtn = document.getElementById('clear-chat-btn');
    const historyList = document.getElementById('history-list');
    const sessionTitle = document.getElementById('current-session-title');
    const clarificationBanner = document.getElementById('clarification-banner');
    const micBtn = document.getElementById('mic-btn');
    const restartMicBtn = document.getElementById('restart-mic-btn');
    const micIcon = micBtn.querySelector('[aria-hidden="true"]');
    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const voiceSupportStatus = document.getElementById('voice-support-status');
    const speechSettings = document.getElementById('speech-settings');
    const speechLanguageSelect = document.getElementById('speech-language-select');
    const speechVoiceSelect = document.getElementById('speech-voice-select');
    const speechVoiceStatus = document.getElementById('speech-voice-status');
    const sidebar = document.getElementById('sidebar');
    const knowledgeBaseToggle = document.getElementById('knowledge-base-toggle');
    const knowledgeBasePanel = document.getElementById('knowledge-base-panel');
    const kbFileInput = document.getElementById('kb-file-input');
    const kbDropzone = document.getElementById('kb-dropzone');
    const kbUploadBtn = document.getElementById('kb-upload-btn');
    const kbFileList = document.getElementById('kb-file-list');
    const kbDocumentList = document.getElementById('kb-document-list');
    const kbAllowedExtensions = new Set(['.pdf', '.docx', '.txt', '.csv']);
    let kbFiles = [];
    let kbDocuments = [];
    let kbUploadInProgress = false;

    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener('click', () => {
            const isOpen = sidebar.classList.toggle('open');
            mobileMenuBtn.setAttribute('aria-expanded', isOpen);
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 768 &&
                sidebar.classList.contains('open') &&
                !sidebar.contains(e.target) &&
                e.target !== mobileMenuBtn) {
                sidebar.classList.remove('open');
                mobileMenuBtn.setAttribute('aria-expanded', 'false');
            }
        });
    }

    // Knowledge Base upload and session-local document metadata
    function kbFormatBytes(bytes) {
        if (bytes < 1024) return `${bytes} B`;
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    }

    function kbExtension(filename) {
        const dot = filename.lastIndexOf('.');
        return dot >= 0 ? filename.slice(dot).toLowerCase() : '';
    }

    function kbAddFiles(fileList) {
        Array.from(fileList).forEach(file => {
            const extension = kbExtension(file.name);
            const duplicate = kbFiles.some(item =>
                item.file.name === file.name &&
                item.file.size === file.size &&
                item.file.lastModified === file.lastModified
            );
            if (duplicate) return;

            const item = {
                file,
                status: kbAllowedExtensions.has(extension) ? 'selected' : 'failed',
                error: kbAllowedExtensions.has(extension)
                    ? ''
                    : 'Unsupported file type. Choose PDF, DOCX, TXT, or CSV.'
            };
            kbFiles.push(item);
        });
        kbFileInput.value = '';
        kbRenderFiles();
    }

    function kbStatusLabel(item) {
        if (item.status === 'selected') return 'Selected';
        if (item.status === 'uploading') return 'Uploading and indexing...';
        if (item.status === 'indexed') return 'Indexed';
        return 'Failed';
    }

    function kbRenderFiles() {
        kbFileList.replaceChildren();
        kbFiles.forEach(item => {
            const row = document.createElement('div');
            row.className = `kb-file-row kb-file-${item.status}`;

            const details = document.createElement('div');
            details.className = 'kb-file-details';
            const name = document.createElement('strong');
            name.textContent = item.file.name;
            const size = document.createElement('span');
            size.textContent = kbFormatBytes(item.file.size);
            details.append(name, size);

            const status = document.createElement('span');
            status.className = 'kb-file-status';
            status.textContent = kbStatusLabel(item);
            row.append(details, status);

            if (item.error) {
                const error = document.createElement('span');
                error.className = 'kb-file-error';
                error.textContent = item.error;
                row.appendChild(error);

                const remove = document.createElement('button');
                remove.type = 'button';
                remove.className = 'kb-file-remove';
                remove.setAttribute('aria-label', `Remove ${item.file.name}`);
                remove.textContent = 'Remove';
                remove.addEventListener('click', () => {
                    kbFiles = kbFiles.filter(file => file !== item);
                    kbRenderFiles();
                });
                row.appendChild(remove);
            }
            kbFileList.appendChild(row);
        });

        kbUploadBtn.disabled = kbUploadInProgress || !kbFiles.some(item => item.status === 'selected');
    }

    function kbRenderDocuments() {
        kbDocumentList.replaceChildren();
        if (!kbDocuments.length) {
            const empty = document.createElement('p');
            empty.className = 'kb-empty-state';
            empty.textContent = 'No documents uploaded in this session.';
            kbDocumentList.appendChild(empty);
            return;
        }

        kbDocuments.forEach(documentInfo => {
            const card = document.createElement('div');
            card.className = 'kb-document-card';
            card.setAttribute('role', 'listitem');

            const name = document.createElement('strong');
            name.className = 'kb-document-name';
            name.textContent = documentInfo.document_name || documentInfo.filename || 'Unnamed document';
            const status = document.createElement('span');
            status.className = 'kb-document-status';
            status.textContent = 'Indexed';
            const metadata = document.createElement('span');
            metadata.className = 'kb-document-metadata';
            const parts = [];
            if (documentInfo.file_type) parts.push(documentInfo.file_type.toUpperCase());
            if (documentInfo.domain) parts.push(documentInfo.domain);
            if (typeof documentInfo.chunk_count === 'number') {
                parts.push(`${documentInfo.chunk_count} chunks`);
            }
            metadata.textContent = parts.join(' · ');
            card.append(name, status, metadata);
            kbDocumentList.appendChild(card);
        });
    }

    async function kbUploadSelected() {
        if (kbUploadInProgress) return;
        kbUploadInProgress = true;
        kbRenderFiles();
        const selected = kbFiles.filter(item => item.status === 'selected');

        for (const item of selected) {
            item.status = 'uploading';
            item.error = '';
            kbRenderFiles();
            try {
                const response = await window.API.uploadDocument(item.file);
                item.status = 'indexed';
                kbDocuments.unshift(response);
                await kbRefreshDocuments();
                kbRenderDocuments();
            } catch (error) {
                item.status = 'failed';
                item.error = error.message || 'Document ingestion failed.';
            }
            kbRenderFiles();
        }

        kbFiles = kbFiles.filter(item => item.status !== 'indexed');
        kbUploadInProgress = false;
        kbRenderFiles();
    }

    async function kbRefreshDocuments() {
        try {
            const response = await window.API.getDocuments();
            kbDocuments = Array.isArray(response.documents) ? response.documents : [];
        } catch (error) {
            // Keep successful upload metadata visible if the list refresh fails.
            console.warn('Failed to refresh indexed documents', error);
        }
        kbRenderDocuments();
    }

    knowledgeBaseToggle.addEventListener('click', () => {
        const isOpen = !knowledgeBasePanel.hidden;
        knowledgeBasePanel.hidden = isOpen;
        knowledgeBaseToggle.setAttribute('aria-expanded', String(!isOpen));
        if (!isOpen) kbRefreshDocuments();
    });
    kbFileInput.addEventListener('change', event => kbAddFiles(event.target.files));
    kbUploadBtn.addEventListener('click', kbUploadSelected);
    kbDropzone.addEventListener('dragover', event => {
        event.preventDefault();
        kbDropzone.classList.add('dragging');
    });
    kbDropzone.addEventListener('dragleave', () => kbDropzone.classList.remove('dragging'));
    kbDropzone.addEventListener('drop', event => {
        event.preventDefault();
        kbDropzone.classList.remove('dragging');
        kbAddFiles(event.dataTransfer.files);
    });
    kbDropzone.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            kbFileInput.click();
        }
    });
    kbRenderFiles();
    kbRenderDocuments();
    kbRefreshDocuments();

    function renderSpeechSettings() {
        if (!speechSettings || !speechLanguageSelect || !speechVoiceSelect || !speechVoiceStatus || !window.TTS?.isSupported) {
            if (speechSettings) speechSettings.hidden = true;
            return;
        }

        speechSettings.hidden = false;
        const languages = window.TTS.getLanguages();
        speechLanguageSelect.replaceChildren();
        speechVoiceSelect.replaceChildren();

        if (!languages.length) {
            speechLanguageSelect.add(new Option('Voice list unavailable', ''));
            speechVoiceSelect.add(new Option('Browser default voice', ''));
            speechLanguageSelect.disabled = true;
            speechVoiceSelect.disabled = true;
            speechVoiceStatus.textContent = 'The browser default voice will be used.';
            return;
        }

        speechLanguageSelect.disabled = false;
        languages.forEach(language => speechLanguageSelect.add(new Option(language, language)));
        speechLanguageSelect.value = window.TTS.selectedLanguage;

        const voices = window.TTS.getVoicesForLanguage(speechLanguageSelect.value);
        speechVoiceSelect.add(new Option('Browser default voice', ''));
        voices.forEach(voice => {
            if (voice.voiceURI) {
                speechVoiceSelect.add(new Option(`${voice.name} (${voice.lang})`, voice.voiceURI));
            }
        });
        speechVoiceSelect.disabled = voices.length === 0;
        speechVoiceSelect.value = window.TTS.selectedVoiceURI;
        speechVoiceStatus.textContent = voices.length
            ? `${voices.length} voice${voices.length === 1 ? '' : 's'} available.`
            : 'No voices are available for this language; the browser default will be used.';
    }

    if (window.TTS?.isSupported) {
        speechLanguageSelect?.addEventListener('change', () => {
            window.TTS.setLanguage(speechLanguageSelect.value);
            renderSpeechSettings();
        });
        speechVoiceSelect?.addEventListener('change', () => {
            window.TTS.setVoice(speechVoiceSelect.value);
            renderSpeechSettings();
        });
        window.TTS.onVoicesChanged = renderSpeechSettings;
    }
    renderSpeechSettings();

    // Voice Input Integration
    let voiceInput = null;
    let baseText = '';

    if (window.VoiceInput) {
        voiceInput = new window.VoiceInput(
            (state) => {
                micBtn.classList.remove('listening', 'processing', 'starting', 'error');
                micBtn.setAttribute('aria-busy', String(state === 'starting' || state === 'processing'));
                micBtn.setAttribute('aria-pressed', String(state === 'listening'));
                if (state === 'unsupported') {
                    micBtn.style.display = 'none';
                    restartMicBtn.hidden = true;
                    voiceSupportStatus.hidden = false;
                } else if (state === 'starting') {
                    micBtn.classList.add('starting');
                    micIcon.textContent = '…';
                    micBtn.title = 'Starting microphone...';
                    micBtn.setAttribute('aria-label', 'Starting voice input');
                    restartMicBtn.hidden = true;
                } else if (state === 'listening') {
                    baseText = input.value.trim() ? input.value.trim() + ' ' : '';
                    micBtn.classList.add('listening');
                    micBtn.title = 'Listening... Click to stop.';
                    micIcon.textContent = '⏹';
                    micBtn.setAttribute('aria-label', 'Stop voice input');
                    restartMicBtn.hidden = false;
                } else if (state === 'processing') {
                    micBtn.classList.add('processing');
                    micBtn.setAttribute('aria-label', 'Voice input stopping');
                    restartMicBtn.hidden = false;
                } else if (state === 'stopped') {
                    micIcon.textContent = '🎤';
                    micBtn.title = 'Voice input';
                    micBtn.setAttribute('aria-label', 'Start voice input');
                    restartMicBtn.hidden = false;
                } else if (state === 'error') {
                    micBtn.classList.add('error');
                    micIcon.textContent = '🎤';
                    micBtn.title = 'Microphone error. Click to retry.';
                    micBtn.setAttribute('aria-label', 'Start voice input after error');
                    restartMicBtn.hidden = false;
                } else if (state === 'idle') {
                    micIcon.textContent = '🎤';
                    micBtn.title = 'Voice input';
                    micBtn.setAttribute('aria-label', 'Start voice input');
                    restartMicBtn.hidden = false;
                } else {
                    micBtn.title = 'Voice input';
                }
            },
            (finalText, interimText) => {
                const voiceText = (finalText + interimText).trim();
                input.value = baseText + voiceText;
                input.style.height = 'auto';
                input.style.height = (input.scrollHeight) + 'px';
                if (!isProcessing) {
                    sendBtn.disabled = input.value.trim() === '';
                }
            },
            (msg) => {
                console.error(msg);
                ui.addAssistantMessage(`Voice Error: ${msg}`, true);
            }
        );

        micBtn.disabled = false;
        micBtn.addEventListener('click', () => {
            if (voiceInput.state === 'starting') return;
            if (voiceInput.state === 'listening') {
                voiceInput.stopRecognition();
            } else {
                voiceInput.startRecognition();
            }
        });
        restartMicBtn.addEventListener('click', () => voiceInput.restartRecognition());
    } else if (voiceSupportStatus) {
        voiceSupportStatus.hidden = false;
    }

    // Sessions Management
    function loadSessionsList() {
        try {
            sessions = JSON.parse(localStorage.getItem('ka_sessions_list')) || [];
        } catch (e) {
            sessions = [];
        }
        renderSessionList();
    }

    function saveSessionsList() {
        localStorage.setItem('ka_sessions_list', JSON.stringify(sessions));
        renderSessionList();
    }

    function addOrUpdateSession(id, title) {
        const existing = sessions.find(s => s.id === id);
        if (existing) {
            if (title) existing.title = title;
            existing.timestamp = Date.now();
        } else {
            sessions.unshift({ id, title: title || `Session: ${id.substring(0, 8)}`, timestamp: Date.now() });
        }
        saveSessionsList();
    }

    function removeSession(id) {
        sessions = sessions.filter(s => s.id !== id);
        saveSessionsList();
    }

    function renderSessionList() {
        historyList.innerHTML = '';

        // Sort by timestamp descending
        sessions.sort((a, b) => b.timestamp - a.timestamp);

        sessions.forEach(session => {
            const li = document.createElement('li');
            const button = document.createElement('button');
            const isCurrent = session.id === currentSessionId;
            button.type = 'button';
            button.className = `history-item${isCurrent ? ' active' : ''}`;
            button.setAttribute('aria-pressed', String(isCurrent));
            if (isCurrent) {
                button.setAttribute('aria-current', 'page');
            }
            button.innerHTML = `
                <span class="history-icon">💬</span>
                <span class="history-title">${escapeHTML(session.title)}</span>
            `;
            button.addEventListener('click', () => switchSession(session.id));
            li.appendChild(button);
            historyList.appendChild(li);
        });
    }

    function escapeHTML(str) {
        const p = document.createElement('p');
        p.textContent = str;
        return p.innerHTML;
    }

    function stopAllMedia() {
        if (voiceInput && voiceInput.state === 'listening') {
            voiceInput.stopRecognition();
        }
        if (window.TTS) {
            window.TTS.stop();
        }
    }

    async function switchSession(id) {
        if (isProcessing || id === currentSessionId) return;

        stopAllMedia();
        currentSessionId = id;
        localStorage.setItem('ka_session_id', id);
        currentClarificationId = null;
        hideClarificationBanner();
        input.value = '';
        input.style.height = 'auto';
        sendBtn.disabled = true;

        renderSessionList();
        await loadSession(id);
    }

    // Init
    loadSessionsList();
    if (currentSessionId) {
        loadSession(currentSessionId);
    } else {
        sessionTitle.textContent = "New Conversation";
    }

    // Auto-resize textarea
    input.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (!isProcessing) {
            sendBtn.disabled = this.value.trim() === '';
        }
    });

    // Enter to send
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!isProcessing && input.value.trim() !== '') {
                sendMessage();
            }
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    newChatBtn.addEventListener('click', () => {
        if (isProcessing) return;
        stopAllMedia();
        currentSessionId = null;
        currentClarificationId = null;
        localStorage.removeItem('ka_session_id');
        ui.clear();
        sessionTitle.textContent = "New Conversation";
        hideClarificationBanner();
        input.value = '';
        input.disabled = false;
        sendBtn.disabled = true;
        input.focus();
        renderSessionList();
    });

    clearChatBtn.addEventListener('click', async () => {
        if (!currentSessionId || isProcessing) return;
        if (confirm("Are you sure you want to delete this conversation?")) {
            const idToDelete = currentSessionId;
            newChatBtn.click(); // Switch immediately for UI responsiveness
            removeSession(idToDelete);
            try {
                await window.API.deleteSession(idToDelete);
            } catch (err) {
                console.error("Failed to delete session on backend", err);
            }
        }
    });

    async function sendMessage() {
        if (isProcessing) return;
        const text = input.value.trim();
        if (!text) return;

        isProcessing = true;
        stopAllMedia();

        input.value = '';
        input.style.height = 'auto';
        sendBtn.disabled = true;
        input.disabled = true;

        ui.addUserMessage(text);
        const loader = ui.showLoading();

        try {
            const result = await window.API.chat(text, currentSessionId, currentClarificationId);
            loader.remove();

            if (!currentSessionId && result.session_id) {
                currentSessionId = result.session_id;
                localStorage.setItem('ka_session_id', currentSessionId);
                // The title can be the first query
                addOrUpdateSession(currentSessionId, text.substring(0, 40) + (text.length > 40 ? '...' : ''));
                sessionTitle.textContent = `Session: ${currentSessionId.substring(0, 8)}`;
            } else if (currentSessionId) {
                addOrUpdateSession(currentSessionId); // Update timestamp
            }

            if (result.status === "clarification_needed") {
                ui.addClarificationMessage(result.answer);
                currentClarificationId = result.m3?.clarification?.clarification_id || null;
                showClarificationBanner();
            } else {
                const wasResolvingClarification = currentClarificationId !== null;
                currentClarificationId = null;
                hideClarificationBanner();
                if (wasResolvingClarification) {
                    ui.addAssistantMessage(result.answer, false, true, result);
                } else {
                    ui.addAssistantMessage(result.answer, false, false, result);
                }
            }
        } catch (error) {
            loader.remove();
            ui.addAssistantMessage(`Error: ${error.message}`, true);
        } finally {
            isProcessing = false;
            input.disabled = false;
            input.focus();
            sendBtn.disabled = input.value.trim() === '';
        }
    }

    async function loadSession(sessionId) {
        ui.clear();
        ui.addAssistantMessage("Loading history...", false, false, null); // Temp loader visually
        try {
            const data = await window.API.getHistory(sessionId);
            ui.clear();
            if (data.history && data.history.length > 0) {
                sessionTitle.textContent = `Session: ${sessionId.substring(0, 8)}`;
                data.history.forEach(turn => {
                    ui.addUserMessage(turn.query);
                    if (turn.clarification_question || turn.query_classification === "ambiguous") {
                        ui.addClarificationMessage(turn.answer);
                    } else {
                        // Pass mock result without transparency for history just to ensure it renders?
                        // Or just render normally
                        ui.addAssistantMessage(turn.answer);
                    }
                });
            } else {
                // Invalid or empty session
                removeSession(sessionId);
                currentSessionId = null;
                localStorage.removeItem('ka_session_id');
                sessionTitle.textContent = "New Conversation";
                renderSessionList();
            }
        } catch (err) {
            console.error("Failed to load history", err);
            ui.clear();
            removeSession(sessionId);
            currentSessionId = null;
            localStorage.removeItem('ka_session_id');
            sessionTitle.textContent = "New Conversation";
            renderSessionList();
        }
    }

    function showClarificationBanner() {
        if (clarificationBanner) {
            clarificationBanner.style.display = 'flex';
            input.placeholder = 'Answer the clarification question...';
        }
    }

    function hideClarificationBanner() {
        if (clarificationBanner) {
            clarificationBanner.style.display = 'none';
            input.placeholder = 'Ask a question...';
        }
    }
});
