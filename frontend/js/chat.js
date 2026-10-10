/**
 * Chat UI Module
 * Handles rendering of messages and UI state.
 */

class ChatUI {
    constructor() {
        this.messagesContainer = document.getElementById('chat-messages');
        this.ttsNoticeShown = false;
    }

    /**
     * Add a user message to the UI
     * @param {string} text
     */
    addUserMessage(text) {
        const div = document.createElement('div');
        div.className = 'message user-message';
        div.innerHTML = `
            <div class="message-avatar">👤</div>
            <div class="message-content">
                <p>${this._escapeHTML(text)}</p>
            </div>
        `;
        this.messagesContainer.appendChild(div);
        this._scrollToBottom();
    }

    /**
     * Add an assistant message to the UI
     * @param {string} text
     * @param {boolean} isError
     * @param {boolean} resumed - True if this is a final answer after a clarification
     * @param {Object|null} result - Raw API result for transparency panel
     */
    addAssistantMessage(text, isError = false, resumed = false, result = null) {
        const wrapper = document.createElement('div');
        wrapper.className = 'message-wrapper';

        const div = document.createElement('div');
        div.className = `message assistant-message ${isError ? 'error-message' : ''} ${resumed ? 'resumed-message' : ''}`;
        const badge = resumed ? `<span class="resumed-badge">✓ Resumed</span>` : '';
        let ttsBtnHTML = '';
        if (!isError && window.TTS && window.TTS.isSupported) {
            ttsBtnHTML = `<div class="tts-controls">
                <button class="tts-btn" title="Read aloud" aria-label="Read aloud">🔊</button>
                <button class="tts-stop-btn" title="Stop reading aloud" aria-label="Stop reading aloud" disabled>⏹</button>
            </div>`;
        }

        div.innerHTML = `
            <div class="message-avatar">✧</div>
            <div class="message-content">
                ${badge}
                <div class="message-text">
                    <p>${this._escapeHTML(text)}</p>
                </div>
                ${ttsBtnHTML}
            </div>
        `;
        wrapper.appendChild(div);

        if (!isError && window.TTS && !window.TTS.isSupported && !this.ttsNoticeShown) {
            const notice = document.createElement('p');
            notice.className = 'tts-unavailable';
            notice.setAttribute('role', 'status');
            notice.textContent = window.TTS.unavailableMessage;
            wrapper.appendChild(notice);
            this.ttsNoticeShown = true;
        }

        if (!isError && window.TTS && window.TTS.isSupported) {
            const ttsBtn = div.querySelector('.tts-btn');
            const ttsStopBtn = div.querySelector('.tts-stop-btn');
            if (ttsBtn) {
                const updateTtsState = (state) => {
                    // Reset all other TTS buttons
                    document.querySelectorAll('.tts-btn').forEach(btn => {
                        if (btn !== ttsBtn) {
                            btn.textContent = '🔊';
                            btn.classList.remove('playing', 'paused');
                        }
                    });
                    document.querySelectorAll('.tts-stop-btn').forEach(btn => {
                        btn.disabled = true;
                    });
                    if (ttsStopBtn) {
                        ttsStopBtn.disabled = state !== 'playing' && state !== 'paused';
                    }

                    if (state === 'playing') {
                        ttsBtn.textContent = '⏸';
                        ttsBtn.title = 'Pause reading aloud';
                        ttsBtn.setAttribute('aria-label', 'Pause reading aloud');
                        ttsBtn.classList.add('playing');
                        ttsBtn.classList.remove('paused');
                    } else if (state === 'paused') {
                        ttsBtn.textContent = '▶';
                        ttsBtn.title = 'Resume reading aloud';
                        ttsBtn.setAttribute('aria-label', 'Resume reading aloud');
                        ttsBtn.classList.add('paused');
                        ttsBtn.classList.remove('playing');
                    } else {
                        ttsBtn.textContent = '🔊';
                        ttsBtn.title = 'Read aloud';
                        ttsBtn.setAttribute('aria-label', 'Read aloud');
                        ttsBtn.classList.remove('playing', 'paused');
                    }
                };

                ttsBtn.addEventListener('click', () => {
                    window.TTS.toggle(text, updateTtsState);
                });
                ttsStopBtn?.addEventListener('click', () => window.TTS.stop());
            }
        }

        // Attach transparency panel if result is provided and not an error
        if (!isError && result && window.TransparencyPanel) {
            const panel = window.TransparencyPanel.build(result);
            if (panel) wrapper.appendChild(panel);
        }

        this.messagesContainer.appendChild(wrapper);
        this._scrollToBottom();
    }

    /**
     * Add a clarification question to the UI
     * @param {string} text
     */
    addClarificationMessage(text) {
        const div = document.createElement('div');
        div.className = 'message assistant-message clarification-message';
        let ttsBtnHTML = '';
        if (window.TTS && window.TTS.isSupported) {
            ttsBtnHTML = `<div class="tts-controls">
                <button class="tts-btn" title="Read aloud" aria-label="Read aloud">🔊</button>
                <button class="tts-stop-btn" title="Stop reading aloud" aria-label="Stop reading aloud" disabled>⏹</button>
            </div>`;
        }
        div.innerHTML = `
            <div class="message-avatar">❓</div>
            <div class="message-content">
                <p><strong>Clarification Needed:</strong></p>
                <div class="message-text">
                    <p>${this._escapeHTML(text)}</p>
                </div>
                ${ttsBtnHTML}
            </div>
        `;
        this.messagesContainer.appendChild(div);

        if (window.TTS && window.TTS.isSupported) {
            const ttsBtn = div.querySelector('.tts-btn');
            const ttsStopBtn = div.querySelector('.tts-stop-btn');
            if (ttsBtn) {
                const fullText = "Clarification Needed: " + text;
                const updateTtsState = (state) => {
                    document.querySelectorAll('.tts-btn').forEach(btn => {
                        if (btn !== ttsBtn) {
                            btn.textContent = '🔊';
                            btn.classList.remove('playing', 'paused');
                        }
                    });
                    document.querySelectorAll('.tts-stop-btn').forEach(btn => {
                        btn.disabled = true;
                    });
                    if (ttsStopBtn) {
                        ttsStopBtn.disabled = state !== 'playing' && state !== 'paused';
                    }
                    if (state === 'playing') {
                        ttsBtn.textContent = '⏸';
                        ttsBtn.title = 'Pause reading aloud';
                        ttsBtn.setAttribute('aria-label', 'Pause reading aloud');
                        ttsBtn.classList.add('playing');
                        ttsBtn.classList.remove('paused');
                    } else if (state === 'paused') {
                        ttsBtn.textContent = '▶';
                        ttsBtn.title = 'Resume reading aloud';
                        ttsBtn.setAttribute('aria-label', 'Resume reading aloud');
                        ttsBtn.classList.add('paused');
                        ttsBtn.classList.remove('playing');
                    } else {
                        ttsBtn.textContent = '🔊';
                        ttsBtn.title = 'Read aloud';
                        ttsBtn.setAttribute('aria-label', 'Read aloud');
                        ttsBtn.classList.remove('playing', 'paused');
                    }
                };

                ttsBtn.addEventListener('click', () => {
                    window.TTS.toggle(fullText, updateTtsState);
                });
                ttsStopBtn?.addEventListener('click', () => window.TTS.stop());
            }
        }
        this._scrollToBottom();
    }

    /**
     * Show a loading indicator
     * @returns {HTMLElement} The loading element, which should be removed when done
     */
    showLoading() {
        const div = document.createElement('div');
        div.className = 'message assistant-message';
        div.innerHTML = `
            <div class="message-avatar">✧</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        `;
        this.messagesContainer.appendChild(div);
        this._scrollToBottom();
        return div;
    }

    clear() {
        // Keep the welcome message, clear the rest
        const welcome = this.messagesContainer.querySelector('.welcome-message');
        this.messagesContainer.innerHTML = '';
        if (welcome) {
            this.messagesContainer.appendChild(welcome);
        }
    }

    _scrollToBottom() {
        this.messagesContainer.scrollTo({
            top: this.messagesContainer.scrollHeight,
            behavior: 'smooth'
        });
    }

    _escapeHTML(str) {
        const p = document.createElement('p');
        p.textContent = str;
        return p.innerHTML;
    }
}

window.ChatUI = ChatUI;
