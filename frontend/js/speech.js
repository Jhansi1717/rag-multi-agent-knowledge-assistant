/**
 * Text-to-Speech Module
 * Wrapper around window.speechSynthesis
 */

class SpeechOutput {
    static instance = null;

    constructor() {
        if (SpeechOutput.instance) {
            return SpeechOutput.instance;
        }

        this.browserWindow = typeof window !== 'undefined' ? window : null;
        this.synth = this.browserWindow ? this.browserWindow.speechSynthesis : null;
        this.utteranceConstructor = this.browserWindow
            ? this.browserWindow.SpeechSynthesisUtterance
            : null;
        this.isSupported = Boolean(
            this.synth &&
            typeof this.synth.speak === 'function' &&
            typeof this.synth.pause === 'function' &&
            typeof this.synth.resume === 'function' &&
            typeof this.synth.cancel === 'function' &&
            typeof this.utteranceConstructor === 'function'
        );
        this.unavailableMessage = 'Text-to-Speech is not supported in this browser.';
        this.currentQueue = [];
        this.currentIndex = 0;
        this.isPlaying = false;
        this.isPaused = false;
        this.playbackId = 0;
        this.selectedLanguage = 'en-US';
        this.selectedVoiceURI = '';
        this.onVoicesChanged = null;

        // Ensure voices are loaded
        this.voices = [];
        if (this.isSupported) {
            this._loadVoices();
        }
        if (this.isSupported && typeof this.synth.addEventListener === 'function') {
            this.synth.addEventListener('voiceschanged', () => this._loadVoices());
        } else if (this.isSupported && this.synth.onvoiceschanged !== undefined) {
            this.synth.onvoiceschanged = () => this._loadVoices();
        }

        // Cleanup on unload
        if (this.browserWindow) {
            this.browserWindow.addEventListener('beforeunload', () => this.stop());
        }

        SpeechOutput.instance = this;
    }

    _loadVoices() {
        try {
            this.voices = this.isSupported && typeof this.synth.getVoices === 'function'
                ? this.synth.getVoices()
                : [];
        } catch (error) {
            this.voices = [];
        }

        const languages = this.getLanguages();
        if (languages.length && !languages.includes(this.selectedLanguage)) {
            this.selectedLanguage = languages[0];
        }
        if (!this.voices.some(voice => voice.voiceURI === this.selectedVoiceURI)) {
            this.selectedVoiceURI = '';
        }
        if (this.onVoicesChanged) this.onVoicesChanged(this.voices);
    }

    getLanguages() {
        return [...new Set(this.voices.map(voice => voice.lang).filter(Boolean))];
    }

    getVoicesForLanguage(language = this.selectedLanguage) {
        return this.voices.filter(voice => voice.lang === language);
    }

    setLanguage(language) {
        if (!this.getLanguages().includes(language)) return false;
        this.selectedLanguage = language;
        const voice = this.voices.find(item => item.voiceURI === this.selectedVoiceURI);
        if (voice && voice.lang !== language) this.selectedVoiceURI = '';
        if (this.isPlaying || this.isPaused) this.stop();
        return true;
    }

    setVoice(voiceURI) {
        const voice = voiceURI
            ? this.voices.find(item => item.voiceURI === voiceURI)
            : null;
        if (voiceURI && !voice) return false;
        if (voice) this.selectedLanguage = voice.lang || this.selectedLanguage;
        this.selectedVoiceURI = voiceURI || '';
        if (this.isPlaying || this.isPaused) this.stop();
        return true;
    }

    _getVoice() {
        return this.voices.find(voice => voice.voiceURI === this.selectedVoiceURI) || null;
    }

    /**
     * Splits text into smaller chunks to avoid browser TTS 15-second cutoff bug
     */
    _chunkText(text) {
        // Split by sentence endings or newlines
        const chunks = text.match(/[^.!?\n]+[.!?\n]+/g) || [text];
        return chunks.map(c => c.trim()).filter(c => c.length > 0);
    }

    play(text, onStateChange) {
        if (!this.isSupported) {
            console.error(this.unavailableMessage);
            if (onStateChange) onStateChange('unsupported');
            return;
        }

        // If something is already playing, stop it completely
        this.stop();

        this.onStateChange = onStateChange; // (state: 'playing'|'paused'|'stopped')
        this.currentQueue = this._chunkText(text);
        this.currentIndex = 0;
        const playbackId = ++this.playbackId;

        this._playNextChunk(playbackId);
    }

    _playNextChunk(playbackId) {
        if (playbackId !== this.playbackId) return;

        if (this.currentIndex >= this.currentQueue.length) {
            this.stop(); // Finished all chunks
            return;
        }

        const text = this.currentQueue[this.currentIndex];
        const utterance = new this.utteranceConstructor(text);

        const voice = this._getVoice();
        if (voice) utterance.voice = voice;

        utterance.lang = voice?.lang || this.selectedLanguage || 'en-US';
        utterance.rate = 1.0;

        utterance.onstart = () => {
            if (playbackId !== this.playbackId) return;
            this.isPlaying = true;
            if (!this.isPaused && this.onStateChange) this.onStateChange('playing');
        };

        utterance.onend = () => {
            if (playbackId !== this.playbackId) return;
            // Only proceed to next if we aren't stopped intentionally
            if (this.isPlaying && !this.isPaused) {
                this.currentIndex++;
                this._playNextChunk(playbackId);
            }
        };

        utterance.onerror = (e) => {
            if (playbackId !== this.playbackId) return;
            console.warn('Speech synthesis error:', e);
            this.stop();
        };

        this.isPlaying = true;
        this.isPaused = false;
        if (this.onStateChange) this.onStateChange('playing');
        try {
            this.synth.speak(utterance);
        } catch (error) {
            console.warn('Failed to start speech synthesis:', error);
            this.stop();
        }
    }

    pause() {
        if (this.isSupported && this.isPlaying && !this.isPaused) {
            this.synth.pause();
            this.isPaused = true;
            if (this.onStateChange) this.onStateChange('paused');
        }
    }

    resume() {
        if (this.isSupported && this.isPaused) {
            this.synth.resume();
            this.isPaused = false;
            if (this.onStateChange) this.onStateChange('playing');
        }
    }

    stop() {
        this.playbackId++;
        if (this.isSupported) {
            this.synth.cancel();
        }
        this.currentQueue = [];
        this.currentIndex = 0;
        this.isPlaying = false;
        this.isPaused = false;
        if (this.onStateChange) this.onStateChange('stopped');
        this.onStateChange = null;
    }

    toggle(text, onStateChange) {
        // If it's already playing THIS text, pause it.
        // If it's paused, resume it.
        // If it's playing OTHER text, or stopped, play new text.

        if (this.isPlaying && this.onStateChange === onStateChange) {
            if (this.isPaused) {
                this.resume();
            } else {
                this.pause();
            }
        } else {
            this.play(text, onStateChange);
        }
    }
}

// Export singleton instance
if (typeof window !== 'undefined') {
    window.TTS = new SpeechOutput();
}
