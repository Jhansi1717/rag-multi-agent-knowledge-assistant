/**
 * Voice Input Module
 * Wrapper around the Web Speech API.
 */

class VoiceInput {
    constructor(onStateChange, onResult, onError) {
        this.onStateChange = onStateChange; // (state) => {}
        this.onResult = onResult;           // (finalText, interimText) => {}
        this.onError = onError;             // (message) => {}

        this.recognition = null;
        this.isListening = false;
        this.restartPending = false;
        this.startTimeout = null;
        this.state = 'idle'; // idle, listening, processing, error, unsupported

        this._init();
    }

    _init() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            this._setState('unsupported');
            return;
        }

        this.recognition = new SpeechRecognition();
        this.recognition.continuous = true;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';

        this.recognition.onstart = () => {
            this._clearStartTimeout();
            this.isListening = true;
            this._setState('listening');
        };

        this.recognition.onresult = (event) => {
            let interimTranscript = '';
            let finalTranscript = '';

            for (let i = 0; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }

            if (this.onResult && (finalTranscript || interimTranscript)) {
                this.onResult(finalTranscript, interimTranscript);
            }
        };

        this.recognition.onerror = (event) => {
            this._clearStartTimeout();
            this.isListening = false;
            const errorMessages = {
                'not-allowed': 'Microphone permission denied. Please allow microphone access.',
                'service-not-allowed': 'Speech recognition is blocked by browser permissions.',
                'audio-capture': 'No microphone is available. Check your microphone connection.',
                'no-speech': 'No speech detected. Try again and speak after starting the microphone.',
                'network': 'Network error occurred during speech recognition.',
                'aborted': 'Speech recognition was interrupted.'
            };

            if (event.error === 'aborted') {
                if (!this.restartPending) this._setState('idle');
                return;
            }

            this._setState(event.error === 'no-speech' ? 'idle' : 'error');
            if (event.error !== 'no-speech') this.restartPending = false;
            this._reportError(errorMessages[event.error] || `Speech recognition error: ${event.error || 'unknown error'}`);
        };

        this.recognition.onend = () => {
            this._clearStartTimeout();
            this.isListening = false;
            if (this.restartPending) {
                this.restartPending = false;
                this._setState('stopped');
                setTimeout(() => this.startRecognition(), 0);
            } else if (this.state === 'listening') {
                // If it ended automatically (e.g., pause in speech in some browsers even with continuous=true)
                this._setState('idle');
            } else if (this.state === 'processing') {
                this._setState('idle');
            } else if (this.state === 'starting') {
                this._setState('error');
                this._reportError('Speech recognition ended before it started. Check microphone permissions.');
            }
        };
    }

    _setState(newState) {
        this.state = newState;
        if (this.onStateChange) {
            this.onStateChange(this.state);
        }
    }

    startRecognition() {
        if (!this.recognition) {
            this._reportError('Speech recognition is not supported in this browser.');
            return;
        }

        if (this.isListening || this.state === 'processing' || this.state === 'starting') return;

        this._setState('starting');
        this.startTimeout = window.setTimeout(() => this._handleStartTimeout(), 10000);
        try {
            this.recognition.start();
        } catch (e) {
            this._clearStartTimeout();
            this.isListening = false;
            this.restartPending = false;
            this._setState('error');
            this._reportError(e.name === 'NotAllowedError'
                ? 'Microphone permission denied. Please allow microphone access.'
                : 'Unable to start speech recognition. Please retry.');
        }
    }

    stopRecognition() {
        if (!this.recognition || !this.isListening) return;

        try {
            this._setState('processing');
            this.recognition.stop();
        } catch (e) {
            this.isListening = false;
            this._setState('error');
            this._reportError('Unable to stop speech recognition cleanly.');
        }
    }

    restartRecognition() {
        if (!this.recognition) {
            this._reportError('Speech recognition is not supported in this browser.');
            return;
        }

        if (this.restartPending) return;

        this.restartPending = true;
        if (this.isListening) {
            this.stopRecognition();
        } else if (this.state !== 'processing') {
            this.restartPending = false;
            this.startRecognition();
        }
    }

    _reportError(message) {
        if (this.onError) this.onError(message);
    }

    _clearStartTimeout() {
        if (this.startTimeout !== null) {
            window.clearTimeout(this.startTimeout);
            this.startTimeout = null;
        }
    }

    _handleStartTimeout() {
        this._clearStartTimeout();
        if (this.state !== 'starting') return;
        this.isListening = false;
        this._setState('error');
        this._reportError('Speech recognition did not start. Check microphone permissions and network connectivity.');
    }
}

window.VoiceInput = VoiceInput;
