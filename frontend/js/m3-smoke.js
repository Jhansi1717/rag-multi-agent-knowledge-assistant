(() => {
    const results = document.getElementById('results');
    const wait = (ms = 10) => new Promise(resolve => setTimeout(resolve, ms));

    function assert(condition, message) {
        if (!condition) throw new Error(message);
    }

    function record(name, error = null) {
        const item = document.createElement('li');
        item.className = error ? 'fail' : 'pass';
        item.textContent = `${error ? 'FAIL' : 'PASS'}: ${name}${error ? ` - ${error.message}` : ''}`;
        results.appendChild(item);
    }

    async function run(name, test) {
        try {
            await test();
            record(name);
        } catch (error) {
            record(name, error);
        }
    }

    function buildResult() {
        const citation = {
            chunk_id: 'chunk-7',
            filename: 'guide.txt',
            document_id: 'guide-doc',
            excerpt: 'Documented fact.'
        };
        return {
            status: 'answered',
            confidence: 0.42,
            confidence_level: 'LOW',
            no_information_found: false,
            citations: [citation],
            retrieval: { sufficient_evidence: true, no_relevant_information: false },
            m3: {
                transparency: {
                    retrieved_count: 1,
                    grounded: false,
                    citations: [citation],
                    source_evidence: [{
                        document_id: 'guide-doc',
                        document_name: 'guide.txt',
                        chunk_id: 'chunk-7',
                        chunk_text: 'Documented fact.',
                        similarity_score: 0.42,
                        page: 2,
                        section: 'Overview'
                    }]
                }
            }
        };
    }

    async function runAll() {
        results.replaceChildren();
        const { recognitionInstances, speechCalls, utterances } = window.__m3Smoke;
        let inputText = '';
        const states = [];
        const voiceErrors = [];
        const transcriptInput = document.createElement('textarea');
        const voice = new window.VoiceInput(
            state => states.push(state),
            (finalText, interimText) => {
                inputText = finalText + interimText;
                transcriptInput.value = inputText;
            },
            message => voiceErrors.push(message)
        );

        await run('1. Microphone enters listening state', () => {
            voice.startRecognition();
            assert(voice.state === 'listening', `state=${voice.state}`);
        });
        await run('2. Transcript appears', () => {
            recognitionInstances.at(-1).emitTranscript('hello ');
            recognitionInstances.at(-1).emitResult([
                { isFinal: true, 0: { transcript: 'hello ' } },
                { isFinal: true, 0: { transcript: 'world' } }
            ], 1);
            assert(inputText === 'hello world', `transcript=${inputText}`);
        });
        await run('3. Transcript remains editable', () => {
            transcriptInput.value = 'edited transcript';
            assert(transcriptInput.value === 'edited transcript', 'editable transcript changed unexpectedly');
        });
        await run('4. Stopping recognition returns to idle', async () => {
            voice.stopRecognition();
            await wait();
            assert(voice.state === 'idle', `state=${voice.state}`);
        });
        await run('5. Restart starts a new recognition session', async () => {
            const recognition = recognitionInstances.at(-1);
            const startsBefore = recognition.startCount;
            voice.restartRecognition();
            await wait();
            assert(recognition.startCount > startsBefore, 'restart did not start');
        });
        await run('6. Unsupported SpeechRecognition does not crash', () => {
            const saved = [window.SpeechRecognition, window.webkitSpeechRecognition];
            delete window.SpeechRecognition;
            delete window.webkitSpeechRecognition;
            const unsupported = new window.VoiceInput(() => {}, () => {}, () => {});
            window.SpeechRecognition = saved[0];
            window.webkitSpeechRecognition = saved[1];
            assert(unsupported.state === 'unsupported', `state=${unsupported.state}`);
        });

        await run('7. Permission denial is visible and stops listening', () => {
            const denied = new window.VoiceInput(() => {}, () => {}, message => voiceErrors.push(message));
            const recognition = recognitionInstances.at(-1);
            denied.startRecognition();
            recognition.emitError('not-allowed');
            assert(denied.state === 'error', `state=${denied.state}`);
            assert(voiceErrors.at(-1).includes('permission denied'), `message=${voiceErrors.at(-1)}`);
        });
        await run('8. Network failure is not reported as success', () => {
            const network = new window.VoiceInput(() => {}, () => {}, message => voiceErrors.push(message));
            const recognition = recognitionInstances.at(-1);
            network.startRecognition();
            recognition.emitError('network');
            assert(network.state === 'error', `state=${network.state}`);
            assert(voiceErrors.at(-1).includes('Network error'), `message=${voiceErrors.at(-1)}`);
        });
        await run('9. No-speech returns to idle with guidance', () => {
            const silent = new window.VoiceInput(() => {}, () => {}, message => voiceErrors.push(message));
            const recognition = recognitionInstances.at(-1);
            silent.startRecognition();
            recognition.emitError('no-speech');
            assert(silent.state === 'idle', `state=${silent.state}`);
            assert(voiceErrors.at(-1).includes('No speech detected'), `message=${voiceErrors.at(-1)}`);
        });
        await run('10. Missing start callback becomes a visible failure', () => {
            const delayed = new window.VoiceInput(() => {}, () => {}, message => voiceErrors.push(message));
            const recognition = recognitionInstances.at(-1);
            recognition.start = () => { recognition.startCount += 1; };
            delayed.startRecognition();
            assert(delayed.state === 'starting', `state=${delayed.state}`);
            delayed._handleStartTimeout();
            assert(delayed.state === 'error', `state=${delayed.state}`);
            assert(voiceErrors.at(-1).includes('did not start'), `message=${voiceErrors.at(-1)}`);
        });

        await run('11. Queued speech can pause before onstart', () => {
            const synthesis = window.TTS.synth;
            const originalSpeak = synthesis.speak;
            try {
                synthesis.speak = utterance => {
                    speechCalls.push('speak-pending');
                    utterances.push(utterance);
                    synthesis.speaking = true;
                };
                window.TTS.play('Pending speech.', () => {});
                assert(window.TTS.isPlaying, 'queued speech was not marked active');
                window.TTS.pause();
                assert(speechCalls.at(-1) === 'pause', 'queued speech did not pause');
                window.TTS.resume();
                assert(speechCalls.at(-1) === 'resume', 'queued speech did not resume');
            } finally {
                window.TTS.stop();
                synthesis.speak = originalSpeak;
            }
        });

        const chat = new window.ChatUI();
        await run('12. Speaker button starts speech', () => {
            chat.addAssistantMessage('First response.');
            const button = document.querySelector('.tts-btn');
            button.click();
            assert(speechCalls.includes('speak'), 'speech did not start');
        });
        await run('13. Same response pauses speech', () => {
            document.querySelector('.tts-btn').click();
            assert(speechCalls.at(-1) === 'pause', 'speech did not pause');
            assert(document.querySelector('.tts-btn').getAttribute('aria-label') === 'Resume reading aloud', 'pause state is not announced');
        });
        await run('14. Paused response resumes speech', () => {
            document.querySelector('.tts-btn').click();
            assert(speechCalls.at(-1) === 'resume', 'speech did not resume');
            assert(document.querySelector('.tts-btn').getAttribute('aria-label') === 'Pause reading aloud', 'playing state is not announced');
        });
        await run('15. Stop cancels speech', () => {
            document.querySelector('.tts-stop-btn').click();
            assert(speechCalls.at(-1) === 'cancel', 'speech was not canceled');
        });
        await run('16. New response cancels previous speech and ignores stale callbacks', () => {
            chat.addAssistantMessage('Second response.');
            const buttons = document.querySelectorAll('.tts-btn');
            buttons[0].click();
            const oldUtterance = utterances.at(-1);
            buttons[1].click();
            assert(speechCalls.includes('cancel'), 'previous speech was not canceled');
            oldUtterance.onend?.();
            assert(window.TTS.isPlaying, 'a stale completion changed the new playback state');
        });
        await run('17. Selected language and voice reach the utterance', () => {
            window.TTS.stop();
            assert(window.TTS.setLanguage('fr-FR'), 'language selection failed');
            assert(window.TTS.setVoice('fr-test'), 'voice selection failed');
            window.TTS.play('Bonjour.', () => {});
            const utterance = utterances.at(-1);
            assert(utterance.lang === 'fr-FR', `language=${utterance.lang}`);
            assert(utterance.voice?.voiceURI === 'fr-test', `voice=${utterance.voice?.voiceURI}`);
            window.TTS.stop();
            window.TTS.setLanguage('en-US');
            window.TTS.setVoice('');
        });
        await run('18. Empty voice list uses the browser default', () => {
            const originalVoices = window.TTS.voices;
            window.TTS.voices = [];
            window.TTS.play('Default voice fallback.', () => {});
            const utterance = utterances.at(-1);
            assert(utterance.lang === 'en-US', `language=${utterance.lang}`);
            assert(!utterance.voice, 'a missing voice list selected a nonexistent voice');
            window.TTS.stop();
            window.TTS.voices = originalVoices;
        });
        await run('19. Unsupported speechSynthesis does not crash', () => {
            const Constructor = window.TTS.constructor;
            const savedSynth = window.speechSynthesis;
            const savedUtterance = window.SpeechSynthesisUtterance;
            Constructor.instance = null;
            delete window.speechSynthesis;
            delete window.SpeechSynthesisUtterance;
            const unsupported = new Constructor();
            window.speechSynthesis = savedSynth;
            window.SpeechSynthesisUtterance = savedUtterance;
            Constructor.instance = window.TTS;
            assert(unsupported.isSupported === false, 'unsupported TTS was marked supported');
        });

        await run('20. Source panel expands', () => {
            chat.addAssistantMessage('Evidence response.', false, false, buildResult());
            const details = document.querySelector('.evidence-accordion');
            details.open = true;
            assert(details.open, 'evidence panel did not expand');
        });
        await run('21. Citation maps to correct source', () => {
            const source = document.querySelector('.source-docname');
            const chunk = document.querySelector('.source-location');
            const citation = document.querySelector('.source-citation');
            assert(source.textContent === 'guide.txt', `source=${source.textContent}`);
            assert(chunk.textContent.includes('chunk-7'), `chunk=${chunk.textContent}`);
            assert(citation?.textContent.includes('Documented fact.'), 'answer citation is not linked to retrieved evidence');
        });
        await run('22. Relevance score is displayed', () => {
            assert(document.querySelector('.source-relevance').textContent.includes('0.42'), 'score missing');
        });
        await run('23. Low-confidence state renders correctly', () => {
            assert(document.querySelector('.conf-low'), 'low confidence indicator missing');
        });

        await run('24. Retrieved evidence stays visible with no-answer fallback', () => {
            chat.addAssistantMessage(
                'No grounded answer was generated.',
                false,
                false,
                { ...buildResult(), no_information_found: true }
            );
            const panels = document.querySelectorAll('.evidence-accordion');
            const latestPanel = panels[panels.length - 1];
            assert(latestPanel, 'retrieved evidence panel is missing');
            const transparencyPanel = latestPanel.closest('.transparency-panel');
            assert(
                latestPanel.querySelector('.source-docname')?.textContent === 'guide.txt',
                'retrieved source is not displayed'
            );
            assert(
                transparencyPanel?.querySelector('.transparency-no-result')?.textContent.includes('Retrieved evidence'),
                'no-answer state does not explain why evidence is shown'
            );
            assert(transparencyPanel?.querySelector('.transparency-no-result')?.getAttribute('role') === 'status', 'no-result state is not announced');
        });

        await run('25. Answer and retrieval status are displayed', () => {
            const panel = window.TransparencyPanel.build(buildResult());
            assert(panel.textContent.includes('Status: Answered'), 'answer status is missing');
            assert(panel.textContent.includes('Retrieval: Evidence found'), 'retrieval status is missing');
        });
        await run('26. Missing confidence is not labeled low', () => {
            const panel = window.TransparencyPanel.build({
                status: 'error',
                m3: { transparency: { retrieved_count: 0, grounded: false } }
            });
            assert(panel.querySelector('.conf-unavailable'), 'missing confidence is not identified');
            assert(!panel.querySelector('.conf-low'), 'missing confidence was fabricated as low');
        });
        await run('27. Unmapped citations are omitted with a warning', () => {
            const panel = window.TransparencyPanel.build({
                status: 'answered',
                confidence: 0.8,
                confidence_level: 'HIGH',
                m3: { transparency: {
                    retrieved_count: 1,
                    grounded: true,
                    citations: [{ chunk_id: 'chunk-7', filename: 'guide.txt', document_id: 'wrong-doc', excerpt: 'Documented fact.' }],
                    source_evidence: [{ document_id: 'guide-doc', document_name: 'guide.txt', chunk_id: 'chunk-7', chunk_text: 'Documented fact.', similarity_score: 0.8 }]
                } }
            });
            assert(!panel.querySelector('.source-citation'), 'unmapped citation was shown as evidence');
            assert(panel.querySelector('.transparency-citation-warning')?.getAttribute('role') === 'alert', 'citation mismatch warning is missing');
        });
        await run('28. Citation excerpt must exist in the matched chunk', () => {
            const panel = window.TransparencyPanel.build({
                status: 'answered',
                confidence: 0.8,
                confidence_level: 'HIGH',
                m3: { transparency: {
                    retrieved_count: 1,
                    grounded: true,
                    citations: [{ chunk_id: 'chunk-7', filename: 'guide.txt', document_id: 'guide-doc', excerpt: 'Fabricated passage.' }],
                    source_evidence: [{ document_id: 'guide-doc', document_name: 'guide.txt', chunk_id: 'chunk-7', chunk_text: 'Documented fact.', similarity_score: 0.8 }]
                } }
            });
            assert(!panel.querySelector('.source-citation'), 'citation excerpt was not present in the matched chunk');
            assert(panel.querySelector('.transparency-citation-warning'), 'excerpt mismatch was not reported');
        });
        await run('29. Comparative evidence maps citations for both sides', () => {
            const panel = window.TransparencyPanel.build({
                status: 'answered',
                confidence: 0.8,
                confidence_level: 'HIGH',
                retrieval: { sufficient_evidence: true, no_relevant_information: false },
                m3: { transparency: {
                    retrieved_count: 2,
                    grounded: true,
                    citations: [
                        { chunk_id: 'python-1', filename: 'languages.csv', document_id: 'languages-doc', excerpt: 'Python is dynamically typed.' },
                        { chunk_id: 'java-1', filename: 'languages.csv', document_id: 'languages-doc', excerpt: 'Java is statically typed.' }
                    ],
                    source_evidence: [
                        { chunk_id: 'python-1', document_id: 'languages-doc', document_name: 'languages.csv', chunk_text: 'Python is dynamically typed.', similarity_score: 0.8 },
                        { chunk_id: 'java-1', document_id: 'languages-doc', document_name: 'languages.csv', chunk_text: 'Java is statically typed.', similarity_score: 0.79 }
                    ]
                } }
            });
            const citations = panel.querySelectorAll('.source-citation');
            assert(citations.length === 2, `mapped citations=${citations.length}`);
            assert(citations[0].textContent.includes('Python is dynamically typed.'), 'first comparison side is missing');
            assert(citations[1].textContent.includes('Java is statically typed.'), 'second comparison side is missing');
        });
        await run('30. Evidence disclosure uses a keyboard-native summary', () => {
            const panel = window.TransparencyPanel.build(buildResult());
            const summary = panel.querySelector('.evidence-summary');
            assert(summary?.tagName === 'SUMMARY', 'evidence disclosure is not a native summary control');
            assert(summary.textContent.includes('Sources & Evidence'), 'evidence control has no readable label');
        });
        await run('31. Error state does not claim retrieval found no evidence', () => {
            const panel = window.TransparencyPanel.build({
                status: 'error',
                no_information_found: true,
                m3: { transparency: { retrieved_count: 0, grounded: false } }
            });
            assert(panel.textContent.includes('Retrieval: Not completed'), 'failed retrieval status is missing');
            assert(panel.textContent.includes('request failed'), 'error state does not explain retrieval was incomplete');
            assert(!panel.textContent.includes('Sufficient evidence was not found'), 'error state is mislabeled as no-result');
        });
    }

    document.getElementById('run-tests').addEventListener('click', runAll);
})();
