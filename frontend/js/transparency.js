/**
 * Transparency Panel Module
 * Renders confidence, source count, and expandable evidence from backend values only.
 * Never fabricates confidence or source metadata.
 */

const EXCERPT_LIMIT = 240; // characters before "show more"

class TransparencyPanel {
    /**
     * Build and return the transparency DOM node for an assistant response.
     * @param {Object} result - The raw /chat API response
     * @returns {HTMLElement|null}
     */
    static build(result) {
        const t = result?.m3?.transparency || {};
        const confidenceLevel = result?.confidence_level || null;
        const confidenceScore = Number.isFinite(result?.confidence) ? result.confidence : null;
        const evidence = Array.isArray(t?.source_evidence) ? t.source_evidence : [];
        const citations = Array.isArray(t?.citations)
            ? t.citations
            : Array.isArray(result?.citations)
                ? result.citations
                : [];

        // Don't render panel for clarification-needed responses
        if (result?.status === 'clarification_needed') return null;

        const panel = document.createElement('div');
        panel.className = 'transparency-panel';

        // ── Metadata bar ────────────────────────────────────────────────
        const metaBar = document.createElement('div');
        metaBar.className = 'transparency-meta-bar';
        metaBar.innerHTML = `
            ${TransparencyPanel._confidenceBadge(confidenceLevel, confidenceScore)}
            ${TransparencyPanel._sourcesChip(t)}
            ${TransparencyPanel._groundedChip(t, result)}
            ${TransparencyPanel._statusChip(result?.status)}
            ${TransparencyPanel._retrievalStatusChip(result)}
        `;
        panel.appendChild(metaBar);

        // ── No-result explanation ────────────────────────────────────────
        const noEvidence = result?.no_information_found === true
            || result?.status === 'unavailable'
            || result?.retrieval?.no_relevant_information === true
            || t?.retrieved_count === 0;
        if (noEvidence) {
            const notice = document.createElement('p');
            notice.className = 'transparency-no-result';
            notice.setAttribute('role', 'status');
            notice.setAttribute('aria-live', 'polite');
            notice.textContent = result?.status === 'error'
                ? 'Evidence could not be retrieved because the request failed.'
                : evidence.length > 0
                ? 'A grounded answer was not generated. Retrieved evidence is shown below.'
                : 'Sufficient evidence was not found in the knowledge base for this query.';
            panel.appendChild(notice);
            if (evidence.length === 0) return panel;
        }

        // ── Evidence accordion ───────────────────────────────────────────
        if (evidence.length > 0) {
            const mappedCitations = citations.filter(citation =>
                evidence.some(item => TransparencyPanel._citationMatchesEvidence(citation, item))
            );
            panel.appendChild(TransparencyPanel._buildAccordion(evidence, mappedCitations));
            if (mappedCitations.length < citations.length) {
                const warning = document.createElement('p');
                warning.className = 'transparency-citation-warning';
                warning.setAttribute('role', 'alert');
                warning.textContent = 'Some citations could not be matched to retrieved evidence and were omitted.';
                panel.appendChild(warning);
            }
        }

        return panel;
    }

    // ── Confidence badge ─────────────────────────────────────────────────
    static _confidenceBadge(level, score) {
        const map = {
            HIGH: { label: 'High', cls: 'conf-high', icon: '🟢' },
            MEDIUM: { label: 'Medium', cls: 'conf-medium', icon: '🟡' },
            LOW: { label: 'Low', cls: 'conf-low', icon: '🔴' },
        };
        const selected = map[level];
        const label = selected?.label || 'Not reported';
        const cls = selected?.cls || 'conf-unavailable';
        const icon = selected?.icon || '';
        const pct = score !== null ? ` · ${Math.round(score * 100)}%` : '';
        return `<span class="trans-chip ${cls}" aria-label="Confidence level ${label}${pct}" title="Confidence: ${label}">
            ${icon ? `<span aria-hidden="true">${icon}</span>` : ''} Confidence: ${label}${pct}
        </span>`;
    }

    // ── Sources chip ─────────────────────────────────────────────────────
    static _sourcesChip(t) {
        if (!Number.isInteger(t?.retrieved_count) || t.retrieved_count < 0) {
            return '<span class="trans-chip" aria-label="Source count not reported">Sources: not reported</span>';
        }
        const count = t.retrieved_count;
        if (count === 0) return `<span class="trans-chip sources-zero" aria-label="Zero sources retrieved">Sources: 0</span>`;
        return `<span class="trans-chip" aria-label="${count} sources retrieved">Sources: ${count}</span>`;
    }

    // ── Grounded chip ────────────────────────────────────────────────────
    static _groundedChip(t, result) {
        if (typeof t?.grounded !== 'boolean') {
            return '<span class="trans-chip" aria-label="Grounding status not reported">Grounding: not reported</span>';
        }
        const grounded = t.grounded;
        const cls = grounded ? 'grounded-yes' : 'grounded-no';
        const icon = grounded ? '✓' : '⚠';
        const label = grounded ? 'Grounded' : 'Ungrounded';
        return `<span class="trans-chip ${cls}" aria-label="${label}">
            <span aria-hidden="true">${icon}</span> ${label}
        </span>`;
    }

    static _statusChip(status) {
        if (typeof status !== 'string' || !status) return '';
        const labels = {
            answered: 'Answered',
            unavailable: 'Unavailable',
            error: 'Error',
            clarification_needed: 'Clarification needed'
        };
        const label = labels[status] || status;
        return `<span class="trans-chip response-status" aria-label="Answer status ${TransparencyPanel._esc(label)}">Status: ${TransparencyPanel._esc(label)}</span>`;
    }

    static _retrievalStatusChip(result) {
        const retrieval = result?.retrieval;
        let status = 'Not reported';
        if (retrieval?.sufficient_evidence === true) {
            status = 'Evidence found';
        } else if (retrieval?.no_relevant_information === true) {
            status = 'No relevant evidence';
        } else if (retrieval) {
            status = 'Insufficient evidence';
        } else if (result?.status === 'error') {
            status = 'Not completed';
        } else if (result?.status === 'unavailable' || result?.no_information_found === true) {
            status = 'No supporting evidence';
        }
        return `<span class="trans-chip retrieval-status" aria-label="Retrieval status ${status}">Retrieval: ${status}</span>`;
    }

    // ── Evidence accordion ────────────────────────────────────────────────
    static _buildAccordion(evidence, citations = []) {
        const wrapper = document.createElement('details');
        wrapper.className = 'evidence-accordion';

        const summary = document.createElement('summary');
        summary.className = 'evidence-summary';
        summary.innerHTML = `<span class="evidence-toggle-icon" aria-hidden="true">▶</span> Sources &amp; Evidence`;
        wrapper.appendChild(summary);

        wrapper.addEventListener('toggle', () => {
            summary.querySelector('.evidence-toggle-icon').textContent = wrapper.open ? '▼' : '▶';
        });

        const body = document.createElement('div');
        body.className = 'evidence-body';

        evidence.forEach((item, idx) => {
            const itemCitations = citations.filter(citation =>
                TransparencyPanel._citationMatchesEvidence(citation, item)
            );
            body.appendChild(TransparencyPanel._buildSourceCard(item, idx + 1, itemCitations));
        });

        wrapper.appendChild(body);
        return wrapper;
    }

    // ── Individual source card ────────────────────────────────────────────
    static _buildSourceCard(item, num, citations = []) {
        const card = document.createElement('div');
        card.className = 'source-card';

        const hasScore = Number.isFinite(item.similarity_score);
        const score = hasScore
            ? item.similarity_score.toFixed(2)
            : 'Not reported';
        const relevanceCls = !hasScore ? 'rel-unavailable'
            : item.similarity_score >= 0.75 ? 'rel-high'
            : item.similarity_score >= 0.50 ? 'rel-medium'
            : 'rel-low';

        const locationParts = [];
        const pageStart = item.page_start ?? item.page;
        const pageEnd = item.page_end ?? item.page;
        if (pageStart != null) {
            const pageLabel = pageEnd != null && pageEnd !== pageStart
                ? `Pages ${pageStart}–${pageEnd}`
                : `Page ${pageStart}`;
            locationParts.push(TransparencyPanel._esc(String(pageLabel)));
        }
        if (item.section) locationParts.push(`Section: ${TransparencyPanel._esc(String(item.section))}`);
        if (item.row_number != null) locationParts.push(`Row ${TransparencyPanel._esc(String(item.row_number))}`);
        else if (item.row_start != null) {
            const rowLabel = item.row_end != null && item.row_end !== item.row_start
                ? `Rows ${item.row_start}–${item.row_end}`
                : `Row ${item.row_start}`;
            locationParts.push(TransparencyPanel._esc(String(rowLabel)));
        }
        const location = locationParts.join(' · ') || '';

        const chunkMeta = item.chunk_id
            ? `Chunk ${TransparencyPanel._esc(String(item.chunk_id))}`
            : '';

        const text = item.chunk_text || '';
        const needsTruncation = text.length > EXCERPT_LIMIT;
        const shortText = needsTruncation ? text.slice(0, EXCERPT_LIMIT) + '…' : text;
        const citationMarkup = citations.map(citation => `
            <div class="source-citation">
                <span class="source-citation-label">Cited in answer</span>
                ${citation.excerpt ? `<blockquote>${TransparencyPanel._esc(String(citation.excerpt))}</blockquote>` : ''}
            </div>
        `).join('');

        card.innerHTML = `
            <div class="source-card-header">
                <span class="source-num">[${num}]</span>
                <span class="source-docname">${TransparencyPanel._esc(item.document_name || 'Document name not reported')}</span>
                <span class="source-relevance ${relevanceCls}">Relevance: ${score}</span>
            </div>
            <div class="source-location">${[location, chunkMeta].filter(Boolean).join(' · ')}</div>
            <blockquote class="source-excerpt" data-full="${TransparencyPanel._attr(text)}" data-short="${TransparencyPanel._attr(shortText)}" data-expanded="false" tabindex="0">
                ${TransparencyPanel._esc(shortText)}
            </blockquote>
            ${needsTruncation ? `<button class="excerpt-toggle" aria-expanded="false" aria-label="Expand excerpt">Show more</button>` : ''}
            ${citationMarkup}
        `;

        if (needsTruncation) {
            const btn = card.querySelector('.excerpt-toggle');
            const bq = card.querySelector('.source-excerpt');
            btn.addEventListener('click', () => {
                const expanded = bq.dataset.expanded === 'true';
                bq.innerHTML = expanded
                    ? TransparencyPanel._esc(bq.dataset.short)
                    : TransparencyPanel._esc(bq.dataset.full);
                bq.dataset.expanded = !expanded;
                btn.textContent = expanded ? 'Show more' : 'Show less';
                btn.setAttribute('aria-expanded', !expanded);
            });
        }

        return card;
    }

    static _citationMatchesEvidence(citation, item) {
        if (!citation?.chunk_id || !citation.filename || citation.chunk_id !== item?.chunk_id) return false;
        if (citation.filename !== (item.document_name || item.filename)) return false;
        if (citation.document_id && citation.document_id !== item.document_id) return false;
        return !citation.excerpt || String(item.chunk_text || '').includes(citation.excerpt);
    }

    static _esc(str) {
        if (!str) return '';
        return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    static _attr(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }
}

window.TransparencyPanel = TransparencyPanel;
