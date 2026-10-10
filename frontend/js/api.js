/**
 * API Module
 * Handles all communication with the FastAPI backend.
 */

const API_BASE = window.location.origin;

class API {
    /**
     * Fetch indexed document metadata from the existing knowledge base.
     * @returns {Promise<Object>}
     */
    static async getDocuments() {
        const response = await fetch(`${API_BASE}/documents`);
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error?.message || `Failed to load documents: HTTP ${response.status}`);
        }
        return await response.json();
    }

    /**
     * Upload one document through the existing ingestion endpoint.
     * @param {File} file
     * @returns {Promise<Object>}
     */
    static async uploadDocument(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(
                errData.error?.message ||
                errData.detail ||
                `Upload failed: HTTP ${response.status}`
            );
        }

        return await response.json();
    }

    /**
     * Send a query to the chat endpoint
     * @param {string} query The user message
     * @param {string|null} sessionId The current session ID
     * @param {string|null} clarificationId If resolving a clarification
     * @returns {Promise<Object>} The response JSON
     */
    static async chat(query, sessionId = null, clarificationId = null) {
        const payload = {
            query: query,
            top_k: 3
        };

        if (sessionId) {
            payload.session_id = sessionId;
        }

        if (clarificationId) {
            payload.clarification_id = clarificationId;
        }

        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.error?.message || `HTTP error! status: ${response.status}`);
        }

        return await response.json();
    }

    /**
     * Fetch conversation history for a session
     * @param {string} sessionId
     * @returns {Promise<Object>}
     */
    static async getHistory(sessionId) {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}/history`);
        if (!response.ok) {
            if (response.status === 404) return { history: [] };
            throw new Error(`Failed to fetch history: ${response.status}`);
        }
        return await response.json();
    }

    /**
     * Delete a session
     * @param {string} sessionId
     * @returns {Promise<Object>}
     */
    static async deleteSession(sessionId) {
        const response = await fetch(`${API_BASE}/sessions/${sessionId}`, {
            method: 'DELETE'
        });
        if (!response.ok && response.status !== 404) {
            throw new Error(`Failed to delete session: ${response.status}`);
        }
        return await response.json();
    }
}

window.API = API;
