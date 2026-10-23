import React, { useState, useEffect, useRef } from "react";

const API_BASE = window.location.port === "3000" ? "http://127.0.0.1:5000" : "";

const SUGGESTED_PROMPTS = [
  "How to calm down right now?",
  "Guide me through 4-7-8 breathing",
  "Explain my stress readings",
  "How can I sleep better tonight?"
];

export default function StressChatbot({ stressLevel, stressPercentage }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [hasGeminiKey, setHasGeminiKey] = useState(false);
  const [geminiModel, setGeminiModel] = useState("gemini-2.5-flash");
  const [messages, setMessages] = useState([
    {
      role: "bot",
      text: "Hello! I am your PRISM AI Stress Copilot. I analyze your biometric telemetry and provide clinically-grounded calm routines, breathing exercises, and focus strategies. How are you feeling right now?",
    },
  ]);

  const messagesEndRef = useRef(null);

  // Check backend Gemini status on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/chat/status`)
      .then((r) => r.json())
      .then((data) => {
        if (data.status === "success") {
          setHasGeminiKey(Boolean(data.has_gemini_key));
          if (data.model) setGeminiModel(data.model);
        }
      })
      .catch(() => {});
  }, []);

  // Auto scroll to bottom of messages
  useEffect(() => {
    if (open) {
      messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, open, loading]);

  const sendMessage = async (customText = null) => {
    const text = (customText || input).trim();
    if (!text || loading) return;

    const nextMessages = [...messages, { role: "user", text }];
    setMessages(nextMessages);
    if (!customText) setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/api/chat/stress`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          stress_level: stressLevel || "Moderate",
          stress_percentage:
            stressPercentage !== null && stressPercentage !== undefined
              ? Number(stressPercentage)
              : null,
        }),
      });

      const data = await response.json();
      if (data.status === "success" && data.reply) {
        if (data.provider === "gemini") {
          setHasGeminiKey(true);
        }
        setMessages((prev) => [...prev, { role: "bot", text: data.reply }]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            role: "bot",
            text: data.message || "I could not process that right now. Please try again.",
          },
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "bot",
          text: `Connection error (${err.message}). Ensure the PRISM backend server is running.`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stress-chatbot-shell">
      {/* Floating Trigger Button on Right Bottom */}
      {!open && (
        <button
          className="stress-chatbot-open-btn"
          onClick={() => setOpen(true)}
          title="Open PRISM AI Copilot"
        >
          <span className="material-symbols-outlined fs-5">psychology</span>
          <span>Stress AI Copilot</span>
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: hasGeminiKey ? "#00e3fd" : "#ffb74d",
              display: "inline-block",
              boxShadow: `0 0 8px ${hasGeminiKey ? "#00e3fd" : "#ffb74d"}`,
            }}
          />
        </button>
      )}

      {/* Floating Right-Docked Card Panel */}
      {open && (
        <div className="stress-chatbot-panel">
          {/* Header */}
          <div className="stress-chatbot-header">
            <div className="stress-chatbot-title">
              <div className="stress-chatbot-avatar">
                <span className="material-symbols-outlined fs-5">smart_toy</span>
              </div>
              <div>
                <strong style={{ display: "block", fontSize: "0.95rem" }}>
                  PRISM AI Copilot
                </strong>
                <span
                  style={{
                    fontSize: "0.72rem",
                    color: "var(--text-muted)",
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: hasGeminiKey ? "#4caf50" : "#ff9800",
                    }}
                  />
                  <span>
                    {hasGeminiKey
                      ? `${geminiModel} Active`
                      : "Local Rule-Based Mode"}
                  </span>
                </span>
              </div>
            </div>

            <button
              className="stress-chatbot-close-btn"
              onClick={() => setOpen(false)}
              title="Close Copilot"
            >
              <span className="material-symbols-outlined fs-5">close</span>
            </button>
          </div>

          {/* Gemini API Key Status Notice */}
          <div className="stress-chatbot-status-bar">
            <span>
              {hasGeminiKey ? (
                <span style={{ color: "var(--primary-color)", fontWeight: 600 }}>
                  ✨ Gemini LLM Reasoning Enabled
                </span>
              ) : (
                <span style={{ color: "#f57c00" }}>
                  ⚡ Offline Mode • Set GEMINI_API_KEY in .env
                </span>
              )}
            </span>
            {stressLevel && (
              <span
                style={{
                  background: "var(--card-bg)",
                  padding: "2px 8px",
                  borderRadius: 6,
                  fontWeight: 600,
                  fontSize: "0.7rem",
                }}
              >
                Telemetry: {stressLevel}
              </span>
            )}
          </div>

          {/* Quick Suggestion Chips */}
          <div className="stress-chatbot-chips">
            {SUGGESTED_PROMPTS.map((prompt, idx) => (
              <button
                key={idx}
                className="stress-chatbot-chip"
                onClick={() => sendMessage(prompt)}
                disabled={loading}
              >
                {prompt}
              </button>
            ))}
          </div>

          {/* Messages Feed */}
          <div className="stress-chatbot-messages">
            {messages.map((msg, idx) => (
              <div
                key={`${msg.role}-${idx}`}
                className={msg.role === "user" ? "chat-msg-user" : "chat-msg-bot"}
              >
                {msg.text}
              </div>
            ))}
            {loading && (
              <div className="chat-msg-bot">
                <div className="chat-typing-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Box */}
          <div className="stress-chatbot-input">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask PRISM about stress, breathing, sleep..."
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
              disabled={loading}
            />
            <button
              className="stress-chatbot-send-btn"
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              title="Send message"
            >
              <span className="material-symbols-outlined fs-5">send</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
