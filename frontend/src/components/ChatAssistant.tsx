"use client";

import React, { useState, useEffect, useRef } from "react";
import { Bot, Send, UserPlus, Mic, MicOff, Volume2, VolumeX } from "lucide-react";

interface Message {
  id: string;
  sender: "user" | "bot";
  text: string;
  intent?: string;
  language?: string;
  tts_supported?: boolean;
  facts_used?: any;
}

interface ChatAssistantProps {
  customerId: string;
  selectedLanguage: string;
}

export const ChatAssistant: React.FC<ChatAssistantProps> = ({ customerId, selectedLanguage }) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      sender: "bot",
      text: "Namaste! I am your Sahaay AI financial companion. How may I assist you with your accounts, EMI schedules, or savings today? (Click mic to speak)",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const recognitionRef = useRef<any>(null);

  // Initialize Web SpeechRecognition
  useEffect(() => {
    if (typeof window !== "undefined") {
      const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
      if (SpeechRecognition) {
        const reco = new SpeechRecognition();
        reco.continuous = false;
        reco.interimResults = false;
        reco.lang = selectedLanguage === "en" ? "en-IN" : `${selectedLanguage}-IN`;

        reco.onstart = () => setIsRecording(true);
        reco.onresult = (event: any) => {
          const transcript = event.results[0][0].transcript;
          setInput(transcript);
          sendMessage(transcript);
        };
        reco.onerror = (event: any) => {
          console.warn("ASR error:", event.error);
          setIsRecording(false);
        };
        reco.onend = () => setIsRecording(false);
        recognitionRef.current = reco;
      }
    }
  }, [selectedLanguage]);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      alert("Speech recognition is not supported in this browser. Please use text input.");
      return;
    }
    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      try {
        recognitionRef.current.lang = selectedLanguage === "en" ? "en-IN" : `${selectedLanguage}-IN`;
        recognitionRef.current.start();
      } catch (err) {
        console.error(err);
      }
    }
  };

  const speakText = (text: string, langCode?: string) => {
    if (!ttsEnabled || typeof window === "undefined" || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    const voices = window.speechSynthesis.getVoices();
    const targetLang = (langCode || selectedLanguage || "en").toLowerCase();
    const matchedVoice = voices.find((v) => v.lang.toLowerCase().startsWith(targetLang));
    if (matchedVoice) utterance.voice = matchedVoice;
    utterance.lang = matchedVoice ? matchedVoice.lang : "en-US";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  const sendMessage = async (msgText?: string) => {
    const textToSend = msgText || input.trim();
    if (!textToSend || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: textToSend,
    };
    setMessages((prev) => [...prev, userMsg]);
    if (!msgText) setInput("");
    setLoading(true);

    try {
      const resp = await fetch(`http://127.0.0.1:8000/customers/${customerId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: textToSend, language: selectedLanguage || undefined }),
      });
      if (resp.ok) {
        const data = await resp.json();
        const botMsg: Message = {
          id: (Date.now() + 1).toString(),
          sender: "bot",
          text: data.reply,
          intent: data.intent,
          language: data.language,
          tts_supported: data.tts_supported,
          facts_used: data.facts_used,
        };
        setMessages((prev) => [...prev, botMsg]);

        if (data.tts_supported) {
          speakText(data.reply, data.language);
        }
      } else {
        setMessages((prev) => [
          ...prev,
          {
            id: (Date.now() + 1).toString(),
            sender: "bot",
            text: "Backend synchronizing. Please try again shortly.",
          },
        ]);
      }
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          sender: "bot",
          text: "Connection error: Please ensure FastAPI backend is running at http://127.0.0.1:8000.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const startKYC = async () => {
    try {
      const resp = await fetch(`http://127.0.0.1:8000/customers/${customerId}/onboarding-step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: "START" }),
      });
      const data = await resp.json();
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          sender: "bot",
          text: `[KYC State Machine: ${data.next_step}] ${data.prompt}`,
          intent: "onboarding_kyc",
        },
      ]);
      speakText(data.prompt, selectedLanguage);
    } catch (e) {}
  };

  return (
    <div className="flex flex-col h-[720px] bg-gray-900/80 backdrop-blur-md rounded-2xl border border-gray-800 overflow-hidden">
      {/* Header */}
      <div className="px-5 py-4 border-b border-gray-800/80 bg-gray-950/40 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center text-white shadow-md">
              <Bot className="w-5 h-5" />
            </div>
            <span className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-emerald-500 ring-2 ring-gray-900" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-white flex items-center gap-2">
              Sahaay Voice Companion
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Gemini NLP + ASR/TTS
              </span>
            </h4>
            <p className="text-[11px] text-gray-400">Strictly grounded in backend financial facts</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setTtsEnabled(!ttsEnabled)}
            className={`p-1.5 rounded-lg border text-xs transition-all ${
              ttsEnabled
                ? "bg-indigo-900/40 border-indigo-500/40 text-indigo-300"
                : "bg-gray-800 border-gray-700 text-gray-400"
            }`}
            title={ttsEnabled ? "TTS Enabled" : "TTS Muted"}
          >
            {ttsEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
          </button>
          <button
            onClick={startKYC}
            className="px-2.5 py-1 text-xs font-semibold rounded-lg bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700 transition-all flex items-center gap-1.5"
          >
            <UserPlus className="w-3.5 h-3.5 text-cyan-400" /> KYC Flow
          </button>
        </div>
      </div>

      {/* Quick Prompt Chips */}
      <div className="px-4 py-2.5 bg-gray-950/40 border-b border-gray-800/60 flex items-center gap-2 overflow-x-auto text-xs">
        <button
          onClick={() => sendMessage("Check my next EMI due")}
          className="px-2.5 py-1 rounded-full bg-gray-800/80 hover:bg-indigo-600 hover:text-white text-gray-300 border border-gray-700/60 whitespace-nowrap transition-all"
        >
          💳 Check EMI
        </button>
        <button
          onClick={() => sendMessage("Explain my last transaction")}
          className="px-2.5 py-1 rounded-full bg-gray-800/80 hover:bg-indigo-600 hover:text-white text-gray-300 border border-gray-700/60 whitespace-nowrap transition-all"
        >
          🧾 Explain Txn
        </button>
        <button
          onClick={() => sendMessage("Why was this product recommended?")}
          className="px-2.5 py-1 rounded-full bg-gray-800/80 hover:bg-indigo-600 hover:text-white text-gray-300 border border-gray-700/60 whitespace-nowrap transition-all"
        >
          ✨ Product Info
        </button>
        <button
          onClick={() => sendMessage("Report suspicious activity on card")}
          className="px-2.5 py-1 rounded-full bg-gray-800/80 hover:bg-rose-600 hover:text-white text-gray-300 border border-gray-700/60 whitespace-nowrap transition-all"
        >
          🚨 Report Fraud
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 p-4 overflow-y-auto space-y-4 text-sm">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`flex items-start gap-3 ${m.sender === "user" ? "justify-end" : "justify-start"}`}
          >
            {m.sender === "bot" && (
              <div className="w-7 h-7 rounded-lg bg-indigo-600 flex items-center justify-center text-white shrink-0 mt-0.5">
                <Bot className="w-4 h-4" />
              </div>
            )}
            <div
              className={`p-3.5 max-w-[85%] rounded-2xl shadow-sm ${
                m.sender === "user"
                  ? "bg-indigo-600 text-white rounded-tr-none"
                  : "bg-gray-800/90 text-gray-200 border border-gray-700/60 rounded-tl-none"
              }`}
            >
              <p className="leading-relaxed">{m.text}</p>
              {m.sender === "bot" && (
                <div className="mt-2 pt-2 border-t border-gray-700/60 flex items-center justify-between text-[10px] text-cyan-400 font-mono gap-2">
                  <span>Intent: {m.intent || "general"}</span>
                  {m.language && <span>Lang: {m.language.toUpperCase()}</span>}
                  <button
                    onClick={() => speakText(m.text, m.language)}
                    className="text-gray-400 hover:text-white ml-auto"
                    title="Read Aloud"
                  >
                    <Volume2 className="w-3.5 h-3.5" />
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="p-3.5 bg-gray-900/90 border-t border-gray-800 flex items-center gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          placeholder="Speak or ask in English, हिन्दी, ગુજરાતી, தமிழ்..."
          className="flex-1 bg-gray-950 text-white placeholder-gray-500 text-sm px-4 py-2.5 rounded-xl border border-gray-700/80 focus:ring-2 focus:ring-indigo-500 outline-none"
        />

        {/* Microphone ASR Button */}
        <button
          onClick={toggleRecording}
          type="button"
          className={`p-2.5 rounded-xl border transition-all cursor-pointer ${
            isRecording
              ? "bg-rose-600 border-rose-500 text-white animate-pulse"
              : "bg-gray-800 border-gray-700 text-gray-300 hover:bg-gray-700"
          }`}
          title={isRecording ? "Listening... Click to stop" : "Click to speak (Voice Input)"}
        >
          {isRecording ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4 text-cyan-400" />}
        </button>

        {/* Send Button */}
        <button
          onClick={() => sendMessage()}
          disabled={loading}
          className="p-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-cyan-500 text-white shadow-md transition-all cursor-pointer"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
