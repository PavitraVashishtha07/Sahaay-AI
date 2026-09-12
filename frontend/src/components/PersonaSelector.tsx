"use client";

import React from "react";
import { UserCheck, Lock, ShieldCheck } from "lucide-react";

export interface Persona {
  id: string;
  name: string;
  title: string;
  customer_id: string;
  archetype: string;
  income: string;
  expected_tier: string;
  expected_path: string;
  key_signals: string;
  expected_action: string;
}

interface PersonaSelectorProps {
  personas: Persona[];
  selectedId: string;
  onSelectPersona: (id: string) => void;
  selectedLanguage: string;
  onSelectLanguage: (lang: string) => void;
  isConsentActive: boolean;
  onToggleConsent: () => void;
}

export const PersonaSelector: React.FC<PersonaSelectorProps> = ({
  personas,
  selectedId,
  onSelectPersona,
  selectedLanguage,
  onSelectLanguage,
  isConsentActive,
  onToggleConsent,
}) => {
  return (
    <header className="sticky top-0 z-50 bg-gray-900/80 backdrop-blur-md border-b border-gray-800 px-6 py-3.5">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
        
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/30">
            <ShieldCheck className="w-6 h-6 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-extrabold tracking-tight bg-gradient-to-r from-white via-indigo-100 to-cyan-300 bg-clip-text text-transparent">
                Sahaay AI
              </span>
              <span className="px-2 py-0.5 text-xs font-semibold bg-indigo-500/20 text-indigo-300 rounded-full border border-indigo-500/30">
                Section 5
              </span>
            </div>
            <p className="text-xs text-gray-400 font-medium">Deterministic Financial Governance & Conversational Companion</p>
          </div>
        </div>

        {/* Persona & Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          
          {/* Persona Selector */}
          <div className="flex items-center bg-gray-900/90 rounded-xl p-1 border border-gray-700/60 shadow-inner">
            <label htmlFor="personaSelect" className="text-xs font-semibold px-2.5 text-gray-400 flex items-center gap-1.5">
              <UserCheck className="w-3.5 h-3.5 text-cyan-400" /> Persona:
            </label>
            <select
              id="personaSelect"
              value={selectedId}
              onChange={(e) => onSelectPersona(e.target.value)}
              className="bg-gray-800 text-sm font-semibold text-white px-3 py-1.5 rounded-lg border-0 focus:ring-2 focus:ring-indigo-500 outline-none cursor-pointer"
            >
              {personas.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.title.split(" ")[0]})
                </option>
              ))}
            </select>
          </div>

          {/* Multilingual Selector */}
          <div className="flex items-center bg-gray-900/90 rounded-xl p-1 border border-gray-700/60">
            <button
              onClick={() => onSelectLanguage("en")}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all ${
                selectedLanguage === "en" ? "bg-indigo-600 text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => onSelectLanguage("hi")}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all ${
                selectedLanguage === "hi" ? "bg-indigo-600 text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              हिन्दी
            </button>
            <button
              onClick={() => onSelectLanguage("gu")}
              className={`px-2.5 py-1 text-xs font-bold rounded-lg transition-all ${
                selectedLanguage === "gu" ? "bg-indigo-600 text-white" : "text-gray-400 hover:text-white"
              }`}
            >
              ગુજરાતી
            </button>
          </div>

          {/* Consent Gating Toggle */}
          <button
            onClick={onToggleConsent}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all ${
              isConsentActive
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 hover:bg-rose-500/20 hover:text-rose-300"
                : "bg-rose-500/20 text-rose-300 border border-rose-500/40"
            }`}
          >
            <Lock className="w-3.5 h-3.5" />
            <span>{isConsentActive ? "AA Consent Active" : "Consent Revoked"}</span>
          </button>

        </div>

      </div>
    </header>
  );
};
