"use client";

import React from "react";
import { Sliders, Activity, ShieldAlert, Sparkles } from "lucide-react";

export interface DecisionTelemetry {
  stress: {
    stress_score: number;
    stress_band: string;
    drivers: string[];
    is_anomaly: boolean;
  };
  fraud: {
    has_anomaly: boolean;
    max_anomaly_score: number;
    flagged_signals: string[];
    anomalous_transactions_count: number;
  };
  recommendation: {
    path_used: string;
    top_recommendations: string[];
    suitability_score?: number;
    shap_reasons?: Record<string, string[]>;
  };
  audit_trail: Array<{
    tier: number;
    name: string;
    passed: boolean;
    reason: string;
  }>;
}

interface AIDecisionPanelProps {
  telemetry: DecisionTelemetry | null;
}

export const AIDecisionPanel: React.FC<AIDecisionPanelProps> = ({ telemetry }) => {
  if (!telemetry) {
    return (
      <div className="bg-gray-900/60 backdrop-blur-md rounded-2xl p-6 border border-gray-800 animate-pulse h-64 flex items-center justify-center">
        <span className="text-sm text-gray-400">Loading AI Engine Telemetry...</span>
      </div>
    );
  }

  const { stress, fraud, recommendation, audit_trail } = telemetry;

  return (
    <div className="bg-gray-900/80 backdrop-blur-md rounded-2xl p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Sliders className="w-5 h-5 text-indigo-400" />
          <h3 className="text-base font-bold text-white tracking-tight">AI Decision Panel</h3>
          <span className="text-[11px] text-gray-400 font-mono bg-gray-950 px-2 py-0.5 rounded border border-gray-800">
            Visible Telemetry
          </span>
        </div>
        <span className="text-xs text-gray-400">Non-Hidden Explainability</span>
      </div>

      {/* 3-Engine Telemetry Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        
        {/* 1. Stress Engine */}
        <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 font-medium">
              <span>1. Stress Engine</span>
              <Activity className="w-4 h-4 text-emerald-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-white font-mono">
                {(stress.stress_score || 0).toFixed(1)}
              </span>
              <span className="text-xs text-gray-400">/ 100</span>
            </div>
            <span
              className={`inline-block mt-1 px-2 py-0.5 text-[11px] font-bold rounded border ${
                stress.stress_band === "SEVERE" || stress.stress_band === "HIGH"
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
                  : stress.stress_band === "MODERATE" || stress.stress_band === "MILD"
                  ? "bg-amber-500/20 text-amber-300 border-amber-500/30"
                  : "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
              }`}
            >
              {stress.stress_band}
            </span>
          </div>
          <div className="mt-3 pt-2 border-t border-gray-800 text-[11px] text-gray-400 font-mono">
            {stress.drivers?.length ? stress.drivers.slice(0, 2).join(" • ") : "Healthy buffer balance"}
          </div>
        </div>

        {/* 2. Fraud & Anomaly Engine */}
        <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 font-medium">
              <span>2. Fraud Engine</span>
              <ShieldAlert className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-2xl font-extrabold text-white font-mono">
                {(fraud.max_anomaly_score || 0).toFixed(2)}
              </span>
              <span className="text-xs text-gray-400">Anomaly Index</span>
            </div>
            <span
              className={`inline-block mt-1 px-2 py-0.5 text-[11px] font-bold rounded border ${
                fraud.has_anomaly
                  ? "bg-rose-500/20 text-rose-300 border-rose-500/30"
                  : "bg-cyan-500/20 text-cyan-300 border-cyan-500/30"
              }`}
            >
              {fraud.has_anomaly ? "ANOMALY DETECTED" : "CLEAN / NORMAL"}
            </span>
          </div>
          <div className="mt-3 pt-2 border-t border-gray-800 text-[11px] text-gray-400 font-mono">
            {fraud.flagged_signals?.length ? `Flags: ${fraud.flagged_signals.join(", ")}` : "0 anomalies flagged"}
          </div>
        </div>

        {/* 3. Recommendation Engine */}
        <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-800 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between text-xs text-gray-400 font-medium">
              <span>3. Rec Engine</span>
              <Sparkles className="w-4 h-4 text-indigo-400" />
            </div>
            <div className="mt-2 flex items-baseline gap-2">
              <span className="text-base font-extrabold text-indigo-300 font-mono">
                {recommendation.path_used === "gmm_fallback" ? "GMM Fallback" : "XGBoost+SHAP"}
              </span>
            </div>
            <span className="inline-block mt-1 px-2 py-0.5 text-[11px] font-bold rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
              Top: {(recommendation.top_recommendations?.[0] || "SAVINGS").replace("_", " ").toUpperCase()}
            </span>
          </div>
          <div className="mt-3 pt-2 border-t border-gray-800 text-[11px] text-gray-400 font-mono">
            Path: {recommendation.path_used || "xgboost"}
          </div>
        </div>

      </div>

      {/* Priority Arbitration Hierarchy Evaluation Trail */}
      <div className="mt-5 pt-4 border-t border-gray-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-bold text-gray-300 uppercase tracking-wider">
            Arbitration Priority Evaluation Log
          </span>
          <span className="text-[11px] text-indigo-400 font-mono">Hierarchy (T1 -&gt; T6)</span>
        </div>
        <div className="space-y-1.5 font-mono text-xs text-gray-400 bg-gray-950/70 p-3 rounded-xl border border-gray-800/80 max-h-32 overflow-y-auto">
          {audit_trail?.map((step, idx) => (
            <div key={idx} className="flex items-center justify-between">
              <span className={step.passed ? "text-emerald-400 font-bold" : "text-gray-400"}>
                {step.passed ? "✓" : "○"} {step.name || `Tier ${step.tier}`}
              </span>
              <span className="text-gray-500 text-[10px]">{step.reason}</span>
            </div>
          ))}
        </div>
      </div>

    </div>
  );
};
