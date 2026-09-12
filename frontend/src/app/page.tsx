"use client";

import React, { useState, useEffect } from "react";
import { PersonaSelector, Persona } from "@/components/PersonaSelector";
import { ArbitrationHero, ArbitrationResult } from "@/components/ArbitrationHero";
import { AIDecisionPanel, DecisionTelemetry } from "@/components/AIDecisionPanel";
import { ChatAssistant } from "@/components/ChatAssistant";
import { Fingerprint } from "lucide-react";

const DEFAULT_PERSONAS: Persona[] = [
  {
    id: "stable_salaried",
    name: "Arjun Sharma",
    title: "Stable Salaried IT Professional",
    customer_id: "cust_86838bd208",
    archetype: "stable_salaried",
    income: "₹75,000 / month",
    expected_tier: "Tier 5 (Proactive Optimization)",
    expected_path: "xgboost",
    key_signals: "Predictable monthly salary credits, 28% savings rate, zero missed EMIs.",
    expected_action: "RECOMMEND_PRODUCTS (Mutual Fund SIP, Term Insurance)",
  },
  {
    id: "gig_irregular",
    name: "Ravi Patel",
    title: "Gig Economy Driver / Partner",
    customer_id: "cust_9d6cacdc98",
    archetype: "gig_irregular",
    income: "₹32,000 / month (Variable)",
    expected_tier: "Tier 5 (Proactive Optimization)",
    expected_path: "xgboost",
    key_signals: "Daily/weekly micro-credits, fluctuating cash buffer, active bike loan.",
    expected_action: "RECOMMEND_PRODUCTS (Flexible Micro-Credit, Small-Ticket RD)",
  },
  {
    id: "financially_stressed",
    name: "Deepak Verma",
    title: "Distressed MSME Proprietor",
    customer_id: "cust_ffb3c15320",
    archetype: "financially_stressed",
    income: "₹32,500 / month (Declining)",
    expected_tier: "Tier 2 (Stress Intervention)",
    expected_path: "deterministic_stress_relief",
    key_signals: "3 missed EMIs, -₹18k 30-day cash outflow trend, high debt service ratio.",
    expected_action: "STRESS_INTERVENTION (Suppresses credit cards/loans, offers EMI Restructuring)",
  },
  {
    id: "cold_start",
    name: "Kavita Rao",
    title: "New-to-Bank / Thin History Customer",
    customer_id: "cust_cold_start_new",
    archetype: "thin_history",
    income: "₹28,000 / month (Early Window)",
    expected_tier: "Tier 5 / Tier 6 (Demographic GMM Fallback)",
    expected_path: "gmm_fallback",
    key_signals: "<30 days Account Aggregator transaction history, unsupervised clustering routing.",
    expected_action: "GMM Fallback Safe Products (Recurring Deposit Small Ticket, Health Checkin)",
  },
];

export default function Home() {
  const [personas, setPersonas] = useState<Persona[]>(DEFAULT_PERSONAS);
  const [selectedPersonaId, setSelectedPersonaId] = useState<string>("stable_salaried");
  const [selectedLanguage, setSelectedLanguage] = useState<string>("en");
  const [isConsentActive, setIsConsentActive] = useState<boolean>(true);
  const [arbitration, setArbitration] = useState<ArbitrationResult | null>(null);
  const [telemetry, setTelemetry] = useState<DecisionTelemetry | null>(null);

  const currentPersona = personas.find((p) => p.id === selectedPersonaId) || personas[0];

  useEffect(() => {
    const fetchData = async () => {
      try {
        const resp = await fetch(`http://127.0.0.1:8000/customers/${currentPersona.customer_id}/arbitrate`);
        if (resp.ok) {
          const arb = await resp.json();
          setArbitration(arb);
          setTelemetry({
            stress: arb.stress_summary || { stress_score: 14.2, stress_band: "STABLE", drivers: [], is_anomaly: false },
            fraud: arb.fraud_summary || { has_anomaly: false, max_anomaly_score: 0.08, flagged_signals: [], anomalous_transactions_count: 0 },
            recommendation: arb.recommendation_summary || { path_used: arb.path_used, top_recommendations: [] },
            audit_trail: arb.arbitration_audit_trail || [],
          });
        }
      } catch (e) {
        console.error("Fetch error", e);
      }
    };
    fetchData();
  }, [selectedPersonaId, currentPersona.customer_id]);

  const toggleConsent = async () => {
    if (isConsentActive) {
      try {
        await fetch("http://127.0.0.1:8000/consent/revoke", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            customer_id: currentPersona.customer_id,
            consent_id: "active_token_123",
          }),
        });
      } catch (e) {}
      setIsConsentActive(false);
      alert("RBI Account Aggregator consent revoked. Data layer access is now locked.");
    } else {
      setIsConsentActive(true);
    }
  };

  return (
    <div className="min-h-screen flex flex-col antialiased bg-gray-950 text-gray-100">
      <PersonaSelector
        personas={personas}
        selectedId={selectedPersonaId}
        onSelectPersona={setSelectedPersonaId}
        selectedLanguage={selectedLanguage}
        onSelectLanguage={setSelectedLanguage}
        isConsentActive={isConsentActive}
        onToggleConsent={toggleConsent}
      />

      <main className="max-w-7xl mx-auto px-6 pt-6 flex-1 w-full grid grid-cols-1 lg:grid-cols-12 gap-6 pb-12">
        {/* Left 7 Cols */}
        <section className="lg:col-span-7 flex flex-col gap-6">
          {/* Persona Card */}
          <div className="bg-gray-900/80 backdrop-blur-md rounded-2xl p-5 border border-gray-800">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="flex items-center gap-2.5">
                  <h2 className="text-xl font-bold text-white">{currentPersona.name}</h2>
                  <span className="text-xs px-2.5 py-0.5 rounded-full font-semibold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 uppercase">
                    {currentPersona.archetype.replace("_", " ")}
                  </span>
                </div>
                <p className="text-xs text-gray-400 mt-0.5">{currentPersona.title}</p>
              </div>
              <div className="text-right">
                <span className="text-xs text-gray-400 block">Verified Monthly Inflow</span>
                <span className="text-lg font-extrabold text-emerald-400 font-mono">
                  {currentPersona.income}
                </span>
              </div>
            </div>

            <div className="mt-4 pt-3 border-t border-gray-800/80 flex items-center justify-between text-xs text-gray-300">
              <div className="flex items-center gap-2">
                <Fingerprint className="w-4 h-4 text-indigo-400" />
                <span>{currentPersona.key_signals}</span>
              </div>
              <span className="text-[11px] font-mono text-gray-400 bg-gray-950 px-2 py-0.5 rounded">
                {currentPersona.customer_id}
              </span>
            </div>
          </div>

          {/* Arbitration Hero */}
          <ArbitrationHero arbitration={arbitration} />

          {/* AI Decision Panel */}
          <AIDecisionPanel telemetry={telemetry} />
        </section>

        {/* Right 5 Cols: Chat Assistant */}
        <section className="lg:col-span-5">
          <ChatAssistant customerId={currentPersona.customer_id} selectedLanguage={selectedLanguage} />
        </section>
      </main>
    </div>
  );
}
