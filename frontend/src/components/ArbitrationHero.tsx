"use client";

import React from "react";
import { TrendingUp, Shield, AlertTriangle, CheckCircle, Cpu, Eye } from "lucide-react";

export interface ArbitrationResult {
  customer_id: string;
  priority_tier_applied: string;
  final_action: string;
  action_headline: string;
  action_body: string;
  decision_rule_used: string;
  path_used: string;
}

interface ArbitrationHeroProps {
  arbitration: ArbitrationResult | null;
}

export const ArbitrationHero: React.FC<ArbitrationHeroProps> = ({ arbitration }) => {
  if (!arbitration) {
    return (
      <div className="bg-gray-900/60 backdrop-blur-md rounded-2xl p-6 border border-gray-800 animate-pulse h-48 flex items-center justify-center">
        <span className="text-sm text-gray-400">Loading live arbitration outcome...</span>
      </div>
    );
  }

  const tier = arbitration.priority_tier_applied || "TIER_5_PRODUCT_RECOMMENDATION";

  let tierBadgeClass = "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
  let tierLabel = "Tier 5: Proactive Optimization";
  let containerBorder = "border-emerald-500/30 bg-emerald-950/10";

  if (tier.includes("TIER_1")) {
    tierBadgeClass = "bg-rose-500/20 text-rose-300 border-rose-500/40";
    tierLabel = "Tier 1: Critical Fraud Shield Active";
    containerBorder = "border-rose-500/40 bg-rose-950/10";
  } else if (tier.includes("TIER_2")) {
    tierBadgeClass = "bg-amber-500/20 text-amber-300 border-amber-500/40";
    tierLabel = "Tier 2: Financial Stress Intervention";
    containerBorder = "border-amber-500/40 bg-amber-950/10";
  } else if (tier.includes("TIER_6")) {
    tierBadgeClass = "bg-teal-500/20 text-teal-300 border-teal-500/40";
    tierLabel = "Tier 6: Peaceful Watch State";
    containerBorder = "border-teal-500/30 bg-teal-950/10";
  }

  return (
    <div className={`rounded-2xl p-6 border backdrop-blur-md transition-all relative overflow-hidden ${containerBorder}`}>
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <span className={`px-3 py-1 text-xs font-extrabold rounded-full tracking-wide uppercase border ${tierBadgeClass}`}>
            {tierLabel}
          </span>
          <span className="px-2.5 py-0.5 text-[11px] font-mono font-medium rounded-md bg-gray-800 text-gray-300 border border-gray-700">
            Path: {arbitration.path_used}
          </span>
        </div>
        <div className="flex items-center gap-1.5 text-xs text-gray-400 font-mono">
          <Cpu className="w-3.5 h-3.5 text-indigo-400" />
          <span>{arbitration.decision_rule_used}</span>
        </div>
      </div>

      <h3 className="text-2xl font-extrabold text-white tracking-tight mt-1">
        {arbitration.action_headline}
      </h3>
      <p className="text-sm text-gray-300 mt-2 leading-relaxed">
        {arbitration.action_body}
      </p>

      {/* Action Chips */}
      <div className="flex flex-wrap gap-2 mt-4 pt-3 border-t border-gray-800/80">
        {tier.includes("TIER_1") && (
          <span className="px-3 py-1 rounded-lg text-xs font-semibold bg-rose-900/50 text-rose-200 border border-rose-700/50 flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 text-rose-400" /> Freeze High-Risk Channels
          </span>
        )}
        {tier.includes("TIER_2") && (
          <span className="px-3 py-1 rounded-lg text-xs font-semibold bg-amber-900/50 text-amber-200 border border-amber-700/50 flex items-center gap-1.5">
            <Shield className="w-3.5 h-3.5 text-amber-400" /> Predatory Credit Blocked
          </span>
        )}
        {tier.includes("TIER_5") && (
          <>
            <span className="px-3 py-1 rounded-lg text-xs font-semibold bg-indigo-900/50 text-indigo-200 border border-indigo-700/50 flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-indigo-400" /> Verified Cash-Flow Match
            </span>
            <span className="px-3 py-1 rounded-lg text-xs font-semibold bg-indigo-900/50 text-indigo-200 border border-indigo-700/50 flex items-center gap-1.5">
              <CheckCircle className="w-3.5 h-3.5 text-indigo-400" /> SHAP Grounded Explanation
            </span>
          </>
        )}
        {tier.includes("TIER_6") && (
          <span className="px-3 py-1 rounded-lg text-xs font-semibold bg-teal-900/50 text-teal-200 border border-teal-700/50 flex items-center gap-1.5">
            <Eye className="w-3.5 h-3.5 text-teal-400" /> All Systems Peaceful
          </span>
        )}
      </div>
    </div>
  );
};
