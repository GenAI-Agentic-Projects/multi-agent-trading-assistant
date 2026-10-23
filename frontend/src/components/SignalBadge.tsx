type SignalTone = "bullish" | "neutral" | "bearish" | "low" | "medium" | "high" | "buy_candidate" | "hold" | "sell_candidate";

const LABELS: Record<SignalTone, string> = {
  bullish: "Bullish",
  neutral: "Neutral",
  bearish: "Bearish",
  low: "Low",
  medium: "Medium",
  high: "High",
  buy_candidate: "Buy candidate",
  hold: "Hold",
  sell_candidate: "Sell candidate",
};

interface SignalBadgeProps {
  value: SignalTone;
  kind?: "trend" | "risk" | "classification" | "confidence";
}

export function SignalBadge({ value, kind = "trend" }: SignalBadgeProps) {
  return <span className={`signal-badge ${kind} tone-${value}`}>{LABELS[value]}</span>;
}