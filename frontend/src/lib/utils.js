import { clsx } from "clsx";
import { twMerge } from "tailwind-merge"

export function cn(...inputs) {
  return twMerge(clsx(inputs));
}

// Unified compact number formatter across the app
// Rules:
// - 1 – 99,999 → 12,345
// - 100,000 – 999,999 → 123.4K (truncated)
// - 1,000,000 – 999,999,999 → 12.3M (truncated)
// - ≥ 1,000,000,000 → 1.2B (truncated)
export function formatCompactNumber(value) {
  if (value === null || value === undefined || isNaN(value)) return '0';
  const abs = Math.abs(value);
  if (abs < 100000) {
    return new Intl.NumberFormat('en-US').format(Math.round(value));
  }
  if (abs < 1000000) {
    const sign = value < 0 ? -1 : 1;
    const kAbs = abs / 1000;
    const truncated = Math.floor(kAbs * 10) / 10;
    const display = (sign * truncated).toFixed(1).replace(/\.0$/, '');
    return `${display}K`;
  }
  if (abs < 1000000000) {
    const sign = value < 0 ? -1 : 1;
    const mAbs = abs / 1000000;
    const truncated = Math.floor(mAbs * 10) / 10;
    const display = (sign * truncated).toFixed(1).replace(/\.0$/, '');
    return `${display}M`;
  }
  {
    const sign = value < 0 ? -1 : 1;
    const bAbs = abs / 1000000000;
    const truncated = Math.floor(bAbs * 10) / 10;
    const display = (sign * truncated).toFixed(1).replace(/\.0$/, '');
    return `${display}B`;
  }
}
