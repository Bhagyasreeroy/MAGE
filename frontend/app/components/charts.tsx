'use client';

/**
 * Minimal, dependency-free chart primitives for rendering VisualizationAgent
 * chart specs. Plain SVG/divs — no charting library — since what matters
 * here is that the numbers are real, not the visual polish.
 */

interface HistogramBin {
  label: string;
  count: number;
}

export function Histogram({ bins }: { bins: HistogramBin[] }) {
  const max = Math.max(1, ...bins.map((b) => b.count));
  return (
    <div className="flex items-end gap-1 h-32">
      {bins.map((bin, i) => (
        <div key={i} className="flex-1 flex flex-col items-center justify-end h-full group relative">
          <div
            className="w-full bg-navy/70 rounded-t hover:bg-navy transition-colors"
            style={{ height: `${(bin.count / max) * 100}%`, minHeight: bin.count > 0 ? 2 : 0 }}
            title={`${bin.label}: ${bin.count}`}
          />
          <span className="text-[8px] text-navy/40 mt-1 rotate-45 origin-top-left whitespace-nowrap">
            {bin.label}
          </span>
        </div>
      ))}
    </div>
  );
}

export function BarChart({ items }: { items: { label: string; value: number }[] }) {
  const max = Math.max(1e-9, ...items.map((i) => Math.abs(i.value)));
  return (
    <div className="space-y-2">
      {items.map((item, i) => (
        <div key={i} className="flex items-center gap-3 text-xs">
          <span className="w-28 shrink-0 truncate text-navy/60">{item.label}</span>
          <div className="flex-1 bg-cream-dark/50 rounded-full h-3 overflow-hidden">
            <div
              className="bg-navy/70 h-full rounded-full"
              style={{ width: `${(Math.abs(item.value) / max) * 100}%` }}
            />
          </div>
          <span className="w-12 shrink-0 text-right text-navy/50 font-mono">
            {typeof item.value === 'number' ? item.value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : item.value}
          </span>
        </div>
      ))}
    </div>
  );
}

export function BoxPlot({
  min,
  q1,
  median,
  q3,
  max,
}: {
  min: number | null;
  q1: number | null;
  median: number | null;
  q3: number | null;
  max: number | null;
  outlierBounds?: [number | null, number | null] | null;
}) {
  if (min == null || q1 == null || median == null || q3 == null || max == null) {
    return <p className="text-xs text-navy/40">Not enough data for a box plot.</p>;
  }
  const range = max - min || 1;
  const pct = (v: number) => ((v - min) / range) * 100;

  return (
    <div className="py-4">
      <svg viewBox="0 0 100 20" className="w-full h-10" preserveAspectRatio="none">
        <line x1={pct(min)} y1="10" x2={pct(max)} y2="10" stroke="#9a8c98" strokeWidth="0.5" />
        <rect x={pct(q1)} y="4" width={pct(q3) - pct(q1)} height="12" fill="#22223b" opacity="0.7" />
        <line x1={pct(median)} y1="2" x2={pct(median)} y2="18" stroke="#f2e9e4" strokeWidth="0.8" />
        <line x1={pct(min)} y1="6" x2={pct(min)} y2="14" stroke="#9a8c98" strokeWidth="0.5" />
        <line x1={pct(max)} y1="6" x2={pct(max)} y2="14" stroke="#9a8c98" strokeWidth="0.5" />
      </svg>
      <div className="flex justify-between text-[10px] text-navy/40 font-mono mt-1">
        <span>min {min.toFixed(1)}</span>
        <span>q1 {q1.toFixed(1)}</span>
        <span>median {median.toFixed(1)}</span>
        <span>q3 {q3.toFixed(1)}</span>
        <span>max {max.toFixed(1)}</span>
      </div>
    </div>
  );
}

export function CorrelationHeatmap({ columns, matrix }: { columns: string[]; matrix: (number | null)[][] }) {
  const cellColor = (v: number | null) => {
    if (v == null) return '#e5e5e5';
    const intensity = Math.min(1, Math.abs(v));
    return v >= 0
      ? `rgba(34, 34, 59, ${intensity})` // navy
      : `rgba(199, 62, 29, ${intensity})`; // dusty-rose-ish
  };

  return (
    <div className="overflow-x-auto">
      <table className="text-[10px] border-collapse">
        <thead>
          <tr>
            <th className="p-1" />
            {columns.map((c) => (
              <th key={c} className="p-1 text-navy/50 font-medium whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((row, i) => (
            <tr key={row}>
              <td className="p-1 text-navy/50 font-medium whitespace-nowrap">{row}</td>
              {matrix[i].map((v, j) => (
                <td
                  key={j}
                  className="w-9 h-9 text-center text-white font-mono"
                  style={{ backgroundColor: cellColor(v) }}
                  title={`${row} × ${columns[j]}: ${v?.toFixed(2) ?? 'n/a'}`}
                >
                  {v?.toFixed(2) ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ClusterScatter({ points }: { points: { x: number; y: number; cluster: number }[] }) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const colors = ['#22223b', '#9a8c98', '#c9ada7', '#4a4e69', '#f2e9e4', '#c73e1d'];

  return (
    <svg viewBox="0 0 100 100" className="w-full h-56 bg-cream/40 rounded-xl">
      {points.map((p, i) => (
        <circle
          key={i}
          cx={((p.x - minX) / rangeX) * 90 + 5}
          cy={90 - ((p.y - minY) / rangeY) * 90 + 5}
          r="1.2"
          fill={colors[p.cluster % colors.length]}
          opacity="0.75"
        />
      ))}
    </svg>
  );
}
