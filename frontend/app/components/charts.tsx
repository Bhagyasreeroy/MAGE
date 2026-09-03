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

/** Categorical series palette — mirrors _CHART_PALETTE in export_service.py so
 *  a chart looks the same on screen and in the exported PDF. */
const SERIES_COLORS = ['#22223b', '#9a8c98', '#4a4e69', '#c9ada7', '#6d6875', '#b5838d'];
const OUTLIER_COLOR = '#c73e1d';

/** Every chart accepts this — 'large' is what the click-to-enlarge modal
 *  renders with, so a reader can actually make out individual points/bars
 *  instead of squinting at a card sized for a two-column grid. */
export type ChartSize = 'default' | 'large';

function sizeClass(defaultCls: string, largeCls: string, size: ChartSize = 'default'): string {
  return size === 'large' ? largeCls : defaultCls;
}

export function Histogram({ bins, size = 'default' }: { bins: HistogramBin[]; size?: ChartSize }) {
  const max = Math.max(1, ...bins.map((b) => b.count));
  return (
    <div className="w-full min-w-0">
      {size === 'large' && <p className="text-[10px] text-navy/35 mb-1">↑ count</p>}
      <div className={`flex items-end gap-1 ${sizeClass('h-56', 'h-96', size)}`}>
        {bins.map((bin, i) => (
          <div key={i} className="flex-1 min-w-0 flex flex-col items-center justify-end h-full group relative">
            <div
              className="w-full bg-navy/70 rounded-t hover:bg-navy transition-colors"
              style={{ height: `${(bin.count / max) * 100}%`, minHeight: bin.count > 0 ? 2 : 0 }}
              title={`${bin.label}: ${bin.count}`}
            />
          </div>
        ))}
      </div>
      {/* Labels in their own row (not rotated) so they can never push the
          chart wider than its container — every-other label if crowded. */}
      <div className="flex gap-1 mt-2">
        {bins.map((bin, i) => (
          <span
            key={i}
            className="flex-1 min-w-0 text-center text-[9px] text-navy/40 truncate"
          >
            {bins.length <= 10 || i % 2 === 0 ? bin.label : ''}
          </span>
        ))}
      </div>
    </div>
  );
}

export function BarChart({
  items,
  valueLabel,
  size = 'default',
}: {
  items: { label: string; value: number }[];
  valueLabel?: string;
  size?: ChartSize;
}) {
  const max = Math.max(1e-9, ...items.map((i) => Math.abs(i.value)));
  return (
    <div className={`w-full min-w-0 ${sizeClass('space-y-3', 'space-y-4', size)}`}>
      {valueLabel && <p className="text-[10px] text-navy/35">{valueLabel} →</p>}
      {items.map((item, i) => (
        <div key={i} className={`flex items-center gap-3 ${sizeClass('text-sm', 'text-base', size)}`}>
          <span className={`shrink-0 truncate text-navy/60 ${sizeClass('w-32', 'w-48', size)}`} title={item.label}>
            {item.label}
          </span>
          <div className={`flex-1 min-w-0 bg-cream-dark/50 rounded-full overflow-hidden ${sizeClass('h-4', 'h-6', size)}`}>
            <div
              className="bg-navy/70 h-full rounded-full"
              style={{ width: `${(Math.abs(item.value) / max) * 100}%` }}
            />
          </div>
          <span className="w-14 shrink-0 text-right text-navy/50 font-mono text-xs">
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
  size = 'default',
}: {
  min: number | null;
  q1: number | null;
  median: number | null;
  q3: number | null;
  max: number | null;
  outlierBounds?: [number | null, number | null] | null;
  size?: ChartSize;
}) {
  if (min == null || q1 == null || median == null || q3 == null || max == null) {
    return <p className="text-xs text-navy/40">Not enough data for a box plot.</p>;
  }

  // A handful of extreme outliers (the exact thing this chart exists to
  // surface) can be 10-100x the IQR on a raw min-max linear scale — the box
  // itself collapses to a sliver of a pixel and the chart reads as a blank
  // line. Scale to the Tukey whisker range (median ± 1.5 IQR, clamped to the
  // real data) instead, and draw anything beyond that as a marker at the
  // edge rather than stretching the axis to fit it.
  const iqr = q3 - q1;
  const whiskerLow = Math.max(min, q1 - 1.5 * iqr);
  const whiskerHigh = Math.min(max, q3 + 1.5 * iqr);
  const domainMin = whiskerLow < whiskerHigh ? whiskerLow : min;
  const domainMax = whiskerLow < whiskerHigh ? whiskerHigh : max;
  const domainRange = domainMax - domainMin || 1;

  const hasLowOutlier = min < domainMin;
  const hasHighOutlier = max > domainMax;
  // Leave room at the edges (0-8 / 92-100) for outlier markers.
  const pct = (v: number) => {
    const clamped = Math.min(domainMax, Math.max(domainMin, v));
    return ((clamped - domainMin) / domainRange) * 84 + 8;
  };

  return (
    <div className="py-4 w-full min-w-0">
      <svg viewBox="0 0 100 20" className={`w-full ${sizeClass('h-20', 'h-36', size)}`} preserveAspectRatio="none">
        <line x1={pct(domainMin)} y1="10" x2={pct(domainMax)} y2="10" stroke="#9a8c98" strokeWidth="0.5" />
        <rect x={pct(q1)} y="4" width={Math.max(0.5, pct(q3) - pct(q1))} height="12" fill="#22223b" opacity="0.7" />
        <line x1={pct(median)} y1="2" x2={pct(median)} y2="18" stroke="#f2e9e4" strokeWidth="0.8" />
        <line x1={pct(domainMin)} y1="6" x2={pct(domainMin)} y2="14" stroke="#9a8c98" strokeWidth="0.5" />
        <line x1={pct(domainMax)} y1="6" x2={pct(domainMax)} y2="14" stroke="#9a8c98" strokeWidth="0.5" />
        {hasLowOutlier && (
          <circle cx="2.5" cy="10" r="1.4" fill="#c73e1d" opacity="0.85">
            <title>{`Outlier(s) down to ${min.toFixed(2)}`}</title>
          </circle>
        )}
        {hasHighOutlier && (
          <circle cx="97.5" cy="10" r="1.4" fill="#c73e1d" opacity="0.85">
            <title>{`Outlier(s) up to ${max.toFixed(2)}`}</title>
          </circle>
        )}
      </svg>
      <div className="flex justify-between text-xs text-navy/40 font-mono mt-2">
        <span>min {min.toFixed(1)}</span>
        <span>q1 {q1.toFixed(1)}</span>
        <span>median {median.toFixed(1)}</span>
        <span>q3 {q3.toFixed(1)}</span>
        <span>max {max.toFixed(1)}</span>
      </div>
    </div>
  );
}

export function CorrelationHeatmap({
  columns,
  matrix,
  size = 'default',
}: {
  columns: string[];
  matrix: (number | null)[][];
  size?: ChartSize;
}) {
  // Tints run 0.08 -> 0.50 alpha, never to full strength, so every cell stays
  // light enough for one constant dark ink. The obvious alternative — alpha
  // straight to 1.0 with white text — makes a near-zero correlation white on
  // near-white and its number vanishes, which is exactly what a reader most
  // needs to see spelled out. Capping the ramp costs a little punch on the
  // 1.00 diagonal and buys 4.8:1 contrast on the worst cell in the matrix.
  const CELL_ALPHA_FLOOR = 0.08;
  const CELL_ALPHA_RANGE = 0.42;
  const cellColor = (v: number | null) => {
    if (v == null) return 'var(--color-cream-dark)';
    const alpha = CELL_ALPHA_FLOOR + CELL_ALPHA_RANGE * Math.min(1, Math.abs(v));
    return v >= 0
      ? `rgba(26, 23, 66, ${alpha})` // navy — positive
      : `rgba(199, 62, 29, ${alpha})`; // rust — negative
  };
  const cellCls = sizeClass('w-12 h-12', 'w-16 h-16 text-sm', size);

  return (
    <div className="w-full min-w-0 overflow-x-auto">
      <table className={sizeClass('text-xs', 'text-sm', size) + ' border-collapse'}>
        <thead>
          <tr>
            <th className="p-1" />
            {columns.map((c) => (
              <th key={c} className="p-1.5 text-navy/50 font-medium whitespace-nowrap">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {columns.map((row, i) => (
            <tr key={row}>
              <td className="p-1.5 text-navy/50 font-medium whitespace-nowrap">{row}</td>
              {matrix[i].map((v, j) => (
                <td
                  key={j}
                  className={`text-center font-mono ${v == null ? 'text-navy/70' : 'text-navy'} ${cellCls}`}
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

export function GroupedBar({
  categories,
  series,
  groupLabel,
  valueLabel,
  size = 'default',
}: {
  categories: string[];
  series: { name: string; values: number[] }[];
  groupLabel?: string;
  valueLabel?: string;
  size?: ChartSize;
}) {
  const max = Math.max(1, ...series.flatMap((s) => s.values));
  return (
    <div className="w-full min-w-0">
      {size === 'large' && <p className="text-[10px] text-navy/35 mb-1">↑ {valueLabel || 'count'}</p>}
      <div className={`flex items-end gap-3 ${sizeClass('h-56', 'h-96', size)}`}>
        {categories.map((category, ci) => (
          <div key={ci} className="flex-1 min-w-0 flex items-end justify-center gap-0.5 h-full">
            {series.map((s, si) => (
              <div
                key={si}
                className="flex-1 min-w-0 rounded-t transition-opacity hover:opacity-80"
                style={{
                  height: `${((s.values[ci] ?? 0) / max) * 100}%`,
                  minHeight: (s.values[ci] ?? 0) > 0 ? 2 : 0,
                  backgroundColor: SERIES_COLORS[si % SERIES_COLORS.length],
                }}
                title={`${category} · ${s.name}: ${s.values[ci] ?? 0}`}
              />
            ))}
          </div>
        ))}
      </div>
      <div className="flex gap-3 mt-2">
        {categories.map((category, ci) => (
          <span key={ci} className="flex-1 min-w-0 text-center text-[9px] text-navy/40 truncate">
            {category}
          </span>
        ))}
      </div>
      {/* Without a legend a grouped bar is just coloured rectangles. */}
      <div className="flex flex-wrap gap-3 mt-3 pt-3 border-t border-dusty-rose/15">
        {series.map((s, si) => (
          <span key={si} className="flex items-center gap-1.5 text-[10px] text-navy/50">
            <span
              className="w-2.5 h-2.5 rounded-sm shrink-0"
              style={{ backgroundColor: SERIES_COLORS[si % SERIES_COLORS.length] }}
            />
            {s.name}
          </span>
        ))}
        {groupLabel && <span className="text-[10px] text-navy/30 ml-auto">grouped by {groupLabel}</span>}
      </div>
    </div>
  );
}

export function BoxByClass({
  groups,
  size = 'default',
}: {
  groups: { label: string; count: number; min: number; q1: number; median: number; q3: number; max: number }[];
  size?: ChartSize;
}) {
  if (groups.length === 0) return <p className="text-xs text-navy/40">No class groups to compare.</p>;
  // One shared scale across classes — per-class scales would make every box
  // look the same width and hide the separation this chart exists to show.
  const lo = Math.min(...groups.map((g) => g.min));
  const hi = Math.max(...groups.map((g) => g.max));
  const range = hi - lo || 1;
  const pct = (v: number) => ((v - lo) / range) * 100;

  return (
    <div className="space-y-3 w-full min-w-0 py-2">
      {groups.map((g, i) => (
        <div key={i} className="flex items-center gap-3">
          <span className="w-20 shrink-0 truncate text-xs text-navy/60" title={g.label}>
            {g.label}
          </span>
          <svg viewBox="0 0 100 12" className={`flex-1 min-w-0 ${sizeClass('h-8', 'h-14', size)}`} preserveAspectRatio="none">
            <line x1={pct(g.min)} y1="6" x2={pct(g.max)} y2="6" stroke="#9a8c98" strokeWidth="0.4" />
            <rect
              x={pct(g.q1)}
              y="2"
              width={Math.max(0.5, pct(g.q3) - pct(g.q1))}
              height="8"
              fill={SERIES_COLORS[i % SERIES_COLORS.length]}
              opacity="0.75"
            />
            <line x1={pct(g.median)} y1="1" x2={pct(g.median)} y2="11" stroke="#f2e9e4" strokeWidth="0.7" />
          </svg>
          <span className="w-14 shrink-0 text-right text-[10px] text-navy/40 font-mono">n={g.count}</span>
        </div>
      ))}
      <div className="flex justify-between text-[10px] text-navy/35 font-mono pt-1">
        <span>{lo.toFixed(1)}</span>
        <span>{hi.toFixed(1)}</span>
      </div>
    </div>
  );
}

export function Pairplot({
  pairs,
  size = 'default',
}: {
  pairs: { x_label: string; y_label: string; r: number; points: { x: number; y: number }[] }[];
  size?: ChartSize;
}) {
  return (
    <div className={size === 'large' ? 'grid grid-cols-2 gap-6 w-full min-w-0' : 'grid grid-cols-2 gap-3 w-full min-w-0'}>
      {pairs.map((pair, i) => {
        // Scale to the 2nd-98th percentile, not min-max. A handful of extreme
        // values (exactly what a planted outlier looks like) otherwise own the
        // whole axis and squash every real point onto a single line — the
        // panel renders, and shows nothing. Out-of-range points are clamped to
        // the edge rather than dropped, so the panel never lies about how many
        // observations there are. Same reasoning as the Tukey scaling in
        // BoxPlot above.
        const pct = (arr: number[], q: number) => {
          const a = [...arr].sort((m, n) => m - n);
          return a[Math.min(a.length - 1, Math.max(0, Math.floor(q * (a.length - 1))))];
        };
        const xs = pair.points.map((p) => p.x);
        const ys = pair.points.map((p) => p.y);
        const minX = pct(xs, 0.02);
        const minY = pct(ys, 0.02);
        const rangeX = pct(xs, 0.98) - minX || 1;
        const rangeY = pct(ys, 0.98) - minY || 1;
        const clamp = (v: number) => Math.min(1, Math.max(0, v));
        return (
          <div key={i} className="min-w-0">
            <svg
              viewBox="0 0 100 100"
              className={`w-full bg-cream/40 rounded-lg ${sizeClass('h-28', 'h-64', size)}`}
            >
              {pair.points.map((p, j) => (
                <circle
                  key={j}
                  cx={clamp((p.x - minX) / rangeX) * 90 + 5}
                  cy={90 - clamp((p.y - minY) / rangeY) * 90 + 5}
                  r="1.6"
                  fill={SERIES_COLORS[i % SERIES_COLORS.length]}
                  opacity="0.65"
                />
              ))}
            </svg>
            {size === 'large' && (
              <div className="flex justify-between text-[9px] text-navy/35 font-mono mt-1">
                <span>{pair.x_label}: {minX.toFixed(1)}–{(minX + rangeX).toFixed(1)}</span>
                <span>{pair.y_label}: {minY.toFixed(1)}–{(minY + rangeY).toFixed(1)}</span>
              </div>
            )}
            <p className="text-[9px] text-navy/40 mt-1 truncate" title={`${pair.x_label} × ${pair.y_label}`}>
              {pair.x_label} × {pair.y_label}{' '}
              <span className="font-mono text-navy/30">r={pair.r.toFixed(2)}</span>
            </p>
          </div>
        );
      })}
    </div>
  );
}

export function HighlightedScatter({
  points,
  xLabel,
  yLabel,
  size = 'default',
}: {
  points: { x: number; y: number; outlier?: boolean }[];
  xLabel?: string;
  yLabel?: string;
  size?: ChartSize;
}) {
  if (points.length === 0) return <p className="text-xs text-navy/40">No points to plot.</p>;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const rangeX = Math.max(...xs) - minX || 1;
  const rangeY = Math.max(...ys) - minY || 1;
  const flagged = points.filter((p) => p.outlier).length;

  return (
    <div className="w-full min-w-0">
      <svg viewBox="0 0 100 100" className={`w-full bg-cream/40 rounded-xl ${sizeClass('h-72', 'h-[32rem]', size)}`}>
        {/* Normal points first so the flagged ones are never painted over. */}
        {points.map((p, i) =>
          p.outlier ? null : (
            <circle
              key={i}
              cx={((p.x - minX) / rangeX) * 90 + 5}
              cy={90 - ((p.y - minY) / rangeY) * 90 + 5}
              r="1.2"
              fill="#9a8c98"
              opacity="0.55"
            />
          ),
        )}
        {points.map((p, i) =>
          p.outlier ? (
            <circle
              key={`o-${i}`}
              cx={((p.x - minX) / rangeX) * 90 + 5}
              cy={90 - ((p.y - minY) / rangeY) * 90 + 5}
              r="2.2"
              fill={OUTLIER_COLOR}
              stroke="#f2e9e4"
              strokeWidth="0.4"
            />
          ) : null,
        )}
      </svg>
      <div className="flex justify-between items-center text-[10px] text-navy/40 mt-2">
        <span>
          {xLabel ? `${xLabel} →` : ''}
          {xLabel && yLabel ? '  ·  ' : ''}
          {yLabel ? `↑ ${yLabel}` : ''}
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: OUTLIER_COLOR }} />
          {flagged} outlier{flagged === 1 ? '' : 's'}
        </span>
      </div>
      {size === 'large' && (
        <div className="flex justify-between text-[9px] text-navy/30 font-mono mt-1">
          <span>x: {minX.toFixed(1)}–{(minX + rangeX).toFixed(1)}</span>
          <span>y: {minY.toFixed(1)}–{(minY + rangeY).toFixed(1)}</span>
        </div>
      )}
    </div>
  );
}

export function Violin({
  bands,
  median,
  size = 'default',
}: {
  bands: { center: number; count: number; width: number }[];
  median?: number | null;
  size?: ChartSize;
}) {
  if (bands.length === 0) return <p className="text-xs text-navy/40">No distribution to show.</p>;
  return (
    <div className="w-full min-w-0 py-2">
      {/* Bands run low-to-high bottom-up and are centred, so the silhouette
          mirrors about the axis the way a violin does. */}
      <div className={`flex flex-col-reverse items-center justify-center gap-px ${sizeClass('h-56', 'h-96', size)}`}>
        {bands.map((band, i) => (
          <div
            key={i}
            className="rounded-sm"
            style={{
              width: `${Math.max(2, band.width * 100)}%`,
              height: `${100 / bands.length}%`,
              backgroundColor: SERIES_COLORS[0],
              opacity: 0.35 + band.width * 0.45,
            }}
            title={`≈${band.center}: ${band.count} row${band.count === 1 ? '' : 's'}`}
          />
        ))}
      </div>
      <div className="flex justify-between text-[10px] text-navy/40 font-mono mt-2">
        <span>{bands[0].center}</span>
        {median != null && <span>median {median}</span>}
        <span>{bands[bands.length - 1].center}</span>
      </div>
    </div>
  );
}

export function LineChart({
  points,
  xLabel,
  yLabel,
  size = 'default',
}: {
  points: { x: string; y: number }[];
  xLabel?: string;
  yLabel?: string;
  size?: ChartSize;
}) {
  if (points.length < 2) return <p className="text-xs text-navy/40">Not enough points for a trend.</p>;
  const ys = points.map((p) => p.y);
  const minY = Math.min(...ys);
  const rangeY = Math.max(...ys) - minY || 1;
  const coords = points.map((p, i) => {
    const x = (i / (points.length - 1)) * 90 + 5;
    const y = 90 - ((p.y - minY) / rangeY) * 80 + 5;
    return `${x},${y}`;
  });

  return (
    <div className="w-full min-w-0">
      {size === 'large' && (
        <p className="text-[10px] text-navy/35 mb-1">
          ↑ {yLabel || 'value'} (range {minY.toFixed(1)}–{(minY + rangeY).toFixed(1)})
        </p>
      )}
      <svg
        viewBox="0 0 100 100"
        className={`w-full bg-cream/40 rounded-xl ${sizeClass('h-64', 'h-[28rem]', size)}`}
        preserveAspectRatio="none"
      >
        <polyline points={coords.join(' ')} fill="none" stroke={SERIES_COLORS[0]} strokeWidth="0.8" />
      </svg>
      <div className="flex justify-between text-[10px] text-navy/40 font-mono mt-2">
        <span>{points[0].x.slice(0, 10)}</span>
        <span className="text-navy/30">
          {yLabel}
          {xLabel ? ` over ${xLabel}` : ''}
        </span>
        <span>{points[points.length - 1].x.slice(0, 10)}</span>
      </div>
    </div>
  );
}

export function ScatterPlot({
  points,
  xLabel,
  yLabel,
  size = 'default',
}: {
  points: { x: number; y: number }[];
  xLabel?: string;
  yLabel?: string;
  size?: ChartSize;
}) {
  if (points.length === 0) {
    return <p className="text-xs text-navy/40">Not enough data for a scatter plot.</p>;
  }
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  return (
    <div className="w-full min-w-0">
      <svg
        viewBox="0 0 100 100"
        className={`w-full max-w-full bg-cream/40 rounded-xl ${sizeClass('h-80', 'h-[32rem]', size)}`}
      >
        {points.map((p, i) => (
          <circle
            key={i}
            cx={((p.x - minX) / rangeX) * 90 + 5}
            cy={90 - ((p.y - minY) / rangeY) * 90 + 5}
            r="1.2"
            fill={SERIES_COLORS[0]}
            opacity="0.6"
          />
        ))}
      </svg>
      {(xLabel || yLabel) && (
        <div className="flex justify-between text-[10px] text-navy/40 mt-1.5">
          <span>{yLabel ? `↑ ${yLabel}` : ''}</span>
          <span>{xLabel ? `${xLabel} →` : ''}</span>
        </div>
      )}
      {size === 'large' && (
        <div className="flex justify-between text-[9px] text-navy/30 font-mono mt-1">
          <span>x: {minX.toFixed(1)}–{maxX.toFixed(1)}</span>
          <span>y: {minY.toFixed(1)}–{maxY.toFixed(1)}</span>
        </div>
      )}
    </div>
  );
}

export function ClusterScatter({
  points,
  size = 'default',
}: {
  points: { x: number; y: number; cluster: number }[];
  size?: ChartSize;
}) {
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const [minX, maxX] = [Math.min(...xs), Math.max(...xs)];
  const [minY, maxY] = [Math.min(...ys), Math.max(...ys)];
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  const colors = ['#22223b', '#9a8c98', '#c9ada7', '#4a4e69', '#f2e9e4', '#c73e1d'];
  // -1 is DBSCAN's own convention for a noise point — every other backend
  // (KMeans) numbers clusters 0..k. Both feed this component, so it's
  // labelled generically here rather than baking in a clustering-method name.
  const clusterIds = [...new Set(points.map((p) => p.cluster))].sort((a, b) => a - b);

  return (
    <div className="w-full min-w-0">
      {/* The two axes are the dataset's numeric columns projected down to 2D
          via PCA — there's no single original column to name, so the
          components are what's actually being plotted (labelled below). */}
      <svg
        viewBox="0 0 100 100"
        className={`w-full max-w-full bg-cream/40 rounded-xl ${sizeClass('h-80', 'h-[32rem]', size)}`}
      >
        {points.map((p, i) => (
          <circle
            key={i}
            cx={((p.x - minX) / rangeX) * 90 + 5}
            cy={90 - ((p.y - minY) / rangeY) * 90 + 5}
            r="1.2"
            fill={p.cluster === -1 ? '#c73e1d' : colors[p.cluster % colors.length]}
            opacity="0.75"
          >
            <title>{p.cluster === -1 ? 'noise' : `cluster ${p.cluster}`}</title>
          </circle>
        ))}
      </svg>
      <div className="flex justify-between text-[10px] text-navy/40 mt-1.5">
        <span>↑ PC2</span>
        <span>PC1 →</span>
      </div>
      <div className="flex flex-wrap gap-3 mt-2">
        {clusterIds.map((id) => (
          <span key={id} className="flex items-center gap-1.5 text-[10px] text-navy/50">
            <span
              className="w-2.5 h-2.5 rounded-full shrink-0"
              style={{ backgroundColor: id === -1 ? '#c73e1d' : colors[id % colors.length] }}
            />
            {id === -1 ? 'noise' : `cluster ${id}`}
          </span>
        ))}
      </div>
      {size === 'large' && (
        <div className="flex justify-between text-[9px] text-navy/30 font-mono mt-1">
          <span>PC1: {minX.toFixed(2)}–{maxX.toFixed(2)}</span>
          <span>PC2: {minY.toFixed(2)}–{maxY.toFixed(2)}</span>
        </div>
      )}
    </div>
  );
}

// Panel geometry, shared by every decomposition panel so the three stack in
// register against one time axis.
const DECOMP_W = 600;
const DECOMP_H = 90;

/** One panel of a decomposition. Declared at module scope: defining it inside
 *  the parent's render would rebuild the component type on every render and
 *  reset its subtree. */
function DecompositionPanel({
  label,
  children,
  size = 'default',
}: {
  label: string;
  children: React.ReactNode;
  size?: ChartSize;
}) {
  return (
    <div className="min-w-0">
      <p className="text-[10px] font-bold text-navy/40 uppercase tracking-widest">{label}</p>
      <svg
        viewBox={`0 0 ${DECOMP_W} ${DECOMP_H}`}
        className={`w-full ${sizeClass('h-24', 'h-48', size)} bg-cream/40 rounded-xl mt-1`}
        role="img"
      >
        {children}
      </svg>
    </div>
  );
}

type DecompositionPoint = {
  t: string;
  observed: number | null;
  trend: number | null;
  seasonal: number | null;
  residual: number | null;
};

/**
 * Seasonal decomposition as three stacked panels sharing one time axis.
 *
 * Small multiples rather than one plot, because the components have genuinely
 * different scales: a residual sits around zero while the observed series may
 * be in the hundreds. Sharing a y-axis would flatten the residual to a
 * straight line; giving them separate y-axes on the same plot would be a
 * dual-axis chart, which is worse than either.
 *
 * `trend` and `residual` are undefined for the first and last half-period —
 * seasonal_decompose has no window there — so the line is *broken* at those
 * points rather than drawn through zero, which would invent a downward spike
 * at both ends of every chart.
 */
export function SeasonalDecomposition({
  points,
  xLabel,
  yLabel,
  resampled,
  size = 'default',
}: {
  points: DecompositionPoint[];
  xLabel?: string;
  yLabel?: string;
  resampled?: string;
  size?: ChartSize;
}) {
  if (points.length < 2) return <p className="text-xs text-navy/40">Not enough points to decompose.</p>;

  // One path per contiguous run of defined values, so gaps stay gaps.
  const pathFor = (values: (number | null)[], min: number, range: number) => {
    const runs: string[] = [];
    let current: string[] = [];
    values.forEach((v, i) => {
      if (v === null || Number.isNaN(v)) {
        if (current.length > 1) runs.push(current.join(' '));
        current = [];
        return;
      }
      const x = (i / (values.length - 1)) * DECOMP_W;
      const y = DECOMP_H - ((v - min) / range) * (DECOMP_H - 12) - 6;
      current.push(`${current.length === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`);
    });
    if (current.length > 1) runs.push(current.join(' '));
    return runs;
  };

  const extent = (values: (number | null)[]) => {
    const nums = values.filter((v): v is number => v !== null && !Number.isNaN(v));
    if (!nums.length) return { min: 0, range: 1 };
    const min = Math.min(...nums);
    return { min, range: Math.max(...nums) - min || 1 };
  };

  const observed = points.map((p) => p.observed);
  const trend = points.map((p) => p.trend);
  const seasonal = points.map((p) => p.seasonal);
  const residual = points.map((p) => p.residual);

  // Panel 1 shares one scale across observed and trend — they are the same
  // quantity, so a common scale is the whole point of overlaying them.
  const topNums = [...observed, ...trend].filter((v): v is number => v !== null && !Number.isNaN(v));
  const topMin = Math.min(...topNums);
  const topRange = Math.max(...topNums) - topMin || 1;

  const seasonalExtent = extent(seasonal);
  const residualExtent = extent(residual);
  const zeroY = (min: number, range: number) =>
    DECOMP_H - ((0 - min) / range) * (DECOMP_H - 12) - 6;

  return (
    <div className="w-full min-w-0 space-y-3">
      {/* Two series in one panel, so identity is never carried by colour
          alone — and the lighter series sits just under 3:1 against the
          surface, which obliges a visible label rather than a bare hue. */}
      <div className="flex items-center gap-4">
        <span className="flex items-center gap-1.5 text-[10px] text-navy/50">
          <svg width="14" height="4" aria-hidden="true">
            <line x1="0" y1="2" x2="14" y2="2" stroke={SERIES_COLORS[1]} strokeWidth="2" />
          </svg>
          Observed
        </span>
        <span className="flex items-center gap-1.5 text-[10px] text-navy/50">
          <svg width="14" height="4" aria-hidden="true">
            <line x1="0" y1="2" x2="14" y2="2" stroke={SERIES_COLORS[0]} strokeWidth="2" />
          </svg>
          Trend
        </span>
      </div>

      <DecompositionPanel size={size} label={yLabel ? `Observed & trend — ${yLabel}` : 'Observed & trend'}>
        {pathFor(observed, topMin, topRange).map((d, i) => (
          <path key={`o${i}`} d={d} fill="none" stroke={SERIES_COLORS[1]} strokeWidth="2"
                vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        ))}
        {pathFor(trend, topMin, topRange).map((d, i) => (
          <path key={`t${i}`} d={d} fill="none" stroke={SERIES_COLORS[0]} strokeWidth="2"
                vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        ))}
      </DecompositionPanel>

      <DecompositionPanel size={size} label="Seasonal">
        <line x1="0" y1={zeroY(seasonalExtent.min, seasonalExtent.range)} x2={DECOMP_W}
              y2={zeroY(seasonalExtent.min, seasonalExtent.range)}
              stroke="currentColor" className="text-navy/10" strokeWidth="1"
              vectorEffect="non-scaling-stroke" />
        {pathFor(seasonal, seasonalExtent.min, seasonalExtent.range).map((d, i) => (
          <path key={`s${i}`} d={d} fill="none" stroke={SERIES_COLORS[2]} strokeWidth="2"
                vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        ))}
      </DecompositionPanel>

      <DecompositionPanel size={size} label="Residual">
        <line x1="0" y1={zeroY(residualExtent.min, residualExtent.range)} x2={DECOMP_W}
              y2={zeroY(residualExtent.min, residualExtent.range)}
              stroke="currentColor" className="text-navy/10" strokeWidth="1"
              vectorEffect="non-scaling-stroke" />
        {pathFor(residual, residualExtent.min, residualExtent.range).map((d, i) => (
          <path key={`r${i}`} d={d} fill="none" stroke={SERIES_COLORS[4]} strokeWidth="2"
                vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
        ))}
      </DecompositionPanel>

      <div className="flex justify-between text-[10px] text-navy/40 font-mono">
        <span>{points[0].t.slice(0, 10)}</span>
        <span className="text-navy/30">{xLabel}</span>
        <span>{points[points.length - 1].t.slice(0, 10)}</span>
      </div>
      {resampled && <p className="text-[10px] text-navy/35 italic">{resampled}</p>}
    </div>
  );
}
