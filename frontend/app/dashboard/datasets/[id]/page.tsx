'use client';

import Link from 'next/link';
import { useParams, useRouter } from 'next/navigation';
import { useEffect, useMemo, useState } from 'react';
import {
  applyTransform,
  askInEnglish,
  fetchDatasetDetail,
  fetchDatasetPreview,
  fetchDatasetVersions,
  runQuery,
  saveQuery,
  type DatasetDetail,
  type DatasetPreview,
  type DatasetSummary,
  type TransformOp,
} from '../../../lib/api';

type Tab = 'overview' | 'spreadsheet' | 'clean' | 'query';

const TABS: { id: Tab; label: string }[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'spreadsheet', label: 'Spreadsheet' },
  { id: 'clean', label: 'Clean' },
  { id: 'query', label: 'Query' },
];

const OP_TYPES = [
  { value: 'drop_columns', label: 'Drop column' },
  { value: 'rename_columns', label: 'Rename column' },
  { value: 'filter_rows', label: 'Filter rows' },
  { value: 'fill_missing', label: 'Fill missing values' },
  { value: 'drop_missing', label: 'Drop rows with missing values' },
  { value: 'dedupe', label: 'Remove duplicate rows' },
  { value: 'cast_dtype', label: 'Change column type' },
] as const;

const FILTER_OPS = ['eq', 'neq', 'gt', 'gte', 'lt', 'lte', 'contains', 'is_null', 'not_null'] as const;
const FILL_STRATEGIES = ['mean', 'median', 'mode', 'constant', 'ffill', 'bfill'] as const;
const CAST_DTYPES = ['int64', 'float64', 'string', 'bool', 'datetime64[ns]', 'category'] as const;

interface StagedOp {
  key: string;
  type: (typeof OP_TYPES)[number]['value'];
  params: Record<string, unknown>;
}

function newStagedOp(type: StagedOp['type']): StagedOp {
  const defaults: Record<StagedOp['type'], Record<string, unknown>> = {
    drop_columns: { columns: [] },
    rename_columns: { old: '', new: '' },
    filter_rows: { column: '', op: 'eq', value: '' },
    fill_missing: { column: '', strategy: 'mean', value: '' },
    drop_missing: { columns: [], how: 'any' },
    dedupe: { subset: [], keep: 'first' },
    cast_dtype: { column: '', dtype: 'string' },
  };
  return { key: crypto.randomUUID(), type, params: defaults[type] };
}

function stagedOpToRequest(op: StagedOp): TransformOp {
  switch (op.type) {
    case 'drop_columns':
      return { type: 'drop_columns', columns: op.params.columns as string[] };
    case 'rename_columns':
      return { type: 'rename_columns', mapping: { [op.params.old as string]: op.params.new as string } };
    case 'filter_rows':
      return {
        type: 'filter_rows',
        column: op.params.column,
        op: op.params.op,
        value: op.params.op === 'is_null' || op.params.op === 'not_null' ? undefined : op.params.value,
      };
    case 'fill_missing':
      return {
        type: 'fill_missing',
        column: op.params.column,
        strategy: op.params.strategy,
        value: op.params.strategy === 'constant' ? op.params.value : undefined,
      };
    case 'drop_missing': {
      const cols = op.params.columns as string[];
      return { type: 'drop_missing', columns: cols.length ? cols : null, how: op.params.how };
    }
    case 'dedupe': {
      const subset = op.params.subset as string[];
      return { type: 'dedupe', subset: subset.length ? subset : null, keep: op.params.keep };
    }
    case 'cast_dtype':
      return { type: 'cast_dtype', column: op.params.column, dtype: op.params.dtype };
  }
}

function opSummary(op: StagedOp): string {
  switch (op.type) {
    case 'drop_columns':
      return `Drop ${(op.params.columns as string[]).length || 0} column(s)`;
    case 'rename_columns':
      return `Rename '${op.params.old}' → '${op.params.new}'`;
    case 'filter_rows':
      return `Filter where '${op.params.column}' ${op.params.op} ${op.params.value ?? ''}`;
    case 'fill_missing':
      return `Fill missing in '${op.params.column}' with ${op.params.strategy}`;
    case 'drop_missing':
      return `Drop rows missing values (${op.params.how})`;
    case 'dedupe':
      return `Remove duplicate rows (keep ${op.params.keep})`;
    case 'cast_dtype':
      return `Cast '${op.params.column}' to ${op.params.dtype}`;
  }
}

function fieldClass() {
  return 'bg-cream/50 border border-dusty-rose/20 rounded-xl px-3 py-2 text-sm text-navy focus:outline-none focus:ring-2 focus:ring-lavender';
}

function PreviewTable({ preview }: { preview: DatasetPreview }) {
  return (
    <div className="w-full overflow-x-auto rounded-2xl border border-dusty-rose/15">
      <table className="text-xs min-w-full">
        <thead>
          <tr className="bg-cream-dark/40">
            {preview.columns.map((c, i) => (
              <th key={c} className="text-left font-semibold text-navy px-3 py-2 whitespace-nowrap">
                {c}
                <span className="block text-[10px] font-normal text-navy/40">{preview.dtypes[i]}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {preview.rows.map((row, r) => (
            <tr key={r} className="border-t border-dusty-rose/10">
              {row.map((cell, c) => (
                <td key={c} className="px-3 py-2 text-navy/70 whitespace-nowrap">
                  {cell === null ? <span className="text-navy/25 italic">null</span> : String(cell)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {preview.rows.length === 0 && (
        <p className="text-navy/40 text-sm p-6 text-center">No rows.</p>
      )}
    </div>
  );
}

export default function DatasetWorkbenchPage() {
  const params = useParams();
  const router = useRouter();
  const datasetId = params.id as string;

  const [tab, setTab] = useState<Tab>('overview');
  const [detail, setDetail] = useState<DatasetDetail | null>(null);
  const [versions, setVersions] = useState<DatasetSummary[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string[] | null>(null);

  useEffect(() => {
    setIsLoading(true);
    setError(null);
    setSuccessMessage(null);
    fetchDatasetDetail(datasetId)
      .then((d) => {
        setDetail(d);
        return fetchDatasetVersions(d.root_id);
      })
      .then(setVersions)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load dataset'))
      .finally(() => setIsLoading(false));
  }, [datasetId]);

  if (isLoading) {
    return <p className="text-navy/40 font-light px-2">Loading…</p>;
  }

  if (error && !detail) {
    return (
      <div className="max-w-3xl mx-auto">
        <div className="bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-5 text-dusty-rose text-sm">
          {error}
        </div>
        <Link href="/dashboard/datasets" className="text-navy/50 text-sm mt-4 inline-block hover:text-navy">
          ← Back to Datasets
        </Link>
      </div>
    );
  }

  if (!detail) return null;

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header */}
      <div className="mb-8 animate-fade-in">
        <Link href="/dashboard/datasets" className="text-navy/40 text-xs font-medium hover:text-navy transition-colors">
          ← Back to Datasets
        </Link>
        <div className="flex items-center gap-3 mt-3">
          <h1 className="font-[family-name:var(--font-serif)] text-3xl font-bold text-navy truncate">
            {detail.filename}
          </h1>
          <span className="bg-lavender-light/50 text-navy text-xs font-bold px-3 py-1 rounded-full shrink-0">
            v{detail.version}
          </span>
        </div>
        <p className="text-navy/50 font-light mt-1">
          {detail.row_count?.toLocaleString() ?? '?'} rows · {detail.column_count ?? '?'} columns
          {detail.transform_type && (
            <span className="text-navy/40"> · derived via {detail.transform_type === 'clean' ? 'cleaning ops' : 'saved query'}</span>
          )}
        </p>
      </div>

      {/* Version lineage */}
      {versions.length > 1 && (
        <div className="flex items-center gap-2 mb-6 flex-wrap animate-fade-in">
          {versions.map((v) => (
            <Link
              key={v.id}
              href={`/dashboard/datasets/${v.id}`}
              className={`text-xs font-semibold px-3 py-1.5 rounded-full transition-colors ${
                v.id === detail.id
                  ? 'bg-navy text-cream'
                  : 'bg-cream-dark/50 text-navy/50 hover:text-navy hover:bg-cream-dark'
              }`}
            >
              v{v.version}
            </Link>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex items-center gap-2 mb-6 border-b border-dusty-rose/15">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`text-sm font-semibold px-4 py-3 border-b-2 -mb-px transition-colors ${
              tab === t.id ? 'text-navy border-navy' : 'text-navy/40 border-transparent hover:text-navy/70'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="mb-6 bg-dusty-rose/10 border border-dusty-rose/30 rounded-2xl p-4 text-dusty-rose text-sm">
          {error}
        </div>
      )}
      {successMessage && (
        <div className="mb-6 bg-lavender-light/30 border border-lavender/40 rounded-2xl p-4 text-navy text-sm space-y-1">
          {successMessage.map((m, i) => (
            <p key={i}>✓ {m}</p>
          ))}
        </div>
      )}

      {tab === 'overview' && <OverviewTab detail={detail} />}
      {tab === 'spreadsheet' && (
        <SpreadsheetTab
          // Keyed on the version, so pending edits and the current page reset
          // with it. Edits are only meaningful against the rows they were made
          // against, and saving produces a *new* version — carrying them over
          // would re-apply them to different data.
          key={detail.id}
          datasetId={detail.id}
          onSaved={(newId) => router.push(`/dashboard/datasets/${newId}`)}
          setError={setError}
        />
      )}
      {tab === 'clean' && (
        <CleanTab
          datasetId={detail.id}
          columns={detail.column_summary.map((c) => c.name)}
          onApplied={(newId) => router.push(`/dashboard/datasets/${newId}`)}
          setError={setError}
        />
      )}
      {tab === 'query' && (
        <QueryTab
          datasetId={detail.id}
          onSaved={(newId) => router.push(`/dashboard/datasets/${newId}`)}
          setError={setError}
        />
      )}
    </div>
  );
}

function OverviewTab({ detail }: { detail: DatasetDetail }) {
  return (
    <div className="animate-fade-in">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {detail.column_summary.map((col) => (
          <div key={col.name} className="bg-warm-white/80 border border-dusty-rose/20 rounded-2xl p-5">
            <p className="font-bold text-navy text-sm truncate">{col.name}</p>
            <p className="text-[10px] text-navy/40 uppercase tracking-wide mt-0.5">{col.dtype}</p>
            <p className="text-xs text-navy/50 mt-2">{col.missing_count} missing</p>
            {col.stats && (
              <div className="text-xs text-navy/40 mt-1 space-y-0.5">
                {col.stats.mean !== null && <p>mean: {col.stats.mean.toFixed(2)}</p>}
                {col.stats.min !== null && col.stats.max !== null && (
                  <p>range: {col.stats.min} – {col.stats.max}</p>
                )}
                {col.stats.unique_count !== null && <p>{col.stats.unique_count} unique</p>}
              </div>
            )}
          </div>
        ))}
      </div>
      {detail.column_summary.length === 0 && (
        <p className="text-navy/40 text-sm">No column profile available.</p>
      )}
    </div>
  );
}

function SpreadsheetTab({
  datasetId,
  onSaved,
  setError,
}: {
  datasetId: string;
  onSaved: (newDatasetId: string) => void;
  setError: (e: string | null) => void;
}) {
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [offset, setOffset] = useState(0);
  const limit = 25;
  // Keyed by **absolute** row index, not by position on the current page.
  // Page-relative keys meant an edit belonged to a screen position rather than
  // to a row: edit row 3 of page 1, page forward, and the edit both appeared
  // on page 2's row 3 and — because the save added the *current* offset —
  // saved itself onto row 28.
  const [edits, setEdits] = useState<Record<number, Record<string, string>>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [isLoadingPage, setIsLoadingPage] = useState(true);

  useEffect(() => {
    setIsLoadingPage(true);
    fetchDatasetPreview(datasetId, offset, limit)
      .then(setPreview)
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load preview'))
      .finally(() => setIsLoadingPage(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId, offset]);

  const dirtyCount = useMemo(
    () => Object.values(edits).reduce((n, cols) => n + Object.keys(cols).length, 0),
    [edits],
  );

  function handleCellChange(rowIndex: number, column: string, value: string) {
    setEdits((prev) => ({ ...prev, [rowIndex]: { ...prev[rowIndex], [column]: value } }));
  }

  async function handleSave() {
    if (!preview) return;
    setIsSaving(true);
    setError(null);
    try {
      // Already absolute — every page's edits travel together, each against
      // the row it was actually made on.
      const editList = Object.entries(edits).flatMap(([rowIndex, cols]) =>
        Object.entries(cols).map(([column, value]) => ({
          row_index: Number(rowIndex),
          column,
          value,
        })),
      );
      const result = await applyTransform(datasetId, [{ type: 'edit_cells', edits: editList }]);
      onSaved(result.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save changes');
    } finally {
      setIsSaving(false);
    }
  }

  if (isLoadingPage && !preview) {
    return <p className="text-navy/40 font-light text-sm">Loading…</p>;
  }
  if (!preview) return null;

  return (
    <div className="animate-fade-in">
      <div className="w-full overflow-x-auto rounded-2xl border border-dusty-rose/15">
        <table className="text-xs min-w-full">
          <thead>
            <tr className="bg-cream-dark/40">
              {preview.columns.map((c, i) => (
                <th key={c} className="text-left font-semibold text-navy px-3 py-2 whitespace-nowrap">
                  {c}
                  <span className="block text-[10px] font-normal text-navy/40">{preview.dtypes[i]}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {preview.rows.map((row, r) => (
              <tr key={offset + r} className="border-t border-dusty-rose/10">
                {row.map((cell, c) => {
                  const column = preview.columns[c];
                  const rowIndex = offset + r;
                  const edited = edits[rowIndex]?.[column];
                  return (
                    <td key={c} className="p-0">
                      {/* Controlled, not `defaultValue`. React assigns
                          `element.value` when it mounts an input, which sets
                          the DOM's dirty-value flag; from then on the value
                          attribute that `defaultValue` writes is ignored, so
                          paging re-rendered the row and left the previous
                          page's numbers on screen. */}
                      <input
                        value={edited ?? (cell === null ? '' : String(cell))}
                        placeholder={cell === null ? 'null' : ''}
                        onChange={(e) => handleCellChange(rowIndex, column, e.target.value)}
                        className={`w-full px-3 py-2 text-navy/80 bg-transparent focus:outline-none focus:bg-lavender-light/20 ${
                          edited !== undefined ? 'bg-peach-light/20' : ''
                        }`}
                      />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="flex items-center justify-between mt-4">
        <div className="flex items-center gap-2">
          <button
            disabled={offset === 0}
            onClick={() => setOffset(Math.max(0, offset - limit))}
            className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-3 py-2 rounded-xl disabled:opacity-40 transition-colors"
          >
            ← Prev
          </button>
          <span className="text-xs text-navy/40">
            {offset + 1}–{Math.min(offset + limit, preview.total_rows)} of {preview.total_rows}
          </span>
          <button
            disabled={offset + limit >= preview.total_rows}
            onClick={() => setOffset(offset + limit)}
            className="text-xs font-medium text-navy/50 hover:text-navy bg-cream-dark/50 px-3 py-2 rounded-xl disabled:opacity-40 transition-colors"
          >
            Next →
          </button>
        </div>
        <button
          disabled={dirtyCount === 0 || isSaving}
          onClick={handleSave}
          className="bg-navy text-cream font-medium px-6 py-2.5 rounded-xl hover:bg-navy-light transition-all disabled:opacity-40 text-sm"
        >
          {isSaving ? 'Saving…' : `Save Changes${dirtyCount ? ` (${dirtyCount})` : ''}`}
        </button>
      </div>
    </div>
  );
}

function CleanTab({
  datasetId,
  columns,
  onApplied,
  setError,
}: {
  datasetId: string;
  columns: string[];
  onApplied: (newDatasetId: string) => void;
  setError: (e: string | null) => void;
}) {
  const [staged, setStaged] = useState<StagedOp[]>([]);
  const [isApplying, setIsApplying] = useState(false);

  function addOp(type: StagedOp['type']) {
    setStaged((prev) => [...prev, newStagedOp(type)]);
  }

  function updateOp(key: string, params: Record<string, unknown>) {
    setStaged((prev) => prev.map((op) => (op.key === key ? { ...op, params: { ...op.params, ...params } } : op)));
  }

  function removeOp(key: string) {
    setStaged((prev) => prev.filter((op) => op.key !== key));
  }

  async function handleApply() {
    if (staged.length === 0) return;
    setIsApplying(true);
    setError(null);
    try {
      const ops = staged.map(stagedOpToRequest);
      const result = await applyTransform(datasetId, ops);
      onApplied(result.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to apply transform');
    } finally {
      setIsApplying(false);
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="flex flex-wrap gap-2 mb-6">
        {OP_TYPES.map((t) => (
          <button
            key={t.value}
            onClick={() => addOp(t.value)}
            className="text-xs font-medium text-navy/60 hover:text-navy bg-cream-dark/50 hover:bg-cream-dark px-3 py-2 rounded-xl transition-colors"
          >
            + {t.label}
          </button>
        ))}
      </div>

      {staged.length === 0 ? (
        <p className="text-navy/40 text-sm px-2">No operations staged yet — add one above.</p>
      ) : (
        <div className="space-y-3 mb-6">
          {staged.map((op, i) => (
            <div key={op.key} className="bg-warm-white/80 border border-dusty-rose/20 rounded-2xl p-5">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold text-navy/40 uppercase tracking-wide">
                  {i + 1}. {OP_TYPES.find((t) => t.value === op.type)?.label}
                </span>
                <button onClick={() => removeOp(op.key)} className="text-xs text-dusty-rose hover:text-red-500">
                  Remove
                </button>
              </div>
              <OpForm op={op} columns={columns} onChange={(params) => updateOp(op.key, params)} />
              <p className="text-[11px] text-navy/40 mt-3 italic">{opSummary(op)}</p>
            </div>
          ))}
        </div>
      )}

      <button
        disabled={staged.length === 0 || isApplying}
        onClick={handleApply}
        className="bg-navy text-cream font-medium px-6 py-2.5 rounded-xl hover:bg-navy-light transition-all disabled:opacity-40 text-sm"
      >
        {isApplying ? 'Applying…' : `Apply ${staged.length || ''} Operation${staged.length === 1 ? '' : 's'}`}
      </button>
    </div>
  );
}

function OpForm({
  op,
  columns,
  onChange,
}: {
  op: StagedOp;
  columns: string[];
  onChange: (params: Record<string, unknown>) => void;
}) {
  const cls = fieldClass();
  switch (op.type) {
    case 'drop_columns':
      return (
        <select
          multiple
          className={`${cls} w-full h-24`}
          value={op.params.columns as string[]}
          onChange={(e) => onChange({ columns: Array.from(e.target.selectedOptions, (o) => o.value) })}
        >
          {columns.map((c) => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      );
    case 'rename_columns':
      return (
        <div className="flex items-center gap-3">
          <select className={cls} value={op.params.old as string} onChange={(e) => onChange({ old: e.target.value })}>
            <option value="">Select column…</option>
            {columns.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <span className="text-navy/40">→</span>
          <input
            className={cls}
            placeholder="new name"
            value={op.params.new as string}
            onChange={(e) => onChange({ new: e.target.value })}
          />
        </div>
      );
    case 'filter_rows':
      return (
        <div className="flex flex-wrap items-center gap-3">
          <select className={cls} value={op.params.column as string} onChange={(e) => onChange({ column: e.target.value })}>
            <option value="">Column…</option>
            {columns.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select className={cls} value={op.params.op as string} onChange={(e) => onChange({ op: e.target.value })}>
            {FILTER_OPS.map((o) => (
              <option key={o} value={o}>{o}</option>
            ))}
          </select>
          {op.params.op !== 'is_null' && op.params.op !== 'not_null' && (
            <input
              className={cls}
              placeholder="value"
              value={op.params.value as string}
              onChange={(e) => onChange({ value: e.target.value })}
            />
          )}
        </div>
      );
    case 'fill_missing':
      return (
        <div className="flex flex-wrap items-center gap-3">
          <select className={cls} value={op.params.column as string} onChange={(e) => onChange({ column: e.target.value })}>
            <option value="">Column…</option>
            {columns.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select className={cls} value={op.params.strategy as string} onChange={(e) => onChange({ strategy: e.target.value })}>
            {FILL_STRATEGIES.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          {op.params.strategy === 'constant' && (
            <input
              className={cls}
              placeholder="value"
              value={op.params.value as string}
              onChange={(e) => onChange({ value: e.target.value })}
            />
          )}
        </div>
      );
    case 'drop_missing':
      return (
        <div className="flex flex-wrap items-center gap-3">
          <select
            multiple
            className={`${cls} h-20`}
            value={op.params.columns as string[]}
            onChange={(e) => onChange({ columns: Array.from(e.target.selectedOptions, (o) => o.value) })}
          >
            {columns.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select className={cls} value={op.params.how as string} onChange={(e) => onChange({ how: e.target.value })}>
            <option value="any">any column missing</option>
            <option value="all">all columns missing</option>
          </select>
          <span className="text-[11px] text-navy/40">(leave columns unselected for "all columns")</span>
        </div>
      );
    case 'dedupe':
      return (
        <div className="flex flex-wrap items-center gap-3">
          <select
            multiple
            className={`${cls} h-20`}
            value={op.params.subset as string[]}
            onChange={(e) => onChange({ subset: Array.from(e.target.selectedOptions, (o) => o.value) })}
          >
            {columns.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select className={cls} value={op.params.keep as string} onChange={(e) => onChange({ keep: e.target.value })}>
            <option value="first">keep first</option>
            <option value="last">keep last</option>
          </select>
          <span className="text-[11px] text-navy/40">(leave unselected to consider all columns)</span>
        </div>
      );
    case 'cast_dtype':
      return (
        <div className="flex flex-wrap items-center gap-3">
          <select className={cls} value={op.params.column as string} onChange={(e) => onChange({ column: e.target.value })}>
            <option value="">Column…</option>
            {columns.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <select className={cls} value={op.params.dtype as string} onChange={(e) => onChange({ dtype: e.target.value })}>
            {CAST_DTYPES.map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </div>
      );
  }
}

function QueryTab({
  datasetId,
  onSaved,
  setError,
}: {
  datasetId: string;
  onSaved: (newDatasetId: string) => void;
  setError: (e: string | null) => void;
}) {
  const [sql, setSql] = useState('SELECT * FROM df LIMIT 100');
  const [result, setResult] = useState<DatasetPreview | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [lastRunSql, setLastRunSql] = useState<string | null>(null);
  const [question, setQuestion] = useState('');
  const [isTranslating, setIsTranslating] = useState(false);

  async function handleRun() {
    setIsRunning(true);
    setError(null);
    setResult(null);
    try {
      const preview = await runQuery(datasetId, sql);
      setResult(preview);
      setLastRunSql(sql);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed');
    } finally {
      setIsRunning(false);
    }
  }

  async function handleTranslate() {
    if (!question.trim()) return;
    setIsTranslating(true);
    setError(null);
    setResult(null);
    try {
      // The LLM only ever produces SQL text — it's shown here and re-runs
      // through the exact same validated/sandboxed path as hand-typed SQL,
      // so the rest of this tab (Run, Save as new version) needs no changes.
      const { sql: translatedSql, preview } = await askInEnglish(datasetId, question);
      setSql(translatedSql);
      setResult(preview);
      setLastRunSql(translatedSql);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not translate that into SQL');
    } finally {
      setIsTranslating(false);
    }
  }

  async function handleSaveAsVersion() {
    setIsSaving(true);
    setError(null);
    try {
      const dataset = await saveQuery(datasetId, sql);
      onSaved(dataset.id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save query result');
    } finally {
      setIsSaving(false);
    }
  }

  return (
    <div className="animate-fade-in">
      <div className="flex items-center gap-3 mb-3">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); handleTranslate(); } }}
          placeholder="Ask in plain English — e.g. 'top 5 regions by revenue'…"
          disabled={isTranslating}
          className="flex-1 bg-warm-white/80 border border-dusty-rose/20 rounded-xl px-4 py-2.5 text-sm text-navy placeholder:text-navy/30 focus:outline-none focus:ring-2 focus:ring-lavender disabled:opacity-60"
        />
        <button
          disabled={isTranslating || !question.trim()}
          onClick={handleTranslate}
          className="text-navy font-medium px-5 py-2.5 rounded-xl bg-peach-light/50 hover:bg-peach-light transition-all disabled:opacity-40 text-sm whitespace-nowrap"
        >
          {isTranslating ? 'Translating…' : 'Translate & Run'}
        </button>
      </div>
      <p className="text-[11px] text-navy/40 mb-4">
        Generates SQL with Gemini, shown below — always visible and editable before it runs again.
      </p>

      <textarea
        value={sql}
        onChange={(e) => setSql(e.target.value)}
        rows={5}
        spellCheck={false}
        className="w-full bg-navy text-cream font-mono text-sm rounded-2xl p-5 focus:outline-none focus:ring-2 focus:ring-lavender resize-y"
      />
      <p className="text-[11px] text-navy/40 mt-2 mb-4">
        Query the table as <code className="bg-cream-dark/60 px-1.5 py-0.5 rounded">df</code>. Read-only — only a single SELECT/WITH statement is allowed.
      </p>

      <div className="flex items-center gap-3 mb-6">
        <button
          disabled={isRunning || !sql.trim()}
          onClick={handleRun}
          className="bg-navy text-cream font-medium px-6 py-2.5 rounded-xl hover:bg-navy-light transition-all disabled:opacity-40 text-sm"
        >
          {isRunning ? 'Running…' : 'Run'}
        </button>
        {result && lastRunSql === sql && (
          <button
            disabled={isSaving}
            onClick={handleSaveAsVersion}
            className="text-navy font-medium px-6 py-2.5 rounded-xl bg-lavender-light/50 hover:bg-lavender-light transition-all disabled:opacity-40 text-sm"
          >
            {isSaving ? 'Saving…' : 'Save as new version'}
          </button>
        )}
      </div>

      {result && (
        <>
          <p className="text-xs text-navy/40 mb-3">{result.total_rows} row(s)</p>
          <PreviewTable preview={result} />
        </>
      )}
    </div>
  );
}
