/**
 * JsonTree — collapsible JSON inspection.
 *
 * Used by PaperView to render LLM trace artifacts (system prompts,
 * user messages, attempts, results, metadata) as a navigable tree
 * instead of a raw <pre>JSON.stringify()</pre> blob.
 *
 * Visual language matches the rest of the design system: mono font,
 * tabular numerics, hairline indent rules, low chroma. No syntax-
 * rainbow — keys are slightly muted, primary color is reserved for
 * primitive values, structural punctuation is muted.
 *
 * Behaviour:
 *   - Objects and arrays are collapsible.
 *   - The first {maxAutoExpandDepth} levels expand by default; the
 *     rest start collapsed. Empty containers don't get a toggle.
 *   - Click a key to copy its dot-path to the clipboard.
 *   - Long string values clamp to a few lines and reveal on click.
 */

import { useCallback, useMemo, useState } from "react";

type Json = unknown;

interface JsonTreeProps {
  value: Json;
  /** Optional root key shown above the tree (e.g. "system_prompt"). */
  rootKey?: string;
  /** Levels expanded by default. Deeper nodes start collapsed. */
  maxAutoExpandDepth?: number;
  /** Callback when a key path is copied. Defaults to clipboard write. */
  onCopyPath?: (path: string) => void;
}

export default function JsonTree({
  value,
  rootKey,
  maxAutoExpandDepth = 2,
  onCopyPath,
}: JsonTreeProps): JSX.Element {
  const handleCopy = useCallback(
    (path: string) => {
      if (onCopyPath) {
        onCopyPath(path);
        return;
      }
      if (typeof navigator !== "undefined" && navigator.clipboard) {
        void navigator.clipboard.writeText(path);
      }
    },
    [onCopyPath],
  );

  return (
    <div className="json-tree" role="tree">
      <Node
        nodeKey={rootKey ?? null}
        value={value}
        depth={0}
        maxAutoExpandDepth={maxAutoExpandDepth}
        path={rootKey ?? ""}
        onCopyPath={handleCopy}
        isLast
      />
    </div>
  );
}

interface NodeProps {
  nodeKey: string | number | null;
  value: Json;
  depth: number;
  maxAutoExpandDepth: number;
  path: string;
  onCopyPath: (path: string) => void;
  isLast: boolean;
}

function Node({
  nodeKey,
  value,
  depth,
  maxAutoExpandDepth,
  path,
  onCopyPath,
  isLast,
}: NodeProps): JSX.Element {
  const kind = classify(value);
  const initiallyOpen = depth < maxAutoExpandDepth;
  const [open, setOpen] = useState(initiallyOpen);

  const isContainer = kind === "object" || kind === "array";
  const entries = useMemo(() => containerEntries(value, kind), [value, kind]);
  const empty = isContainer && entries.length === 0;

  const keyLabel = nodeKey === null ? null : String(nodeKey);

  return (
    <div className="json-node" data-depth={depth}>
      <div className="json-row">
        {isContainer && !empty ? (
          <button
            className="json-toggle"
            type="button"
            onClick={() => setOpen((prev) => !prev)}
            aria-expanded={open}
            aria-label={open ? "Collapse" : "Expand"}
          >
            {open ? "−" : "+"}
          </button>
        ) : (
          <span className="json-toggle json-toggle--empty" aria-hidden="true" />
        )}

        {keyLabel !== null && (
          <button
            className="json-key"
            type="button"
            onClick={() => onCopyPath(path)}
            title={`Copy path: ${path}`}
          >
            {keyLabel}
          </button>
        )}

        {keyLabel !== null && <span className="json-sep">:</span>}

        {isContainer ? (
          <ContainerSummary kind={kind} entries={entries} open={open} />
        ) : (
          <Primitive value={value} kind={kind} />
        )}

        {!isLast && <span className="json-sep">,</span>}
      </div>

      {isContainer && open && !empty && (
        <div className="json-children">
          {entries.map((entry, index) => (
            <Node
              key={`${entry.key}-${index}`}
              nodeKey={entry.key}
              value={entry.value}
              depth={depth + 1}
              maxAutoExpandDepth={maxAutoExpandDepth}
              path={joinPath(path, entry.key, kind === "array")}
              onCopyPath={onCopyPath}
              isLast={index === entries.length - 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

interface ContainerSummaryProps {
  kind: "object" | "array";
  entries: Array<{ key: string | number; value: Json }>;
  open: boolean;
}

function ContainerSummary({ kind, entries, open }: ContainerSummaryProps): JSX.Element {
  const open_brace = kind === "array" ? "[" : "{";
  const close_brace = kind === "array" ? "]" : "}";
  const noun = kind === "array" ? "item" : "key";
  const count = entries.length;

  if (count === 0) {
    return (
      <span className="json-empty">
        {open_brace}
        {close_brace}
      </span>
    );
  }

  if (open) {
    return <span className="json-brace">{open_brace}</span>;
  }

  return (
    <span className="json-collapsed">
      <span className="json-brace">{open_brace}</span>
      <span className="json-count">
        {count} {noun}
        {count === 1 ? "" : "s"}
      </span>
      <span className="json-brace">{close_brace}</span>
    </span>
  );
}

interface PrimitiveProps {
  value: Json;
  kind: PrimitiveKind;
}

function Primitive({ value, kind }: PrimitiveProps): JSX.Element {
  if (kind === "null") {
    return <span className="json-null">null</span>;
  }
  if (kind === "undefined") {
    return <span className="json-null">undefined</span>;
  }
  if (kind === "boolean") {
    return <span className="json-bool">{String(value)}</span>;
  }
  if (kind === "number") {
    return <span className="json-num numeric">{formatNumber(value as number)}</span>;
  }
  if (kind === "string") {
    return <StringValue value={value as string} />;
  }
  return <span className="json-num">{String(value)}</span>;
}

const STRING_PREVIEW_CHARS = 140;

function StringValue({ value }: { value: string }): JSX.Element {
  const [expanded, setExpanded] = useState(false);
  const isLong = value.length > STRING_PREVIEW_CHARS;
  const looksMultiline = value.includes("\n");
  const visible = expanded || !isLong ? value : `${value.slice(0, STRING_PREVIEW_CHARS)}…`;

  if (!isLong && !looksMultiline) {
    return <span className="json-str">{value}</span>;
  }

  return (
    <button
      type="button"
      className={`json-str json-str--block ${expanded ? "is-open" : ""}`}
      onClick={() => setExpanded((prev) => !prev)}
      title={expanded ? "Collapse" : "Expand full string"}
    >
      {visible}
    </button>
  );
}

/* ----------------------------- helpers ----------------------------- */

type PrimitiveKind =
  | "string"
  | "number"
  | "boolean"
  | "null"
  | "undefined"
  | "object"
  | "array";

function classify(value: Json): PrimitiveKind {
  if (value === null) return "null";
  if (value === undefined) return "undefined";
  if (Array.isArray(value)) return "array";
  if (typeof value === "object") return "object";
  if (typeof value === "string") return "string";
  if (typeof value === "number") return "number";
  if (typeof value === "boolean") return "boolean";
  return "string";
}

function containerEntries(
  value: Json,
  kind: PrimitiveKind,
): Array<{ key: string | number; value: Json }> {
  if (kind === "array") {
    return (value as Json[]).map((item, index) => ({ key: index, value: item }));
  }
  if (kind === "object") {
    return Object.entries(value as Record<string, Json>).map(([key, val]) => ({
      key,
      value: val,
    }));
  }
  return [];
}

function joinPath(parent: string, key: string | number, isArrayParent: boolean): string {
  if (parent === "") {
    return isArrayParent ? `[${key}]` : String(key);
  }
  return isArrayParent ? `${parent}[${key}]` : `${parent}.${key}`;
}

function formatNumber(value: number): string {
  if (!Number.isFinite(value)) {
    return String(value);
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString("en-US");
  }
  // Up to 4 significant decimal places, no trailing zeros
  return value
    .toLocaleString("en-US", {
      maximumFractionDigits: 4,
      useGrouping: false,
    })
    .replace(/\.?0+$/, (match) => (match.startsWith(".") ? "" : match));
}
