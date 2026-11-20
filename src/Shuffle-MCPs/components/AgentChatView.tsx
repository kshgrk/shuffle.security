/**
 * AgentChatView — the "Chat" tab of AgentUI.
 *
 * Shows an agent run as a conversation: the user's messages, the steps the
 * agent took (collapsed once done), its questions with one-click replies, and
 * its answers. It renders the run AgentUI already polls and uses AgentUI's own
 * actions to start runs, answer questions and continue finished runs, so the
 * Start / Simple / Detailed tabs see exactly the same run.
 *
 * Workflow edits are announced with AGENT_WORKFLOW_CHANGED_EVENT and node
 * links with AGENT_FOCUS_NODE_EVENT, so a host that shows the workflow can
 * reload and highlight it. Workflow runs the agent starts are shown from the
 * execution itself, not the agent's words, and announced with
 * AGENT_OPEN_EXECUTION_EVENT so the host can open its own run view. Without a
 * listener the chat still works fully.
 */
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Box, Button, ButtonBase, CircularProgress, Collapse, IconButton, InputBase, MenuItem, Popover, Select, Tooltip, Typography } from '@mui/material';
import { Bookmark, History, MessageCirclePlus, Undo2 } from 'lucide-react';
import ShuffleMarkdown from '@/Shuffle-Core/components/Markdown';
import { getRunTitle, getTimeAgo } from '@/Shuffle-MCPs/components/AgentActivityList';
import { searchAgentActivity, type AgentRun } from '@/Shuffle-MCPs/agentActivity';
import { toast } from '@/Shuffle-Core/toast';
import {
  AGENT_FOCUS_NODE_EVENT,
  AGENT_OPEN_EXECUTION_EVENT,
  AGENT_WORKFLOW_CHANGED_EVENT,
  ASK_FIRST_PREFIX,
  asksForApproval,
  buildChatThread,
  describeStep,
  formatDuration,
  getAnswersFromModelReplies,
  getAppsNeedingAuth,
  getCalloutTone,
  getChangedNodeIds,
  getChangesAfter,
  getChatResources,
  getRunResources,
  getDecisionOutput,
  getFailureReason,
  getNodeLabels,
  getField,
  getQuickReplies,
  getResult,
  getRunIdleSeconds,
  getRunNodes,
  getStatusTone,
  getStepDuration,
  getStepStatus,
  getWorkflowProblems,
  getWorkflowRuns,
  humanize,
  isAction,
  isAppliedRevert,
  isAsk,
  isFinish,
  pickRevisionBefore,
  parseJson,
  toSeconds,
  type AgentFocusNodeDetail,
  type AgentOpenExecutionDetail,
  type BackendJson,
  type AgentWorkflowChangedDetail,
  type AgentTone,
  type ChatDecision,
  type ChatThreadItem,
  type ChatWorkflowRun,
  stripModePrefix,
} from '@/Shuffle-MCPs/agentChat';
import type { ChatResource } from '@/Shuffle-MCPs/agentChat';
import { resolveAgentContext, setPageContextChoice } from '@/Shuffle-MCPs/agentContextRegistry';

// Agent replies only, so the shared renderer and its other callers stay unchanged
// --warning isn't defined in every host theme, hence the fallback.
const AGENT_TONES: Record<AgentTone, { color: string; background: string; label: string }> = {
  error: { color: 'hsl(var(--destructive))', background: 'hsl(var(--destructive) / 0.1)', label: 'Error' },
  warning: { color: 'hsl(var(--warning, 38 92% 50%))', background: 'hsl(var(--warning, 38 92% 50%) / 0.12)', label: 'Warning' },
  success: { color: 'hsl(142 71% 45%)', background: 'hsl(142 71% 45% / 0.12)', label: 'Success' },
  note: { color: 'hsl(var(--primary))', background: 'hsl(var(--primary) / 0.08)', label: 'Note' },
};

const isLineBreak = (node: React.ReactNode) => React.isValidElement(node) && node.type === 'br';
const isBlank = (node: React.ReactNode) => typeof node === 'string' && !node.trim();

const AgentCallout = ({ children, node: _node, ...props }: { children?: React.ReactNode; node?: unknown }) => {
  const items = React.Children.toArray(children);
  const firstIndex = items.findIndex((item) => React.isValidElement(item));
  const first = items[firstIndex] as React.ReactElement<{ children?: React.ReactNode }> | undefined;
  const parts = first ? React.Children.toArray(first.props.children) : [];
  const callout = typeof parts[0] === 'string' ? getCalloutTone(parts[0]) : null;
  if (!first || !callout) return <blockquote {...props}>{children}</blockquote>;

  let rest = callout.rest ? [callout.rest, ...parts.slice(1)] : parts.slice(1);
  while (rest.length && (isLineBreak(rest[0]) || isBlank(rest[0]))) rest = rest.slice(1);
  const tone = AGENT_TONES[callout.tone];
  return (
    <Box
      role="note"
      sx={{ my: 1, px: 1.5, py: 1, borderRadius: 1, borderLeft: `3px solid ${tone.color}`, backgroundColor: tone.background, '& > *:last-child': { mb: 0 } }}
    >
      <Box sx={{ color: tone.color, fontWeight: 600, fontSize: '0.85em', mb: 0.25 }}>{tone.label}</Box>
      {rest.length > 0 && React.cloneElement(first, {}, ...rest)}
      {items.slice(firstIndex + 1)}
    </Box>
  );
};

const AgentStatusCode = ({ children, node: _node, ...props }: { children?: React.ReactNode; node?: unknown; className?: string }) => {
  const tone = !props.className && typeof children === 'string' ? getStatusTone(children) : null;
  if (!tone) return <code {...props}>{children}</code>;
  const { color, background } = AGENT_TONES[tone];
  // Inline style, as the renderer's own `& code` rule is more specific than a class
  return <code {...props} style={{ color, backgroundColor: background, fontWeight: 600 }}>{children}</code>;
};

const agentMarkdownComponents = { blockquote: AgentCallout, code: AgentStatusCode };

const STALL_MS = 90 * 1000;
// Space between the inline composer and the bottom of the window
const CHAT_BOTTOM_GAP = 16;
const RUN_POLL_MS = 3000;
const RUN_STUCK_SECONDS = 120;
const AUTH_WATCH_MS = 4000;
const MODE_STORAGE_KEY = 'shuffle-agent-chat-mode';

export interface AgentChatViewProps {
  /** The execution AgentUI is showing ({ execution_id, authorization, status, ... }). */
  execution: { execution_id?: string; authorization?: string; status?: string; workflow?: { id?: string } } | null;
  /** The parsed AI Agent result (decisions, output, original_input, llm_responses, ...). */
  agentData: Record<string, BackendJson>;
  /** True while the run is working, including right after the user replied. */
  isRunning: boolean;
  /** True while a request to start or answer is in flight. */
  busy: boolean;
  /** Live-update warning from AgentUI's poller, if any. */
  pollWarning?: string | null;
  error?: string | null;
  /** The workflow the agent works on, when known. Enables revert and canvas events. */
  workflowId?: string;
  placeholder?: string;
  /** Where to render the composer (AgentUI's pinned footer in sidebar layout). Inline when omitted. */
  composerContainer?: HTMLElement | null;
  /** Extra horizontal space on both sides of the conversation. Defaults to none. */
  contentPaddingX?: number;

  /** Where the bookmark, history and new chat buttons render. Above the chat when omitted. */
  headerActionsContainer?: HTMLElement | null;
  onStart: (text: string) => void | Promise<void>;
  /** Answers a question, approval or continuation, using AgentUI's submitQuestions shape. */
  onSubmit: (decisionId: string, answers: Record<string, BackendJson>, isContinuation?: boolean) => void | Promise<void>;
  onStop: () => void;
  /** True while a Stop request is in flight. */
  stopping?: boolean;
  onRefresh: () => void;
  isAppAuthenticated: (appName: string, appId?: string | null) => boolean;
  /** True while the auth list is loading or failed to load: no app is reported as missing auth then. */
  authUnknown?: boolean;
  /** Opens the credentials flow for an app (AgentUI's app drawer, or the app page). */
  onAuthenticateApp: (app: { name: string; id: string; url?: string }) => void;
  onRefreshAuthenticatedApps: () => void;
  resolveUrl: (path: string) => string;
  resolveHeaders: () => Record<string, string>;
  /** Backend for the past chats list (the Agents page run list). */
  apiBaseUrl?: string;
  /** Opens an earlier agent run in this chat. Past chats are hidden without it. */
  onOpenRun?: (executionId: string, authorization?: string, status?: string) => void;
  /** Clears the current run so the next message starts a new one. */
  onNewChat?: () => void;
  /** Opens a page of the host app (path with search). A full page load when omitted. */
  onOpenPage?: (path: string) => void;
  /** Switches to the Timeline tab and highlights the exact decision (and operation, if any) that produced a change. */
  onFocusDecision?: (target: {
    decisionId?: string;
    decisionIndex?: number;
    operationIndex?: number;
    changeText?: string;
    nodeId?: string;
  }) => void;
}

// Kept in this browser: the backend has no bookmarks yet
const BOOKMARKS_STORAGE_KEY = 'shuffle:agent_chat_bookmarks';
type ChatBookmark = { execution_id: string; authorization?: string; title: string; started_at: string; workflow_id?: string };

const readBookmarks = (): ChatBookmark[] => {
  try {
    const stored = JSON.parse(window.localStorage.getItem(BOOKMARKS_STORAGE_KEY) || '[]');
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
};

const writeBookmarks = (bookmarks: ChatBookmark[]) => {
  try {
    window.localStorage.setItem(BOOKMARKS_STORAGE_KEY, JSON.stringify(bookmarks));
  } catch {
    // Storage blocked: the bookmark lasts until the page reloads
  }
};

const shortTimeAgo = (startedAt: string) => (startedAt ? getTimeAgo(startedAt).replace(' ago', '') : '');

const headerIconButtonSx = {
  color: 'hsl(var(--muted-foreground))',
  p: 0.75,
  borderRadius: '8px',
  '&:hover': { color: 'hsl(var(--foreground))', bgcolor: 'hsl(var(--muted))' },
  '&.Mui-disabled': { color: 'hsl(var(--muted-foreground) / 0.4)' },
};

export const readMode = (): 'auto' | 'ask' => {
  try {
    return window.localStorage.getItem(MODE_STORAGE_KEY) === 'ask' ? 'ask' : 'auto';
  } catch {
    return 'auto';
  }
};

const textSx = { fontSize: '0.8125rem', color: 'hsl(var(--foreground))', lineHeight: 1.55 };
const mutedSx = { fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' };
const linkButtonSx = {
  textTransform: 'none',
  fontSize: '0.75rem',
  color: 'hsl(var(--muted-foreground))',
  p: 0,
  minWidth: 0,
  '&:hover': { color: 'hsl(var(--foreground))', bgcolor: 'transparent' },
} as const;
const primaryActionButtonSx = {
  textTransform: 'none',
  fontSize: '0.75rem',
  minHeight: 0,
  py: 0.25,
  bgcolor: 'hsl(var(--foreground))',
  color: 'hsl(var(--background))',
  boxShadow: 'none',
  '&:hover': {
    bgcolor: 'hsl(var(--foreground) / 0.88)',
    boxShadow: 'none',
  },
  '&.Mui-disabled': {
    bgcolor: 'hsl(var(--muted))',
    color: 'hsl(var(--muted-foreground))',
  },
} as const;
const outlinedActionButtonSx = {
  textTransform: 'none',
  fontSize: '0.75rem',
  minHeight: 0,
  py: 0.25,
  borderColor: 'hsl(var(--border))',
  color: 'hsl(var(--foreground))',
  '&:hover': {
    borderColor: 'hsl(var(--foreground) / 0.4)',
    bgcolor: 'hsl(var(--muted) / 0.5)',
  },
} as const;
const replySx = {
  textTransform: 'none',
  fontSize: '0.75rem',
  borderRadius: 9999,
  px: 1.25,
  py: 0.25,
  minHeight: 0,
  color: 'hsl(var(--foreground))',
  borderColor: 'hsl(var(--border))',
} as const;

// ── Motion ──
// Keyframe names are global, declared once on the chat root. Items fade up once when they
// mount (keys are stable, so updates don't replay it); agent replies reveal top to bottom.
const EASE_OUT = 'cubic-bezier(0.2, 0.8, 0.2, 1)';
const WARNING_COLOR = 'hsl(var(--warning, 38 92% 50%))';
const chatMotionSx = {
  '@keyframes chatItemIn': { from: { opacity: 0, transform: 'translateY(6px)' }, to: { opacity: 1, transform: 'none' } },
  '@keyframes chatReveal': { from: { maskPosition: '0 100%', WebkitMaskPosition: '0 100%' }, to: { maskPosition: '0 0', WebkitMaskPosition: '0 0' } },
  '@keyframes chatShimmer': { from: { backgroundPosition: '100% 0' }, to: { backgroundPosition: '-100% 0' } },
  '@keyframes chatPulse': { '50%': { opacity: 0.3 } },
  '@keyframes chatDraw': { to: { strokeDashoffset: 0 } },
  '@media (prefers-reduced-motion: reduce)': { '& *': { animation: 'none !important', transition: 'none !important' } },
};
const enterSx = { animation: `chatItemIn 260ms ${EASE_OUT} both` };
// The mask is 220% tall with its edge at ~101%, so it starts fully hidden and ends fully shown
const revealSx = {
  ...enterSx,
  maskImage: 'linear-gradient(to bottom, #000 46%, transparent 54%)',
  WebkitMaskImage: 'linear-gradient(to bottom, #000 46%, transparent 54%)',
  maskSize: '100% 220%',
  WebkitMaskSize: '100% 220%',
  animation: `chatItemIn 260ms ${EASE_OUT} both, chatReveal 750ms ease-out both`,
};
const shimmerSx = {
  background: 'linear-gradient(90deg, hsl(var(--muted-foreground) / 0.55) 0%, hsl(var(--foreground)) 50%, hsl(var(--muted-foreground) / 0.55) 100%)',
  backgroundSize: '200% 100%',
  WebkitBackgroundClip: 'text',
  backgroundClip: 'text',
  color: 'transparent',
  animation: 'chatShimmer 1.8s linear infinite',
};

// Errors are shown as one short line; the full text stays under the step's request and response
const shortFailure = (text: string) => {
  const clean = text.replace(/\s*\(available:[\s\S]*$/i, '').replace(/\s+/g, ' ').trim();
  return clean.length > 160 ? `${clean.slice(0, 157)}...` : clean;
};
const stepId = (decision: ChatDecision) => decision.run_details?.id || `${decision.i}`;
const stepKind = (decision: ChatDecision) => `${(decision.tool || '').toLowerCase()}|${(decision.action || '').toLowerCase()}`;

const runNodeLabel: Record<string, string> = { done: 'Done', failed: 'Failed', skipped: 'Skipped', running: 'Running', waiting: 'Waiting', 'not run': 'Not run' };

/** A workflow run the agent started, read from the execution itself. */
const RunCard: React.FC<{
  run: ChatWorkflowRun;
  onOpen?: () => void;
  resolveUrl: (path: string) => string;
  resolveHeaders: () => Record<string, string>;
}> = ({ run, onOpen, resolveUrl, resolveHeaders }) => {
  const [execution, setExecution] = useState<BackendJson>(null);
  const [failedToLoad, setFailedToLoad] = useState(false);
  const [now, setNow] = useState(() => Date.now());
  const runStatus = String(execution?.status || '').toUpperCase();
  const done = ['FINISHED', 'ABORTED', 'FAILURE'].includes(runStatus);

  useEffect(() => {
    if (done) return undefined;
    let cancelled = false;
    let lastText = '';
    const load = async () => {
      try {
        const resp = await fetch(resolveUrl('/api/v1/streams/results'), {
          method: 'POST',
          credentials: 'include',
          headers: { 'Content-Type': 'application/json', ...resolveHeaders() },
          body: JSON.stringify({ execution_id: run.executionId, authorization: run.authorization }),
        });
        if (!resp.ok) throw new Error(`status ${resp.status}`);
        const text = await resp.text();
        if (!cancelled) {
          // An unchanged run keeps the same object, so the card's derived data isn't rebuilt
          if (text !== lastText) setExecution(JSON.parse(text));
          lastText = text;
          setFailedToLoad(false);
          setNow(Date.now());
        }
      } catch {
        if (!cancelled) setFailedToLoad(true);
      }
    };
    load();
    const id = window.setInterval(load, RUN_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [done, run.executionId, run.authorization, resolveUrl, resolveHeaders]);

  const nodes = execution ? getRunNodes(execution) : [];
  const waitingOn = nodes.find((node) => node.status === 'waiting' || node.status === 'running');
  const idle = execution && !done ? getRunIdleSeconds(execution, now) : 0;
  const elapsed = toSeconds(execution?.completed_at || now) - toSeconds(execution?.started_at);
  const heading = !execution ? 'Workflow run'
    : runStatus === 'FINISHED' ? `Workflow run finished${elapsed > 0 ? ` in ${formatDuration(elapsed)}` : ''}`
    : done ? `Workflow run ${runStatus === 'ABORTED' ? 'aborted' : 'failed'}`
    : `Workflow running${elapsed > 0 ? ` for ${formatDuration(elapsed)}` : ''}`;

  return (
    <Box sx={{ my: 1, p: 1.5, border: '1px solid hsl(var(--border))', borderRadius: '12px', bgcolor: 'hsl(var(--card, var(--background)))', ...enterSx }}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <Typography sx={{ ...textSx, flex: 1, fontWeight: 600 }}>{heading}</Typography>
        {!done && execution && <CircularProgress size={11} sx={{ color: 'hsl(var(--muted-foreground))' }} />}
      </Box>
      {!execution && <Typography sx={{ ...mutedSx, mt: 0.5 }}>{failedToLoad ? 'Could not load this run.' : 'Loading the run...'}</Typography>}
      {nodes.map((node) => (
        <Box key={node.id} sx={{ mt: 0.5 }}>
          <Box sx={{ display: 'flex', gap: 1, alignItems: 'baseline' }}>
            <Typography sx={{ ...textSx, flex: 1 }}>{node.label}</Typography>
            <Typography sx={{ ...mutedSx, flexShrink: 0, color: node.status === 'failed' ? 'hsl(var(--destructive))' : 'hsl(var(--muted-foreground))' }}>
              {runNodeLabel[node.status]}
            </Typography>
          </Box>
          {node.reason && (
            <Typography title={node.reason} sx={{ ...mutedSx, wordBreak: 'break-word', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
              {shortFailure(node.reason)}
            </Typography>
          )}
        </Box>
      ))}
      {idle > RUN_STUCK_SECONDS && (
        <Typography sx={{ ...mutedSx, mt: 0.75 }}>
          No progress for {formatDuration(idle)}.{waitingOn ? ` Still waiting on ${waitingOn.label}.` : ''} The run may be stuck.
        </Typography>
      )}
      {onOpen && (
        <Button size="small" disableRipple onClick={onOpen} sx={{ ...linkButtonSx, mt: 0.75 }}>Open in debug view</Button>
      )}
    </Box>
  );
};

// ── Workflow preview ──
// The saved workflow drawn top to bottom with each node's app image, shown when the
// agent finishes a chat that changed it. Read from the workflow itself so it always
// matches what was built, not what the agent says it built.
const PREVIEW_NODE_W = 176;
const PREVIEW_NODE_H = 40;
const PREVIEW_GAP_X = 14;
const PREVIEW_GAP_Y = 30;

type PreviewNode = { id: string; label: string; app: string; image: string; trigger: boolean; x: number; y: number; depth: number };
type PreviewEdge = { from: PreviewNode; to: PreviewNode };

/** Layered layout: depth is the longest path from a node with no parents; each layer is centered. */
const layoutWorkflowPreview = (workflow: BackendJson): { nodes: PreviewNode[]; edges: PreviewEdge[]; width: number; height: number } => {
  const raw: BackendJson[] = [
    ...((workflow?.triggers || []) as BackendJson[]).map((t) => ({ ...t, isTrigger: true })),
    ...((workflow?.actions || []) as BackendJson[]),
  ].filter((n) => n?.id);
  const ids = new Set(raw.map((n) => String(n.id)));
  const allLinks = ((workflow?.branches || []) as BackendJson[])
    .map((b) => [String(b?.source_id || ''), String(b?.destination_id || '')])
    .filter(([s, d]) => ids.has(s) && ids.has(d) && s !== d);

  // Drop loop-back edges (DFS from triggers and the start node first), so a loop can't push the trigger down
  const out: Record<string, string[]> = {};
  allLinks.forEach(([s, d]) => { (out[s] ||= []).push(d); });
  const state: Record<string, 1 | 2> = {};
  const back = new Set<string>();
  const visit = (id: string) => {
    state[id] = 1;
    (out[id] || []).forEach((next) => {
      if (state[next] === 1) back.add(`${id}>${next}`);
      else if (!state[next]) visit(next);
    });
    state[id] = 2;
  };
  [...raw.filter((n) => n.isTrigger), ...raw.filter((n) => n.id === workflow?.start), ...raw].forEach((n) => { if (!state[n.id]) visit(n.id); });
  const links = allLinks.filter(([s, d]) => !back.has(`${s}>${d}`));

  const depth: Record<string, number> = {};
  raw.forEach((n) => { depth[n.id] = 0; });
  // Longest path over the now loop-free graph; capped at node count as a safety net
  for (let round = 0; round < raw.length; round += 1) {
    let changed = false;
    links.forEach(([s, d]) => {
      if (depth[d] < depth[s] + 1 && depth[s] + 1 < raw.length) { depth[d] = depth[s] + 1; changed = true; }
    });
    if (!changed) break;
  }

  const layers: BackendJson[][] = [];
  raw.forEach((n) => { (layers[depth[n.id]] ||= []).push(n); });
  const compact = layers.filter(Boolean);
  const widest = Math.max(1, ...compact.map((layer) => layer.length));
  const width = widest * PREVIEW_NODE_W + (widest - 1) * PREVIEW_GAP_X;
  const byId: Record<string, PreviewNode> = {};
  const nodes: PreviewNode[] = [];
  compact.forEach((layer, row) => {
    const layerWidth = layer.length * PREVIEW_NODE_W + (layer.length - 1) * PREVIEW_GAP_X;
    layer.forEach((n, col) => {
      const node: PreviewNode = {
        id: String(n.id),
        label: humanize(n.label || n.name || n.app_name) || 'Step',
        app: humanize(n.app_name || n.name || ''),
        image: String(n.large_image || n.image || ''),
        trigger: !!n.isTrigger,
        x: (width - layerWidth) / 2 + col * (PREVIEW_NODE_W + PREVIEW_GAP_X),
        y: row * (PREVIEW_NODE_H + PREVIEW_GAP_Y),
        depth: row,
      };
      byId[node.id] = node;
      nodes.push(node);
    });
  });
  const edges = links.map(([s, d]) => ({ from: byId[s], to: byId[d] })).filter((e) => e.from && e.to && e.to.y > e.from.y);
  const height = compact.length * PREVIEW_NODE_H + Math.max(0, compact.length - 1) * PREVIEW_GAP_Y;
  return { nodes, edges, width, height };
};

const WorkflowPreview: React.FC<{
  workflowId: string;
  /** Changes whenever the agent edits the workflow, so the preview reloads */
  version: string;
  onSelectNode?: (nodeId: string) => void;
  /** Called once the preview has drawn, so the chat can bring it into view */
  onShown?: () => void;
  resolveUrl: (path: string) => string;
  resolveHeaders: () => Record<string, string>;
}> = ({ workflowId, version, onSelectNode, onShown, resolveUrl, resolveHeaders }) => {
  const [workflow, setWorkflow] = useState<BackendJson>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(resolveUrl(`/api/v1/workflows/${workflowId}`), { credentials: 'include', headers: { ...resolveHeaders() } })
      .then((resp) => (resp.ok ? resp.json() : null))
      .then((data) => { if (!cancelled && data?.id) setWorkflow(data); })
      .catch(() => { /* No preview: the chat and the canvas still show the workflow */ });
    return () => { cancelled = true; };
    // The resolvers are new functions on every render; refetch only when the workflow changes
  }, [workflowId, version]); // eslint-disable-line react-hooks/exhaustive-deps

  const layout = useMemo(() => (workflow ? layoutWorkflowPreview(workflow) : null), [workflow]);
  useEffect(() => {
    if (layout?.nodes.length) onShown?.();
  }, [layout]); // eslint-disable-line react-hooks/exhaustive-deps
  if (!layout || layout.nodes.length === 0) return null;
  const { nodes, edges, width, height } = layout;

  return (
    <Box sx={{ my: 1.5, border: '1px solid hsl(var(--border))', borderRadius: '12px', bgcolor: 'hsl(var(--card, var(--background)))', overflow: 'hidden', ...enterSx }}>
      <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1, px: 1.5, py: 1.125, borderBottom: '1px solid hsl(var(--border))' }}>
        <Typography noWrap sx={{ ...textSx, fontWeight: 600, flex: 1, minWidth: 0 }}>{humanize(workflow?.name) || 'Workflow'}</Typography>
        <Typography sx={{ ...mutedSx, flexShrink: 0 }}>{nodes.length} step{nodes.length === 1 ? '' : 's'}</Typography>
      </Box>
      <Box sx={{
        overflowX: 'auto', px: 2, py: 2,
        backgroundImage: 'radial-gradient(hsl(var(--muted-foreground) / 0.18) 1px, transparent 1px)', backgroundSize: '14px 14px',
      }}>
        <Box sx={{ position: 'relative', width, height, mx: 'auto' }}>
          <Box component="svg" width={width} height={height} aria-hidden sx={{ position: 'absolute', inset: 0, overflow: 'visible' }}>
            {edges.map(({ from, to }) => {
              const x1 = from.x + PREVIEW_NODE_W / 2;
              const y1 = from.y + PREVIEW_NODE_H;
              const x2 = to.x + PREVIEW_NODE_W / 2;
              const y2 = to.y;
              const mid = (y1 + y2) / 2;
              return (
                <Box
                  component="path"
                  key={`${from.id}-${to.id}`}
                  d={`M ${x1} ${y1} C ${x1} ${mid}, ${x2} ${mid}, ${x2} ${y2}`}
                  pathLength={1}
                  sx={{
                    fill: 'none', stroke: 'hsl(var(--muted-foreground) / 0.5)', strokeWidth: 1.25,
                    strokeDasharray: 1, strokeDashoffset: 1,
                    animation: `chatDraw 420ms ease-out ${120 + to.depth * 90}ms forwards`,
                  }}
                />
              );
            })}
          </Box>
          {nodes.map((node) => (
            <ButtonBase
              key={node.id}
              onClick={onSelectNode ? () => onSelectNode(node.id) : undefined}
              disabled={!onSelectNode}
              title={node.app && node.app !== node.label ? `${node.label} (${node.app})` : node.label}
              sx={{
                position: 'absolute', left: node.x, top: node.y, width: PREVIEW_NODE_W, height: PREVIEW_NODE_H,
                display: 'flex', alignItems: 'center', justifyContent: 'flex-start', gap: 1, px: 1, textAlign: 'left',
                border: '1px solid hsl(var(--border))', borderRadius: node.trigger ? '20px' : '10px', bgcolor: 'hsl(var(--background))',
                boxShadow: '0 1px 2px hsl(0 0% 0% / 0.12)', transition: 'border-color 150ms, transform 150ms',
                animation: `chatItemIn 280ms ${EASE_OUT} ${node.depth * 90}ms both`,
                '&:hover, &.Mui-focusVisible': { borderColor: 'hsl(var(--foreground) / 0.35)', transform: 'translateY(-1px)' },
                '&.Mui-disabled': { opacity: 1 },
              }}
            >
              {node.image ? (
                <Box component="img" src={node.image} alt="" sx={{ width: 22, height: 22, borderRadius: '6px', objectFit: 'contain', flexShrink: 0 }} />
              ) : (
                <Box sx={{
                  width: 22, height: 22, borderRadius: '6px', flexShrink: 0, display: 'grid', placeItems: 'center',
                  bgcolor: 'hsl(var(--muted))', ...mutedSx, fontWeight: 600,
                }}>
                  {(node.app || node.label).charAt(0).toUpperCase()}
                </Box>
              )}
              <Typography noWrap sx={{ ...textSx, fontSize: '0.75rem', minWidth: 0 }}>{node.label}</Typography>
            </ButtonBase>
          ))}
        </Box>
      </Box>
    </Box>
  );
};

export const AgentChatView: React.FC<AgentChatViewProps> = ({
  execution,
  agentData,
  isRunning,
  busy,
  pollWarning,
  error,
  workflowId: propWorkflowId,
  placeholder,
  composerContainer,
  contentPaddingX = 0,
  headerActionsContainer,
  onStart,

  onSubmit,
  onStop,
  stopping = false,
  onRefresh,
  isAppAuthenticated,
  authUnknown = false,
  onAuthenticateApp,
  onRefreshAuthenticatedApps,
  resolveUrl,
  resolveHeaders,
  apiBaseUrl,
  onOpenRun,
  onNewChat,
  onOpenPage,
  onFocusDecision,
}) => {
  const executionId = execution?.execution_id || '';
  const decisions = useMemo<ChatDecision[]>(
    () => [...((agentData?.decisions as ChatDecision[]) || [])].sort((a, b) => (a.i || 0) - (b.i || 0)),
    [agentData?.decisions],
  );
  const status = String(execution?.status || agentData?.status || '').toUpperCase();
  const finished = status === 'FINISHED' && !isRunning;
  const stopped = ['FAILURE', 'ABORTED', 'CANCELLED', 'CANCELED'].includes(status);

  const [input, setInput] = useState('');
  const [mode, setMode] = useState<'auto' | 'ask'>(readMode);
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const [openSteps, setOpenSteps] = useState<Record<string, boolean>>({});
  const [rawSteps, setRawSteps] = useState<Record<string, boolean>>({});
  const [watchingAuth, setWatchingAuth] = useState(false);
  const [reverting, setReverting] = useState(false);
  const [bookmarks, setBookmarks] = useState<ChatBookmark[]>(readBookmarks);
  const [historyAnchor, setHistoryAnchor] = useState<HTMLElement | null>(null);
  const [historyTab, setHistoryTab] = useState<'all' | 'bookmarked'>('all');
  const [historyQuery, setHistoryQuery] = useState('');
  const [historyRuns, setHistoryRuns] = useState<AgentRun[] | null>(null);
  const [historyError, setHistoryError] = useState('');
  // A reply or follow-up the backend hasn't recorded yet, shown so it doesn't vanish until it has
  const [pendingMessage, setPendingMessage] = useState('');
  const [now, setNow] = useState(() => Date.now());

  // Answers seen in this session, by finish decision id. A follow-up rewrites that
  // decision, so this keeps the answer on screen without waiting for the replies.
  const knownAnswersRef = useRef<Record<string, string>>({});
  const announcedEditsRef = useRef<{ executionId: string; ids: Set<string> } | null>(null);
  const autoAnsweredRef = useRef<string>('');
  const progressRef = useRef<{ signature: string; at: number }>({ signature: '', at: Date.now() });
  const endRef = useRef<HTMLDivElement | null>(null);

  const workflowId = useMemo(() => {
    if (propWorkflowId) return propWorkflowId;
    for (const decision of decisions) {
      const body = isAction(decision, 'put_agent_update') ? parseJson(getField(decision, 'body')?.value) : null;
      if (body?.workflow_id) return String(body.workflow_id);
    }
    return '';
  }, [propWorkflowId, decisions]);

  const finishDecision = useMemo(() => [...decisions].reverse().find(isFinish), [decisions]);
  const finalAnswer = getDecisionOutput(finishDecision) || agentData?.output || agentData?.message || '';
  if (finishDecision?.run_details?.id && finalAnswer) knownAnswersRef.current[finishDecision.run_details.id] = finalAnswer;

  const labels = useMemo(() => getNodeLabels(decisions), [decisions]);
  const runs = useMemo(() => getWorkflowRuns(decisions), [decisions]);
  const announcedRunsRef = useRef<{ executionId: string; ids: Set<string> } | null>(null);
  const thread = useMemo(() => buildChatThread({
    prompt: agentData?.original_input,
    decisions,
    finalAnswer,
    finished,
    knownAnswers: knownAnswersRef.current,
    modelAnswers: getAnswersFromModelReplies(agentData?.llm_responses),
  }), [agentData?.original_input, agentData?.llm_responses, decisions, finalAnswer, finished]);

  // What the agent is waiting on, if anything
  const waitingQuestion = decisions.find((d) => isAsk(d) && (d.run_details?.status || '').toUpperCase() === 'WAITING');
  const waitingApproval = !waitingQuestion
    ? decisions.find((d) => !isAsk(d) && (d.run_details?.status || '').toUpperCase() === 'WAITING')
    : undefined;
  const questionIndex = waitingQuestion ? Math.max(0, (waitingQuestion.fields || []).findIndex((f) => f.key === 'question')) : 0;
  const questionText = waitingQuestion?.fields?.[questionIndex]?.value || '';
  const needsApproval = asksForApproval(questionText);

  const appsNeedingAuth = useMemo(() => getAppsNeedingAuth(decisions), [decisions]);
  const pendingAuth = authUnknown ? [] : appsNeedingAuth.filter((app) => !isAppAuthenticated(app.name, app.id));
  const connectedAuth = appsNeedingAuth.filter((app) => isAppAuthenticated(app.name, app.id));
  const replies = waitingQuestion && !busy ? getQuickReplies(questionText) : [];

  // The finished chat changed the workflow: preview it. The version reloads it after each edit.
  const workflowEdits = decisions.filter((d) => (isAction(d, 'put_agent_update') || isAppliedRevert(d)) && getStepStatus(d) === 'done').length;
  // On a workflow's page that workflow; otherwise the newest one the chat created or changed
  const createdWorkflowId = [...getChatResources(decisions)].reverse().find((resource) => resource.kind === 'workflow')?.id || '';
  const previewWorkflowId = workflowId || createdWorkflowId;
  const showWorkflowPreview = finished && !!previewWorkflowId && (workflowEdits > 0 || !!createdWorkflowId);

  // A failed step the agent later redid successfully (same tool and action) is a retry, not a failure
  const recoveredSteps = useMemo(() => {
    const recovered = new Set<string>();
    decisions.forEach((decision, index) => {
      if (getStepStatus(decision) !== 'failed') return;
      const kind = stepKind(decision);
      if (decisions.slice(index + 1).some((later) => stepKind(later) === kind && getStepStatus(later) === 'done')) recovered.add(stepId(decision));
    });
    return recovered;
  }, [decisions]);

  // Inline composer: the chat is a column exactly as tall as the rest of the screen. Only the
  // messages scroll, so the composer under them never moves. Re-measured when shown or resized.
  const rootRef = useRef<HTMLDivElement | null>(null);
  const threadRef = useRef<HTMLDivElement | null>(null);
  // Whether the reader was at the bottom before the latest change; new content only follows them if so
  const atBottomRef = useRef(true);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);
  const [fillHeight, setFillHeight] = useState(0);
  useLayoutEffect(() => {
    const root = rootRef.current;
    if (composerContainer || !root) return undefined;
    const measure = () => {
      // Hidden (Start / Timeline tab): its top reads as 0, which would size it to the whole screen
      if (root.getClientRects().length === 0) return;
      // From the chat's top to the bottom of the window, nothing else. Its top is taken as if every
      // container above were scrolled to the start, so scrolling never changes the size, and nothing
      // around or below the chat in the host page is measured.
      let top = root.getBoundingClientRect().top;
      for (let el = root.parentElement; el; el = el.parentElement) top += el.scrollTop;
      setFillHeight(Math.max(320, Math.floor(window.innerHeight - top - CHAT_BOTTOM_GAP)));
    };
    measure();
    // Anything above the chat changing height moves it, so watch the page as well as the chat
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(measure) : null;
    observer?.observe(root);
    observer?.observe(document.body);
    window.addEventListener('resize', measure);
    return () => {
      observer?.disconnect();
      window.removeEventListener('resize', measure);
    };
  }, [composerContainer]);

  // The composer grows with its text up to 160px
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [input]);

  // Reset per-run state when a different run is shown
  useEffect(() => {
    setOpenGroups({});
    setOpenSteps({});
    setRawSteps({});
    setWatchingAuth(false);
    autoAnsweredRef.current = '';
  }, [executionId]);

  // Spot a stuck run: nothing new for a while although it should be working
  const signature = `${status}|${decisions.map((d) => `${d.run_details?.id}:${d.run_details?.status}`).join(',')}`;
  if (signature !== progressRef.current.signature) progressRef.current = { signature, at: Date.now() };
  useEffect(() => {
    if (!isRunning) return undefined;
    const id = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(id);
  }, [isRunning]);
  const stalledFor = isRunning && !waitingQuestion && !waitingApproval ? now - progressRef.current.at : 0;

  // Tell the host about workflow edits it hasn't seen. Edits already in a run
  // when it is first shown are old news. An agent revert reloads the canvas
  // like the chat's own revert does.
  useEffect(() => {
    if (!executionId) return;
    const edits = decisions.filter((d) => (isAction(d, 'put_agent_update') || isAppliedRevert(d)) && getStepStatus(d) === 'done' && d.run_details?.id);
    if (announcedEditsRef.current?.executionId !== executionId) {
      announcedEditsRef.current = { executionId, ids: new Set(edits.map((d) => d.run_details!.id!)) };
      return;
    }
    const fresh = edits.filter((d) => !announcedEditsRef.current!.ids.has(d.run_details!.id!));
    if (fresh.length === 0 || !workflowId) return;
    fresh.forEach((d) => announcedEditsRef.current!.ids.add(d.run_details!.id!));
    const detail: AgentWorkflowChangedDetail = fresh.some(isAppliedRevert)
      ? { workflowId, nodeIds: [], reverted: true }
      : { workflowId, nodeIds: fresh.flatMap(getChangedNodeIds) };
    window.dispatchEvent(new CustomEvent(AGENT_WORKFLOW_CHANGED_EVENT, { detail }));
  }, [decisions, executionId, workflowId]);

  // Open runs the agent starts in the host's run view. Runs already in the
  // agent run when it is first shown are not reopened.
  const openRun = useCallback((run: ChatWorkflowRun) => {
    if (!propWorkflowId) return;
    const detail: AgentOpenExecutionDetail = { workflowId: run.workflowId || propWorkflowId, executionId: run.executionId, authorization: run.authorization };
    window.dispatchEvent(new CustomEvent(AGENT_OPEN_EXECUTION_EVENT, { detail }));
  }, [propWorkflowId]);

  useEffect(() => {
    if (!executionId) return;
    if (announcedRunsRef.current?.executionId !== executionId) {
      announcedRunsRef.current = { executionId, ids: new Set(runs.map((run) => run.executionId)) };
      return;
    }
    const fresh = runs.filter((run) => !announcedRunsRef.current!.ids.has(run.executionId));
    fresh.forEach((run) => announcedRunsRef.current!.ids.add(run.executionId));
    if (fresh.length > 0) openRun(fresh[fresh.length - 1]);
  }, [runs, executionId, openRun]);

  // Credentials: once the user opened the setup, keep refreshing the org's
  // credentials until they exist, then carry on for them if that was all the
  // agent asked. A question that also wants approval waits for the user.
  useEffect(() => {
    // While auth is unknown pendingAuth is empty too, which must not count as "connected"
    if (!watchingAuth || authUnknown || pendingAuth.length > 0) return undefined;
    setWatchingAuth(false);
    const names = connectedAuth.map((app) => app.name).join(', ');
    const decisionId = waitingQuestion?.run_details?.id;
    if (!decisionId || needsApproval || autoAnsweredRef.current === decisionId) {
      if (names) toast.success(`${names} connected. Reply to the agent to continue.`);
      return undefined;
    }
    autoAnsweredRef.current = decisionId;
    toast.success(`${names} connected. Continuing.`);
    onSubmit(decisionId, { answer: { index: questionIndex, value: `I've added the ${names} credentials.` } });
    return undefined;
  }, [watchingAuth, authUnknown, pendingAuth.length]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!watchingAuth) return undefined;
    const id = window.setInterval(onRefreshAuthenticatedApps, AUTH_WATCH_MS);
    return () => window.clearInterval(id);
  }, [watchingAuth, onRefreshAuthenticatedApps]);

  // Drop the pending message once the run shows it, or when sending it failed
  const pendingRecorded = !!pendingMessage && thread.some((item) => item.type === 'user' && item.text === pendingMessage);
  useEffect(() => {
    if (pendingMessage && (pendingRecorded || (!busy && !isRunning))) setPendingMessage('');
  }, [pendingMessage, pendingRecorded, busy, isRunning]);

  // Keep the newest message in view. A new message always scrolls; new steps only
  // follow along when the reader is already near the bottom, so reading isn't interrupted.
  const scrolledRef = useRef<{ executionId: string; length: number }>({ executionId: '', length: 0 });
  useEffect(() => {
    const end = endRef.current;
    if (!end) return;
    const last = scrolledRef.current;
    const freshChat = last.executionId !== executionId || last.length === 0;
    const grew = thread.length > last.length || !!pendingMessage;
    scrolledRef.current = { executionId, length: thread.length };
    const behavior: ScrollBehavior = freshChat ? 'auto' : 'smooth';
    // Inline: the thread is its own scroll area. Sidebar: AgentUI's panel scrolls, so scroll the end into view.
    const scroller = composerContainer ? null : threadRef.current;
    if (scroller) {
      if (freshChat || grew || atBottomRef.current) scroller.scrollTo({ top: scroller.scrollHeight, behavior });
      return;
    }
    const nearBottom = end.getBoundingClientRect().top - window.innerHeight < 320;
    if (freshChat || grew || nearBottom) end.scrollIntoView({ block: 'end', behavior });
  }, [composerContainer, fillHeight, executionId, thread.length, decisions.length, pendingMessage, busy, waitingQuestion?.run_details?.id, finished]);

  const changeMode = (value: 'auto' | 'ask') => {
    setMode(value);
    try {
      window.localStorage.setItem(MODE_STORAGE_KEY, value);
    } catch { /* ignore */ }
  };

  const compose = (text: string) => (mode === 'ask' && workflowId ? ASK_FIRST_PREFIX + text : text);

  const send = async (raw?: string) => {
    const text = (raw ?? input).trim();
    if (!text || busy || (isRunning && !waitingQuestion)) return;
    setInput('');

    if (waitingQuestion?.run_details?.id) {
      setPendingMessage(text);
      await onSubmit(waitingQuestion.run_details.id, { answer: { index: questionIndex, value: text } });
      return;
    }
    if (finished && finishDecision?.run_details?.id) {
      setPendingMessage(text);
      await onSubmit(finishDecision.run_details.id, { continue: compose(text) }, true);
      return;
    }
    await onStart(compose(text));
  };

  // The workflow as it was before this chat comes from its saved revisions, not from
  // memory, so revert still works after closing the panel or reloading the page.
  const chatStartedAt = Number((execution as { started_at?: unknown } | undefined)?.started_at) || 0;
  const [revertTarget, setRevertTarget] = useState<{ anchor: HTMLElement; until: number } | null>(null);
  const revert = async (until: number) => {
    if (!workflowId || !until || reverting) return;
    const toTurn = until !== chatStartedAt;
    setReverting(true);
    try {
      const listResp = await fetch(resolveUrl(`/api/v1/workflows/${workflowId}/revisions?count=100`), {
        credentials: 'include',
        headers: { ...resolveHeaders() },
      });
      if (!listResp.ok) throw new Error(`loading saved versions failed with status ${listResp.status}`);
      const revisions = await listResp.json();
      const before = pickRevisionBefore(Array.isArray(revisions) ? revisions : [], until);
      if (!before) throw new Error(`there is no saved version from ${toTurn ? 'that point' : 'before this chat'}`);
      const { revision_id: _revisionId, ...workflow } = before;
      const resp = await fetch(resolveUrl(`/api/v1/workflows/${workflowId}`), {
        method: 'PUT',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json', ...resolveHeaders() },
        body: JSON.stringify(workflow),
      });
      if (!resp.ok) throw new Error(`status ${resp.status}`);
      toast.success(toTurn ? 'Reverted the workflow to that point in the chat' : 'Reverted the changes from this chat');
      const detail: AgentWorkflowChangedDetail = { workflowId, nodeIds: [], reverted: true };
      window.dispatchEvent(new CustomEvent(AGENT_WORKFLOW_CHANGED_EVENT, { detail }));
    } catch (err) {
      toast.error(`Failed to revert: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setReverting(false);
      setRevertTarget(null);
    }
  };

  const focusNode = (nodeId?: string) => {
    if (!nodeId || !workflowId) return;
    const detail: AgentFocusNodeDetail = { workflowId, nodeId };
    window.dispatchEvent(new CustomEvent(AGENT_FOCUS_NODE_EVENT, { detail }));
  };

  const connect = (app: { id: string; name: string; url?: string }) => {
    onAuthenticateApp(app);
    setWatchingAuth(true);
  };

  const renderStep = (decision: ChatDecision) => {
    const id = stepId(decision);
    const stepStatus = getStepStatus(decision);
    const retried = stepStatus === 'failed' && recoveredSteps.has(id);
    const failedForGood = stepStatus === 'failed' && !retried;
    const active = stepStatus === 'running' || stepStatus === 'waiting';
    const { label, details } = describeStep(decision, labels);
    const failure = getFailureReason(decision);
    const duration = getStepDuration(decision);
    const nodeIds = isAction(decision, 'put_agent_update') ? getChangedNodeIds(decision) : [];
    const open = !!openSteps[id];
    const toggle = () => setOpenSteps((prev) => ({ ...prev, [id]: !prev[id] }));
    const dotColor = active ? 'hsl(var(--primary))'
      : retried ? WARNING_COLOR
      : failedForGood ? 'hsl(var(--destructive))'
      : 'hsl(var(--muted-foreground) / 0.45)';
    const meta = retried ? 'Retried'
      : failedForGood ? 'Failed'
      : stepStatus === 'done' ? formatDuration(duration)
      : stepStatus === 'waiting' ? 'Waiting'
      : '';

    return (
      <Box key={id} sx={enterSx}>
        <Box
          role="button"
          tabIndex={0}
          aria-expanded={open}
          onClick={toggle}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } }}
          sx={{
            display: 'grid', gridTemplateColumns: '6px minmax(0, 1fr) auto', columnGap: 1.25, alignItems: 'baseline',
            py: 0.375, cursor: 'pointer', borderRadius: 1,
            '& .step-label': { transition: 'color 120ms' },
            '&:hover .step-label, &:focus-visible .step-label': { color: 'hsl(var(--foreground))' },
            '&:focus-visible': { outline: '2px solid hsl(var(--ring, var(--primary)))', outlineOffset: 2 },
          }}
        >
          <Box sx={{
            width: 6, height: 6, borderRadius: '50%', bgcolor: dotColor, transform: 'translateY(-1px)', transition: 'background-color 200ms',
            animation: active ? 'chatPulse 1.2s ease-in-out infinite' : undefined,
          }} />
          <Typography className="step-label" sx={{ ...textSx, color: 'hsl(var(--foreground) / 0.82)', minWidth: 0, wordBreak: 'break-word' }}>{label}</Typography>
          <Typography sx={{
            ...mutedSx, fontVariantNumeric: 'tabular-nums',
            color: retried ? WARNING_COLOR : failedForGood ? 'hsl(var(--destructive) / 0.85)' : 'hsl(var(--muted-foreground))',
          }}>
            {meta}
          </Typography>
          {failure && (
            <Typography title={failure} sx={{
              ...mutedSx, gridColumn: '2 / 4', mt: 0.125, wordBreak: 'break-word',
              display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical', overflow: 'hidden',
            }}>
              {shortFailure(failure)}
            </Typography>
          )}
        </Box>
        <Collapse in={open} timeout={200} unmountOnExit>
          <Box sx={{ ml: '16px', mt: 0.25, mb: 0.75, display: 'flex', flexDirection: 'column', gap: 0.25 }}>
            {details.map((detail, index) => {
              const clickableDetail = Boolean(onFocusDecision && isAction(decision, 'put_agent_update'));
              const focusDetail = () => onFocusDecision?.({ decisionId: id, operationIndex: index, changeText: detail });
              return (
                <Typography
                  key={index}
                  role={clickableDetail ? 'button' : undefined}
                  tabIndex={clickableDetail ? 0 : undefined}
                  onClick={clickableDetail ? focusDetail : undefined}
                  onKeyDown={clickableDetail ? (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      focusDetail();
                    }
                  } : undefined}
                  sx={{
                    ...mutedSx,
                    cursor: clickableDetail ? 'pointer' : 'default',
                    '&:hover': clickableDetail ? { color: 'hsl(var(--foreground))', textDecoration: 'underline' } : undefined,
                  }}
                >
                  {detail}
                </Typography>
              );
            })}
            <Box sx={{ display: 'flex', flexWrap: 'wrap', columnGap: 1.5, rowGap: 0.25, mt: 0.25 }}>
              {onFocusDecision && (
                <Button size="small" disableRipple onClick={() => onFocusDecision({ decisionId: id })} sx={linkButtonSx}>Show in timeline</Button>
              )}
              {nodeIds.length > 0 && workflowId && (
                <Button size="small" disableRipple onClick={() => focusNode(nodeIds[0])} sx={linkButtonSx}>Show on canvas</Button>
              )}
              <Button size="small" disableRipple onClick={() => setRawSteps((prev) => ({ ...prev, [id]: !prev[id] }))} sx={linkButtonSx}>
                {rawSteps[id] ? 'Hide details' : 'Details'}
              </Button>
            </Box>
            <Collapse in={!!rawSteps[id]} timeout={180} unmountOnExit>
              <Box component="pre" sx={{
                fontSize: '0.6875rem', color: 'hsl(var(--muted-foreground))', bgcolor: 'hsl(var(--muted) / 0.4)',
                borderRadius: '8px', p: 1, mt: 0.5, mb: 0, maxHeight: 240, overflow: 'auto', whiteSpace: 'pre-wrap', wordBreak: 'break-word',
              }}>
                {JSON.stringify({ tool: decision.tool, action: decision.action, fields: decision.fields, result: getResult(decision) ?? decision.run_details?.raw_response }, null, 2)}
              </Box>
            </Collapse>
          </Box>
        </Collapse>
      </Box>
    );
  };

  // Only the newest item can be live. Right after the user sends a message the
  // agent has no steps for it yet, so the previous turn's steps must stay finished.
  const working = isRunning && !waitingQuestion;
  const lastItem = thread[thread.length - 1];
  const liveGroupKey = working && lastItem?.type === 'steps' ? lastItem.key : undefined;

  // A later message counts as sent when the agent began on it: the editor saves right before
  const revertPoints: Record<string, number> = {};
  thread.forEach((item, index) => {
    if (item.type !== 'user') return;
    const next = thread.slice(index + 1).find((later) => later.type === 'steps');
    const until = item.key === 'prompt' ? chatStartedAt : toSeconds(next?.type === 'steps' ? next.steps[0]?.run_details?.started_at : 0);
    if (until && getChangesAfter(decisions, labels, until).length > 0) revertPoints[item.key] = until;
  });

  const renderItem = (item: ChatThreadItem) => {
    if (item.type === 'user') {
      const cleanUserText = stripModePrefix(item.text);
      if (!cleanUserText) return null;
      const until = workflowId && !working ? revertPoints[item.key] : 0;
      return (
        <Box key={item.key} sx={{ display: 'flex', justifyContent: 'flex-end', my: 1.5, pr: 1.5, pt: 1.25, ...enterSx, '&:hover .revert-to-here': { opacity: 1 } }}>
          <Box sx={{
            ...textSx, position: 'relative', maxWidth: '85%', px: 1.75, py: 1.125, whiteSpace: 'pre-wrap', wordBreak: 'break-word',
            bgcolor: 'hsl(var(--muted))', borderRadius: '16px 16px 4px 16px',
          }}>
            {cleanUserText}
            {!!until && (
              <Tooltip title="Revert Workflow to this point">
                <IconButton
                  className="revert-to-here"
                  aria-label="Revert Workflow to this point"
                  onClick={(event) => setRevertTarget({ anchor: event.currentTarget, until })}
                  sx={{
                    position: 'absolute', top: -10, right: -10, width: 22, height: 22, p: 0,
                    opacity: revertTarget?.until === until ? 1 : 0, transition: 'opacity 120ms',
                    bgcolor: 'hsl(var(--background))', border: '1px solid hsl(var(--border))', color: 'hsl(var(--muted-foreground))',
                    '&:hover': { bgcolor: 'hsl(var(--muted))', color: 'hsl(var(--foreground))' },
                    '&:focus-visible': { opacity: 1 },
                    '@media (hover: none)': { opacity: 1 },
                  }}
                >
                  <Undo2 size={13} />
                </IconButton>
              </Tooltip>
            )}
          </Box>
        </Box>
      );
    }

    if (item.type === 'agent') {
      return (
        <Box key={item.key} sx={{ ...textSx, ...revealSx, my: 1.5, minWidth: 0, '& pre, & table': { maxWidth: '100%', overflowX: 'auto' }, '& p': { my: 0.75 }, '& > *:first-of-type > *:first-child': { mt: 0 } }}>
          <ShuffleMarkdown components={agentMarkdownComponents}>{item.text}</ShuffleMarkdown>
        </Box>
      );
    }

    // Steps: one quiet line. The live group shows what the agent is doing now; finished ones fold away.
    const live = item.key === liveGroupKey;
    const open = openGroups[item.key] ?? live;
    const failedSteps = item.steps.filter((step) => getStepStatus(step) === 'failed');
    const retriedCount = failedSteps.filter((step) => recoveredSteps.has(stepId(step))).length;
    const failedCount = failedSteps.length - retriedCount;
    const starts = item.steps.map((s) => toSeconds(s.run_details?.started_at)).filter(Boolean);
    const ends = item.steps.map((s) => toSeconds(s.run_details?.completed_at)).filter(Boolean);
    const duration = starts.length && ends.length ? Math.max(...ends) - Math.min(...starts) : 0;
    const current = item.steps[item.steps.length - 1];
    // The reason is written in the present tense ("Looking up actions for Gmail..."), the label in the past
    const currentText = current ? (current.reason || describeStep(current, labels).label) : 'Working';
    const toggleGroup = () => setOpenGroups((prev) => ({ ...prev, [item.key]: !open }));

    const stepIds = new Set(item.steps.map(stepId));
    const groupRuns = runs.filter((run) => stepIds.has(run.decisionId));

    return (
      <React.Fragment key={item.key}>
      <Box sx={{ my: 1, ...enterSx }}>
        <ButtonBase
          onClick={toggleGroup}
          aria-expanded={open}
          sx={{
            ...mutedSx, display: 'flex', justifyContent: 'flex-start', gap: 0.75, maxWidth: '100%', py: 0.375, borderRadius: 1,
            textAlign: 'left', transition: 'color 120ms',
            '&:hover, &.Mui-focusVisible': { color: 'hsl(var(--foreground))' },
            '&:hover .group-toggle, &.Mui-focusVisible .group-toggle': { opacity: 1 },
          }}
        >
          {live ? (
            <Box component="span" sx={{ ...shimmerSx, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{currentText}</Box>
          ) : (
            <Box component="span" sx={{ minWidth: 0 }}>
              Worked{duration > 0 ? ` for ${formatDuration(duration)}` : ''}
              {` · ${item.steps.length} step${item.steps.length === 1 ? '' : 's'}`}
              {retriedCount > 0 && <Box component="span" sx={{ color: WARNING_COLOR }}>{` · ${retriedCount} retried`}</Box>}
              {failedCount > 0 && <Box component="span" sx={{ color: 'hsl(var(--destructive) / 0.85)' }}>{` · ${failedCount} failed`}</Box>}
            </Box>
          )}
          <Box component="span" className="group-toggle" sx={{ flexShrink: 0, opacity: 0.6, transition: 'opacity 120ms' }}>{open ? 'Hide' : 'Show'}</Box>
        </ButtonBase>
        <Collapse in={open} timeout={220}>
          <Box sx={{ mt: 0.5, ml: '2px', pl: 1.5, borderLeft: '1px solid hsl(var(--border))', display: 'flex', flexDirection: 'column' }}>
            {item.steps.map(renderStep)}
          </Box>
        </Collapse>
      </Box>
      {groupRuns.map((run) => (
        <RunCard
          key={run.executionId}
          run={run}
          onOpen={propWorkflowId && (!run.workflowId || run.workflowId === propWorkflowId) ? () => openRun(run) : undefined}
          resolveUrl={resolveUrl}
          resolveHeaders={resolveHeaders}
        />
      ))}
      </React.Fragment>
    );
  };

  const renderWaiting = () => {
    if (waitingApproval?.run_details?.id) {
      const { label } = describeStep(waitingApproval, labels);
      const approvalId = waitingApproval.run_details.id;
      return (
        <Box sx={{ my: 1.25, p: 1.5, border: '1px solid hsl(var(--border))', borderRadius: '12px', bgcolor: 'hsl(var(--muted) / 0.25)', ...enterSx }}>
          <Typography sx={textSx}>The agent wants to: {label}</Typography>
          {waitingApproval.reason && <Typography sx={{ ...mutedSx, mt: 0.5 }}>{waitingApproval.reason}</Typography>}
          <Box sx={{ display: 'flex', gap: 1, mt: 1.25 }}>
            <Button size="small" variant="contained" disabled={busy} onClick={() => onSubmit(approvalId, { approve: 'true' })} sx={primaryActionButtonSx}>Approve</Button>
            <Button size="small" variant="outlined" disabled={busy} onClick={() => onSubmit(approvalId, { approve: 'false' })} sx={outlinedActionButtonSx}>Deny</Button>
          </Box>
        </Box>
      );
    }
    if (!waitingQuestion) return null;

    return (
      <Box sx={{ my: 1, ...enterSx }}>
        {connectedAuth.map((app) => (
          <Typography key={app.id} sx={{ ...mutedSx, mb: 0.75 }}>{app.name} connected</Typography>
        ))}
        {pendingAuth.map((app) => (
          <Box key={app.id} sx={{ display: 'flex', alignItems: 'center', gap: 1.5, p: 1.25, mb: 1, border: '1px solid hsl(var(--border))', borderRadius: '12px', bgcolor: 'hsl(var(--muted) / 0.25)' }}>
            <Box sx={{ flex: 1, minWidth: 0 }}>
              <Typography sx={textSx}>Connect {app.name}</Typography>
              <Typography sx={mutedSx}>
                {watchingAuth
                  ? (needsApproval ? 'Waiting for the credentials. You will still confirm the plan yourself.' : 'Waiting for the credentials. The agent continues once they are saved.')
                  : (needsApproval ? 'Add credentials, then reply to the agent.' : 'Add credentials and the agent continues by itself.')}
              </Typography>
            </Box>
            <Button size="small" variant="contained" onClick={() => connect(app)} sx={{ ...primaryActionButtonSx, flexShrink: 0 }}>Connect</Button>
          </Box>
        ))}
        {replies.length > 0 && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75 }}>
            {replies.map((reply) => (
              <Button key={reply} size="small" variant="outlined" onClick={() => send(reply)} sx={replySx}>{reply}</Button>
            ))}
          </Box>
        )}
      </Box>
    );
  };

  const renderSummary = () => {
    if (!finished) return null;
    const problems: { text: string; nodeId?: string; decisionId?: string; decisionIndex?: number }[] = [
      ...pendingAuth.map((app) => ({ text: `${app.name} needs credentials before it can run`, nodeId: undefined as string | undefined })),
      ...getWorkflowProblems(decisions),
    ];
    if (problems.length === 0) return null;

    return (
      <Box sx={{ my: 1.25, p: 1.5, border: '1px solid hsl(var(--border))', borderRadius: '12px', ...enterSx }}>
        {problems.length > 0 && (
          <>
            <Typography sx={{ ...mutedSx, mb: 0.5 }}>Needs attention before running</Typography>
            {problems.map((problem, index) => {
              const linkedProblem = Boolean(
                (onFocusDecision && (problem.decisionId !== undefined || problem.decisionIndex !== undefined)) ||
                (problem.nodeId && workflowId),
              );
              const handleSelectProblem = () => {
                focusNode(problem.nodeId);
                if (problem.decisionId !== undefined || problem.decisionIndex !== undefined) {
                  onFocusDecision?.({
                    decisionId: problem.decisionId,
                    decisionIndex: problem.decisionIndex,
                    nodeId: problem.nodeId,
                  });
                }
              };
              return (
                <Typography
                  key={index}
                  role={linkedProblem ? 'button' : undefined}
                  tabIndex={linkedProblem ? 0 : undefined}
                  onClick={linkedProblem ? handleSelectProblem : undefined}
                  onKeyDown={linkedProblem ? (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleSelectProblem();
                    }
                  } : undefined}
                  sx={{
                    ...textSx,
                    cursor: linkedProblem ? 'pointer' : 'default',
                    '&:hover': linkedProblem ? { color: 'hsl(var(--foreground))', textDecoration: 'underline' } : undefined,
                  }}
                >
                  {problem.text}
                </Typography>
              );
            })}
          </>
        )}
      </Box>
    );
  };

  // The pages of what the agent created or changed elsewhere, e.g. the workflow it built from the docs.
  // Opening one makes this chat that page's chat, so it is there when the user comes back.
  const openResource = (resource: ChatResource) => {
    if (!resource.url || !executionId) return;
    const authorization = execution?.authorization || '';
    setPageContextChoice(resolveAgentContext(resource.url).storageKey, {
      executionId,
      authorization: authorization || null,
      executionStatus: status || null,
      viewMode: 'chat',
    });
    const search = new URLSearchParams({ execution_id: executionId, ...(authorization ? { authorization } : {}), agentView: 'chat' });
    const target = `${resource.url}?${search.toString()}`;
    if (onOpenPage) onOpenPage(target);
    else window.location.assign(target);
  };

  const renderResourceLinks = () => {
    if (!finished || !executionId || typeof window === 'undefined') return null;
    const here = window.location.pathname.replace(/\/+$/, '');
    const elsewhere = getChatResources(decisions).filter((resource) => resource.url && resource.url.replace(/\/+$/, '') !== here);
    if (elsewhere.length === 0) return null;

    return (
      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.75, my: 1.25 }}>
        {elsewhere.map((resource) => (
          <Button key={`${resource.kind}:${resource.id}`} size="small" variant="outlined" onClick={() => openResource(resource)} sx={replySx}>
            {resource.name ? `Open ${resource.name}` : `Open the ${resource.kind}`}
          </Button>
        ))}
      </Box>
    );
  };

  const composerPlaceholder = waitingQuestion
    ? 'Reply to the agent...'
    : stopping ? 'Stopping the agent...'
    : isRunning ? 'The agent is working...'
    : finished && finishDecision ? 'Ask a follow-up or for more changes...'
    : placeholder || 'Ask about this workflow or describe a change...';
  const canType = !busy && (!isRunning || !!waitingQuestion);

  const composer = (
    <Box
      component="form"
      onSubmit={(e) => { e.preventDefault(); send(); }}
      onClick={() => inputRef.current?.focus()}
      sx={{
        border: '1px solid',
        borderColor: waitingQuestion ? 'hsl(var(--foreground) / 0.4)' : 'hsl(var(--border))',
        borderRadius: '14px',
        bgcolor: 'hsl(var(--card, var(--background)))',
        boxShadow: '0 6px 24px hsl(0 0% 0% / 0.12)',
        px: 1.5,
        pt: 1.25,
        pb: 0.875,
        cursor: 'text',
        transition: 'border-color 150ms, box-shadow 150ms',
        '&:focus-within': { borderColor: 'hsl(var(--foreground) / 0.35)', boxShadow: '0 8px 28px hsl(0 0% 0% / 0.16)' },
      }}
    >
      <Box
        component="textarea"
        ref={inputRef}
        value={input}
        disabled={!canType}
        placeholder={composerPlaceholder}
        rows={1}
        aria-label="Message the agent"
        onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) => setInput(e.target.value)}
        onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
          if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
            e.preventDefault();
            send();
          }
        }}
        sx={{
          display: 'block', width: '100%', border: 'none', outline: 'none', resize: 'none', bgcolor: 'transparent', p: 0,
          color: 'hsl(var(--foreground))', fontFamily: 'inherit', fontSize: '0.875rem', lineHeight: 1.5, maxHeight: 160, overflowY: 'auto',
          transition: 'height 120ms ease-out',
          '&::placeholder': { color: 'hsl(var(--muted-foreground) / 0.8)' },
          '&:disabled': { cursor: 'default' },
        }}
      />
      <Box sx={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', columnGap: 1.5, rowGap: 0.5, mt: 0.75 }}>
        {workflowId && (
          <Tooltip title="Auto-build makes changes right away. Ask before changing shows a plan and waits for your OK.">
            <Select
              size="small"
              variant="standard"
              disableUnderline
              value={mode}
              onClick={(e) => e.stopPropagation()}
              onChange={(e) => changeMode(e.target.value as 'auto' | 'ask')}
              sx={{ fontSize: '0.75rem', color: 'hsl(var(--muted-foreground))' }}
            >
              <MenuItem value="auto" sx={{ fontSize: '0.8125rem' }}>Auto-build</MenuItem>
              <MenuItem value="ask" sx={{ fontSize: '0.8125rem' }}>Ask before changing</MenuItem>
            </Select>
          </Tooltip>
        )}
        <Box sx={{ flex: 1 }} />
        {isRunning && !waitingQuestion ? (
          <Button size="small" variant="outlined" disabled={stopping} onClick={onStop} sx={{ ...outlinedActionButtonSx, borderRadius: '8px', px: 1.5 }}>{stopping ? 'Stopping...' : 'Stop'}</Button>
        ) : (
          <Button type="submit" size="small" variant="contained" disabled={!canType || !input.trim()} sx={{ ...primaryActionButtonSx, borderRadius: '8px', px: 1.5, transition: 'background-color 150ms, color 150ms' }}>
            {busy ? 'Sending...' : 'Send'}
          </Button>
        )}
      </Box>
    </Box>
  );

  const renderRevertConfirm = () => {
    if (!revertTarget) return null;
    const undone = getChangesAfter(decisions, labels, revertTarget.until);
    const whole = revertTarget.until === chatStartedAt;
    const close = () => { if (!reverting) setRevertTarget(null); };
    return (
      <Popover
        open
        anchorEl={revertTarget.anchor}
        onClose={close}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: {
              mt: 1, p: 1.5, width: 320,
              bgcolor: 'hsl(var(--card))', color: 'hsl(var(--foreground))',
              border: '1px solid hsl(var(--border))', backgroundImage: 'none',
            },
          },
        }}
      >
        <Typography sx={{ ...textSx, fontWeight: 600 }}>{whole ? 'Revert all changes from this chat?' : 'Revert workflow to this point?'}</Typography>
        <Typography sx={{ ...mutedSx, mt: 0.5, mb: 0.75 }}>
          Undoes {undone.length} change{undone.length === 1 ? '' : 's'} the agent made {whole ? 'in this chat' : 'after this message'}:
        </Typography>
        <Box component="ul" sx={{ m: 0, pl: 2.25, maxHeight: 180, overflowY: 'auto' }}>
          {undone.map((text, index) => (
            <Typography component="li" key={index} sx={{ ...textSx, fontSize: '0.75rem' }}>{text}</Typography>
          ))}
        </Box>
        <Typography sx={{ ...mutedSx, mt: 1 }}>Anything saved since then is replaced too, including your own edits and other people's.</Typography>
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 1, mt: 1.5 }}>
          <Button size="small" variant="outlined" disabled={reverting} onClick={close} sx={outlinedActionButtonSx}>Cancel</Button>
          <Button size="small" variant="contained" disabled={reverting} onClick={() => revert(revertTarget.until)} sx={primaryActionButtonSx}>
            {reverting ? 'Reverting...' : 'Revert'}
          </Button>
        </Box>
      </Popover>
    );
  };

  const bookmarked = !!executionId && bookmarks.some((bookmark) => bookmark.execution_id === executionId);
  const toggleBookmark = () => {
    if (!executionId) return;
    const next = bookmarked
      ? bookmarks.filter((bookmark) => bookmark.execution_id !== executionId)
      : [{
        execution_id: executionId,
        authorization: execution?.authorization,
        title: getRunTitle(execution as unknown as AgentRun),
        started_at: String((execution as { started_at?: unknown } | undefined)?.started_at || ''),
        workflow_id: workflowId || undefined,
      }, ...bookmarks];
    setBookmarks(next);
    writeBookmarks(next);
  };

  const loadHistory = useCallback(() => {
    setHistoryError('');
    searchAgentActivity({ limit: 50, apiBaseUrl, workflowId: 'AGENT', light: true })
      .then((result) => {
        if (result.success) setHistoryRuns(result.runs || []);
        else setHistoryError('Could not load your chats');
      })
      .catch(() => setHistoryError('Could not load your chats'));
  }, [apiBaseUrl]);

  useEffect(() => {
    if (onOpenRun) loadHistory();
  }, [loadHistory, onOpenRun]);

  const openHistory = (anchor: HTMLElement) => {
    setHistoryAnchor(anchor);
    setHistoryQuery('');
    loadHistory();
  };

  const openChat = (chatId: string, authorization?: string, runStatus?: string) => {
    setHistoryAnchor(null);
    if (chatId !== executionId) onOpenRun?.(chatId, authorization, runStatus);
  };

  const headerActions = (
    <>
      <Tooltip title={!executionId ? 'Send a message first to bookmark this chat' : bookmarked ? 'Remove bookmark' : 'Bookmark this chat'}>
        <span>
          <IconButton size="small" aria-label={bookmarked ? 'Remove bookmark' : 'Bookmark this chat'} aria-pressed={bookmarked} disabled={!executionId} onClick={toggleBookmark} sx={headerIconButtonSx}>
            <Bookmark size={16} fill={bookmarked ? 'currentColor' : 'none'} />
          </IconButton>
        </span>
      </Tooltip>
      {onOpenRun && (
        <Tooltip title="Chat history">
          <IconButton size="small" aria-label="Chat history" aria-haspopup="dialog" onClick={(event) => openHistory(event.currentTarget)} sx={headerIconButtonSx}>
            <History size={16} />
          </IconButton>
        </Tooltip>
      )}
      {onNewChat && (
        <Tooltip title="New chat">
          <span>
            <IconButton size="small" aria-label="New chat" disabled={!executionId || busy} onClick={onNewChat} sx={headerIconButtonSx}>
              <MessageCirclePlus size={16} />
            </IconButton>
          </span>
        </Tooltip>
      )}
    </>
  );

  const renderHistory = () => {
    const query = historyQuery.trim().toLowerCase();
    const items = historyTab === 'bookmarked'
      ? bookmarks.map((bookmark) => ({ id: bookmark.execution_id, authorization: bookmark.authorization, status: undefined as string | undefined, title: bookmark.title, startedAt: bookmark.started_at, workflowIds: [bookmark.workflow_id] }))
      : (historyRuns || []).map((run) => ({ id: run.execution_id, authorization: run.authorization, status: run.status as string | undefined, title: getRunTitle(run), startedAt: run.started_at,
        // A chat belongs to the workflow it started on and to every workflow it created or changed
        workflowIds: [run.execution_argument, ...getRunResources(run).filter((resource) => resource.kind === 'workflow').map((resource) => resource.id)] }));
    const shown = items
      .filter((item) => !query || item.title.toLowerCase().includes(query))
      .sort((a, b) => Number(b.startedAt || 0) - Number(a.startedAt || 0));
    // A chat on a workflow keeps that workflow's ID as its argument
    const sections = workflowId
      ? [
        { label: 'This workflow', items: shown.filter((item) => item.workflowIds.includes(workflowId)) },
        { label: 'Other chats', items: shown.filter((item) => !item.workflowIds.includes(workflowId)) },
      ]
      : [{ label: '', items: shown }];
    const loading = historyTab === 'all' && historyRuns === null && !historyError;
    const emptyText = historyTab === 'all' && historyError
      ? historyError
      : query ? 'No chats match your search'
        : historyTab === 'bookmarked' ? 'No bookmarked chats yet. Bookmark one from the top of a chat.' : 'No chats yet';

    return (
      <Popover
        open={!!historyAnchor}
        anchorEl={historyAnchor}
        onClose={() => setHistoryAnchor(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: {
            sx: {
              mt: 1, p: 1, width: 360, maxWidth: 'calc(100vw - 32px)',
              bgcolor: 'hsl(var(--card))', color: 'hsl(var(--foreground))', backgroundImage: 'none',
              border: '1px solid hsl(var(--border))', borderRadius: '12px', boxShadow: '0 12px 32px rgba(0, 0, 0, 0.28)',
            },
          },
        }}
      >
        <Box role="tablist" aria-label="Chats" sx={{ display: 'flex', p: 0.375, gap: 0.375, borderRadius: '10px', bgcolor: 'hsl(var(--muted) / 0.6)' }}>
          {(['all', 'bookmarked'] as const).map((tab) => {
            const active = historyTab === tab;
            return (
              <ButtonBase
                key={tab}
                role="tab"
                aria-selected={active}
                onClick={() => setHistoryTab(tab)}
                sx={{
                  flex: 1, py: 0.75, borderRadius: '8px', fontSize: '0.8125rem', fontWeight: active ? 600 : 500,
                  color: active ? 'hsl(var(--foreground))' : 'hsl(var(--muted-foreground))',
                  bgcolor: active ? 'hsl(var(--background))' : 'transparent',
                  boxShadow: active ? '0 1px 2px rgba(0, 0, 0, 0.2)' : 'none',
                  transition: 'background-color 0.15s ease, color 0.15s ease',
                  '&:hover': { color: 'hsl(var(--foreground))' },
                }}
              >
                {tab === 'all' ? 'All chats' : `Bookmarked${bookmarks.length ? ` (${bookmarks.length})` : ''}`}
              </ButtonBase>
            );
          })}
        </Box>
        <InputBase
          autoFocus
          fullWidth
          value={historyQuery}
          onChange={(event) => setHistoryQuery(event.target.value)}
          placeholder="Search chats..."
          inputProps={{ 'aria-label': 'Search chats' }}
          sx={{ mt: 1, px: 1.25, py: 0.5, fontSize: '0.8125rem', color: 'hsl(var(--foreground))', borderBottom: '1px solid hsl(var(--border) / 0.6)' }}
        />
        <Box sx={{ mt: 0.75, maxHeight: 380, overflowY: 'auto' }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', py: 3 }}><CircularProgress size={16} sx={{ color: 'hsl(var(--muted-foreground))' }} /></Box>
          ) : shown.length === 0 ? (
            <Typography sx={{ ...mutedSx, px: 1.25, py: 2.5, textAlign: 'center' }}>{emptyText}</Typography>
          ) : sections.map((section, sectionIndex) => (section.items.length === 0 && !section.label ? null : (
            <Box key={section.label || 'all'} role="group" aria-label={section.label || undefined} sx={{ mt: sectionIndex > 0 ? 1 : 0 }}>
              {section.label && (
                <Typography sx={{ ...mutedSx, fontSize: '0.6875rem', fontWeight: 600, letterSpacing: '0.04em', textTransform: 'uppercase', px: 1.25, pt: 0.5, pb: 0.5 }}>
                  {section.label}
                </Typography>
              )}
              {section.items.length === 0 && (
                <Typography sx={{ ...mutedSx, px: 1.25, pb: 0.75 }}>
                  {query ? 'No matching chats' : historyTab === 'bookmarked' ? 'No bookmarked chats here' : 'No chats yet'}
                </Typography>
              )}
              {section.items.map((item) => {
            const current = item.id === executionId;
            return (
              <ButtonBase
                key={item.id}
                onClick={() => openChat(item.id, item.authorization, item.status)}
                aria-current={current ? 'true' : undefined}
                sx={{
                  width: '100%', display: 'flex', alignItems: 'center', gap: 1.5, px: 1.25, py: 0.875, borderRadius: '8px', textAlign: 'left',
                  bgcolor: current ? 'hsl(var(--muted))' : 'transparent',
                  '&:hover, &.Mui-focusVisible': { bgcolor: 'hsl(var(--muted) / 0.7)' },
                }}
              >
                <Typography noWrap sx={{ ...textSx, flex: 1, minWidth: 0, fontWeight: current ? 600 : 400 }}>{item.title}</Typography>
                <Typography sx={{ ...mutedSx, flexShrink: 0 }}>{shortTimeAgo(item.startedAt)}</Typography>
              </ButtonBase>
            );
              })}
            </Box>
          )))}
        </Box>
      </Popover>
    );
  };

  const stalled = stalledFor > STALL_MS;

  return (
    // contentPaddingX is plain pixels: MUI multiplies numeric spacing values by
    // the theme's 8px unit, so 15 would become 120px.
    <Box
      ref={rootRef}
      sx={{
        ...chatMotionSx,
        display: 'flex', flexDirection: 'column', width: '100%', minHeight: 0,
        height: composerContainer ? undefined : (fillHeight || undefined),
        pl: contentPaddingX ? `${contentPaddingX}px` : 0, pr: contentPaddingX ? `${contentPaddingX}px` : 0,
      }}
    >
      {headerActionsContainer
        ? createPortal(headerActions, headerActionsContainer)
        : <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 0.25 }}>{headerActions}</Box>}
      {renderHistory()}

      <Box
        ref={threadRef}
        onScroll={(e: React.UIEvent<HTMLDivElement>) => {
          const el = e.currentTarget;
          atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
        }}
        sx={{
          flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0,
          ...(composerContainer ? {} : { overflowY: 'auto', overflowX: 'hidden', overscrollBehavior: 'contain', pb: 1 }),
        }}
      >
        {!executionId && !busy && thread.length === 0 && (
          <Box sx={{ my: 'auto', py: 3, textAlign: 'center', ...enterSx }}>
            <Typography sx={{ ...textSx, fontWeight: 600 }}>Chat with the agent</Typography>
            <Typography sx={{ ...mutedSx, mt: 0.5 }}>
              Ask questions or describe a change. Each step shows as the agent works, and it asks when it needs something from you.
            </Typography>
          </Box>
        )}

        {thread.map(renderItem)}
        {pendingMessage && !pendingRecorded && renderItem({ type: 'user', text: pendingMessage, key: 'pending-message' })}
        {/* Starting a run or sending a reply waits for the agent's first LLM round, which takes a while */}
        {(busy || working) && !liveGroupKey && (busy || lastItem) && (
          <Box sx={{ my: 1, py: 0.375, ...enterSx }}>
            <Box component="span" sx={{ ...mutedSx, ...shimmerSx }}>Thinking</Box>
          </Box>
        )}
        {renderWaiting()}
        {showWorkflowPreview && (
          <WorkflowPreview
            workflowId={previewWorkflowId}
            version={`${executionId}:${workflowEdits}`}
            onSelectNode={workflowId ? focusNode : undefined}
            onShown={() => {
              const scroller = threadRef.current;
              if (scroller && !composerContainer && atBottomRef.current) {
                scroller.scrollTo({ top: scroller.scrollHeight, behavior: 'smooth' });
              }
            }}
            resolveUrl={resolveUrl}
            resolveHeaders={resolveHeaders}
          />
        )}
        {renderSummary()}
        {renderResourceLinks()}
        {renderRevertConfirm()}

        {stopped && (
          <Box sx={{ my: 1, p: 1.25, border: '1px solid hsl(var(--destructive) / 0.4)', borderRadius: '12px', ...enterSx }}>
            <Typography sx={textSx}>The agent stopped ({status.toLowerCase()}). Send a message to start a new chat.</Typography>
          </Box>
        )}

        {(stalled || pollWarning || error) && (
          <Box sx={{ my: 1, p: 1.25, border: '1px solid hsl(var(--border))', borderRadius: '12px', ...enterSx }}>
            <Typography sx={textSx}>
              {error || pollWarning || `No progress for ${formatDuration(stalledFor / 1000)}. The agent may be stuck.`}
            </Typography>
            {(stalled || pollWarning) && (
              <Box sx={{ display: 'flex', gap: 1.5, mt: 0.75 }}>
                <Button size="small" disableRipple onClick={onRefresh} sx={linkButtonSx}>Check again</Button>
                <Button size="small" disableRipple onClick={onStop} sx={linkButtonSx}>Stop the agent</Button>
              </Box>
            )}
          </Box>
        )}
        {/* '1px', not 1: MUI reads 1 as 100%, which made this a full-height blank block */}
        <Box ref={endRef} sx={{ height: '1px', flexShrink: 0 }} />
      </Box>

      {composerContainer ? createPortal(composer, composerContainer) : (
        // Outside the scrolling thread, so it never moves. The fade above it softens messages scrolling under its edge.
        <Box sx={{
          position: 'relative', flexShrink: 0, pt: 1, pb: 1.5,
          '&::before': {
            content: '""', position: 'absolute', left: 0, right: 0, top: -20, height: 20, pointerEvents: 'none',
            background: 'linear-gradient(to bottom, hsl(var(--background) / 0), hsl(var(--background)))',
          },
        }}>
          {composer}
        </Box>
      )}
    </Box>
  );
};

export default AgentChatView;
