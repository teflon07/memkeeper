// memkeeper dashboard — read-only browser over the local serve HTTP endpoint.
// Talks to POST /api with {command, payload}; the server injects the store path
// and enforces a read-only command allowlist, so this file never sees a path.

"use strict";

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    node.hidden = true;
  }, 4500);
}

// One request to the read-only API. Throws on transport or envelope error.
async function api(command, payload) {
  let response;
  try {
    response = await fetch("/api", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ command, payload: payload || {} }),
    });
  } catch (err) {
    throw new Error("cannot reach the dashboard server");
  }
  let body;
  try {
    body = await response.json();
  } catch (err) {
    throw new Error("server returned a non-JSON response");
  }
  if (!body.ok) {
    const message = body.error ? body.error.message : "request failed";
    throw new Error(message);
  }
  return body;
}

// ---- Tabs ----------------------------------------------------------------
function showView(name) {
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("is-active", t.dataset.view === name);
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.classList.toggle("is-active", v.id === "view-" + name);
  });
  if (name === "graph") {
    // Render the entire graph the first time the tab is opened; clicking a node
    // opens that entity's memories.
    if (!graph.loaded) {
      graph.loaded = true;
      loadFullGraph();
    } else if (graph.svg) {
      applySvgTransform();
    }
  }
}
document.querySelectorAll(".tab").forEach((t) => {
  t.addEventListener("click", () => showView(t.dataset.view));
});

// ---- Memory browser ------------------------------------------------------
const browser = { items: [], selected: null };

function renderMemoryItems(list, items, onClick) {
  list.innerHTML = "";
  if (!items.length) {
    list.appendChild(el("li", "empty-state", "No memories found."));
    return;
  }
  items.forEach((item) => {
    const li = el("li", "result-item");
    li.dataset.id = item.memory_id;
    const title = item.summary || firstLine(item.snippet) || item.memory_id;
    li.appendChild(el("div", "result-title", title));
    if (item.snippet) li.appendChild(el("div", "result-snippet", item.snippet));
    const tags = el("div", "result-tags");
    if (item.kind) tags.appendChild(el("span", "chip kind", item.kind));
    if (typeof item.score === "number") {
      tags.appendChild(el("span", "chip score", item.score.toFixed(3)));
    }
    (item.tags || []).slice(0, 4).forEach((tag) => {
      tags.appendChild(el("span", "chip", tag));
    });
    li.appendChild(tags);
    li.addEventListener("click", () => onClick(item, li));
    list.appendChild(li);
  });
}

function renderResultList(items) {
  const list = $("#result-list");
  if (!list) return;
  renderMemoryItems(list, items, (item, li) => selectMemory(item.memory_id, li));
}

function renderGraphMemoryList(items) {
  const list = $("#graph-memory-list");
  if (!list) return;
  renderMemoryItems(list, items, focusGraphMemory);
}

function memoryRecordToListItem(memory) {
  return {
    memory_id: memory.id,
    version_id: memory.version_id,
    space: memory.space,
    silo: memory.silo,
    scope: memory.scope,
    project: memory.project,
    kind: memory.kind,
    status: memory.status,
    confidence: memory.confidence,
    pinned: memory.pinned,
    summary: memory.summary,
    snippet: memory.content,
    tags: memory.tags || [],
    entity_key: memory.entity_key,
    claim_key: memory.claim_key,
    observed_at: memory.observed_at,
    created_at: memory.created_at,
    updated_at: memory.updated_at,
  };
}

function firstLine(text) {
  if (!text) return "";
  const nl = text.indexOf("\n");
  return nl === -1 ? text : text.slice(0, nl);
}

async function selectMemory(id, liNode) {
  document.querySelectorAll(".result-item").forEach((n) => n.classList.remove("is-selected"));
  if (liNode) liNode.classList.add("is-selected");
  const detail = $("#detail");
  if (!detail) return;
  detail.innerHTML = "";
  detail.appendChild(el("div", "detail-empty", "Loading…"));
  try {
    const body = await api("get", { id });
    renderDetail(body.result.memory);
  } catch (err) {
    detail.innerHTML = "";
    detail.appendChild(el("div", "detail-empty", "Could not load memory: " + err.message));
  }
}

function renderDetail(memory) {
  const detail = $("#detail");
  if (!detail) return;
  detail.innerHTML = "";
  detail.appendChild(el("h2", null, memory.summary || "(no summary)"));
  detail.appendChild(el("div", "detail-id", memory.id));
  detail.appendChild(el("div", "detail-content", memory.content || ""));
  const dl = el("dl");
  const rows = [
    ["kind", memory.kind],
    ["status", memory.status],
    ["space", memory.space],
    ["silo", memory.silo],
    ["scope", memory.scope],
    ["project", memory.project],
    ["entity_key", memory.entity_key],
    ["claim_key", memory.claim_key],
    ["confidence", memory.confidence],
    ["pinned", memory.pinned],
    ["observed_at", memory.observed_at],
    ["updated_at", memory.updated_at],
    ["tags", (memory.tags || []).join(", ")],
  ];
  rows.forEach(([k, v]) => {
    if (v == null || v === "") return;
    dl.appendChild(el("dt", null, k));
    dl.appendChild(el("dd", null, String(v)));
  });
  detail.appendChild(dl);
}

async function runSearch(query) {
  const meta = $("#browser-meta");
  if (meta) meta.textContent = "Searching…";
  try {
    const body = await api("search", {
      query,
      limit: 50,
      rerank: $("#opt-rerank") ? $("#opt-rerank").checked : true,
    });
    const results = body.result.results || [];
    browser.items = results;
    renderResultList(results);
    renderGraphMemoryList(results);
    const s = body.result.search || {};
    if (meta) {
      meta.textContent =
        `${results.length} result(s) · strategy ${s.strategy || "?"}` +
        (s.reranked ? " · reranked" : "") +
        ` · ${body.elapsed_ms}ms`;
    }
  } catch (err) {
    if (meta) meta.textContent = "";
    toast(err.message);
  }
}

async function loadRecent() {
  const meta = $("#browser-meta");
  if (meta) meta.textContent = "Loading recent…";
  if ($("#search-input")) $("#search-input").value = "";
  try {
    const body = await api("memory-list", { limit: 50 });
    const results = body.result.results || [];
    browser.items = results;
    renderResultList(results);
    renderGraphMemoryList(results);
    if (meta) meta.textContent = `${results.length} recent mem(s) · ${body.elapsed_ms}ms`;
  } catch (err) {
    if (meta) meta.textContent = "";
    toast(err.message);
  }
}

if ($("#search-form")) {
  $("#search-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const query = $("#search-input").value.trim();
    if (query) runSearch(query);
    else loadRecent();
  });
}
if ($("#btn-recent")) $("#btn-recent").addEventListener("click", loadRecent);

// ---- Graph view ----------------------------------------------------------
const graph = {
  svg: null,
  viewport: null,
  linkLayer: null,
  nodeLayer: null,
  nodes: [],
  links: [],
  entities: [],
  selectedKey: null,
  loaded: false,
  raf: null,
  transform: { x: 0, y: 0, scale: 1 },
  dragging: null,
  panning: null,
};

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function ensureSvg() {
  if (graph.svg) return graph.svg;
  const host = $("#cy");
  host.innerHTML = "";
  const svg = svgEl("svg", {
    class: "memory-graph-svg",
    role: "img",
    "aria-label": "interactive memory graph",
  });
  const viewport = svgEl("g", { class: "memory-graph-viewport" });
  graph.linkLayer = svgEl("g", { class: "memory-graph-links" });
  graph.nodeLayer = svgEl("g", { class: "memory-graph-nodes" });
  viewport.appendChild(graph.linkLayer);
  viewport.appendChild(graph.nodeLayer);
  svg.appendChild(viewport);
  host.appendChild(svg);
  graph.svg = svg;
  graph.viewport = viewport;

  svg.addEventListener("pointerdown", (event) => {
    if (event.target.closest(".memory-node")) return;
    graph.panning = { x: event.clientX, y: event.clientY, tx: graph.transform.x, ty: graph.transform.y };
    svg.setPointerCapture(event.pointerId);
  });
  svg.addEventListener("pointermove", (event) => {
    if (graph.panning) {
      graph.transform.x = graph.panning.tx + event.clientX - graph.panning.x;
      graph.transform.y = graph.panning.ty + event.clientY - graph.panning.y;
      applySvgTransform();
    }
    if (graph.dragging) {
      const point = screenToGraph(event.clientX, event.clientY);
      graph.dragging.fx = point.x;
      graph.dragging.fy = point.y;
      graph.dragging.x = point.x;
      graph.dragging.y = point.y;
      graph.dragging.vx = 0;
      graph.dragging.vy = 0;
      scheduleGraphTicks(28);
    }
  });
  svg.addEventListener("pointerup", (event) => {
    if (graph.dragging) {
      graph.dragging.fx = null;
      graph.dragging.fy = null;
      graph.dragging = null;
    }
    graph.panning = null;
    try {
      svg.releasePointerCapture(event.pointerId);
    } catch (_) {}
  });
  svg.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const before = screenToGraph(event.clientX, event.clientY);
      const factor = event.deltaY < 0 ? 1.12 : 0.9;
      graph.transform.scale = Math.max(0.35, Math.min(2.8, graph.transform.scale * factor));
      graph.transform.x = event.clientX - before.x * graph.transform.scale;
      graph.transform.y = event.clientY - before.y * graph.transform.scale;
      applySvgTransform();
    },
    { passive: false }
  );
  applySvgTransform();
  return svg;
}

function applySvgTransform() {
  if (!graph.viewport) return;
  const { x, y, scale } = graph.transform;
  graph.viewport.setAttribute("transform", `translate(${x} ${y}) scale(${scale})`);
}

function screenToGraph(clientX, clientY) {
  const rect = graph.svg.getBoundingClientRect();
  return {
    x: (clientX - rect.left - graph.transform.x) / graph.transform.scale,
    y: (clientY - rect.top - graph.transform.y) / graph.transform.scale,
  };
}

function resetGraphTransform() {
  if (!graph.svg) return;
  const rect = graph.svg.getBoundingClientRect();
  graph.transform = { x: rect.width * 0.5, y: rect.height * 0.5, scale: 1 };
  applySvgTransform();
}

function renderEntityList(results) {
  const list = $("#entity-list");
  if (!list) return;
  list.innerHTML = "";
  if (!results.length) {
    list.appendChild(el("li", "empty-state", "No entities yet — a fresh store's graph is empty. Entities and relationships build up from your memories (via the `dream` synthesis pipeline) or explicit entity/relationship upserts."));
    return;
  }
  results.forEach((r) => {
    const entity = r.entity || r;
    const li = el("li", "entity-item");
    li.dataset.key = entity.entity_key;
    li.appendChild(el("div", "entity-key", entity.canonical_name || entity.entity_key));
    li.appendChild(el("div", "entity-type", entity.entity_type || ""));
    li.addEventListener("click", () => {
      document.querySelectorAll(".entity-item").forEach((n) => n.classList.remove("is-selected"));
      li.classList.add("is-selected");
      loadGraph(entity.entity_key);
    });
    list.appendChild(li);
  });
}

async function searchEntities(query) {
  const meta = $("#graph-meta");
  meta.textContent = "Searching entities…";
  try {
    // The engine rejects an empty `query` string but lists all entities when
    // it is omitted — so browse-all (initial open / cleared box) omits it.
    const q = (query || "").trim();
    const body = await api("entity-search", q ? { query: q, limit: 40 } : { limit: 40 });
    const results = body.result.results || [];
    graph.entities = results;
    renderEntityList(results);
    meta.textContent = `${results.length} entit(ies) · ${body.elapsed_ms}ms`;
    const first = results[0] && (results[0].entity || results[0]);
    if (first && first.entity_key) loadGraph(first.entity_key);
  } catch (err) {
    meta.textContent = "";
    toast(err.message);
  }
}

// Load a fresh neighborhood around an entity key (replaces the graph).
async function loadGraph(entityKey) {
  $("#graph-hint").style.display = "none";
  const meta = $("#graph-meta");
  meta.textContent = "Loading neighborhood…";
  try {
    const depth = parseInt($("#graph-depth").value, 10) || 2;
    const body = await api("graph-neighbors", { entity_key: entityKey, depth, max_edges: 120 });
    graph.selectedKey = entityKey;
    const { nodes, links } = neighborhoodGraph(body.result, entityKey);
    renderSvgGraph(nodes, links, entityKey);
    meta.textContent =
      `${nodes.length} node(s), ${links.length} edge(s) · depth ${depth} · ${body.elapsed_ms}ms`;
  } catch (err) {
    meta.textContent = "";
    toast(err.message);
  }
}

// Expand an already-shown node in place (merges new nodes/edges).
async function expandEntity(entityKey) {
  try {
    const depth = parseInt($("#graph-depth").value, 10) || 2;
    const body = await api("graph-neighbors", { entity_key: entityKey, depth: 1, max_edges: 120 });
    const next = neighborhoodGraph(body.result, entityKey);
    mergeSvgGraph(next.nodes, next.links);
  } catch (err) {
    toast(err.message);
  }
}

// Render the entire connected graph up front. Large stores are capped to the
// most-connected entities so the SVG force layout stays responsive; the meta
// line reports the cap. Nodes are keyed by entity_key (graph-full has no entity ids).
const FULL_GRAPH_MAX_NODES = 300;
function memorySubjectNodes() {
  const seen = new Set();
  const nodes = [];
  browser.items.forEach((item) => {
    if (!item.entity_key || seen.has(item.entity_key)) return;
    seen.add(item.entity_key);
    nodes.push({
      entity_key: item.entity_key,
      canonical_name: item.summary || firstLine(item.snippet) || item.entity_key,
      entity_type: item.kind || "memory",
      degree: 0,
    });
  });
  return nodes;
}

function explicitLinkDegree(nodes, links) {
  const degree = new Map(nodes.map((n) => [n.entity_key, 0]));
  links.forEach((link) => {
    if (link.provisional) return;
    degree.set(link.source, (degree.get(link.source) || 0) + 1);
    degree.set(link.target, (degree.get(link.target) || 0) + 1);
  });
  return degree;
}

function tokenSet(value) {
  return new Set(String(value || "").toLowerCase().split(/[^a-z0-9]+/).filter(Boolean));
}

function tokenOverlap(a, b) {
  let score = 0;
  a.forEach((token) => {
    if (b.has(token)) score += 1;
  });
  return score;
}

function addProvisionalOrphanLinks(nodes, links) {
  if (nodes.length < 2) return links;
  const degree = explicitLinkDegree(nodes, links);
  const anchors = nodes.filter((n) => (degree.get(n.entity_key) || 0) > 0);
  const fallback = anchors[0] || nodes[0];
  const anchorTokens = anchors.map((node) => ({
    node,
    tokens: tokenSet(`${node.entity_key} ${node.canonical_name || ""}`),
    degree: degree.get(node.entity_key) || 0,
  }));
  const output = links.slice();
  nodes.forEach((node) => {
    if ((degree.get(node.entity_key) || 0) > 0 || node.entity_key === fallback.entity_key) return;
    const sourceTokens = tokenSet(`${node.entity_key} ${node.canonical_name || ""}`);
    let best = null;
    anchorTokens.forEach((anchor) => {
      const score = tokenOverlap(sourceTokens, anchor.tokens);
      if (!best || score > best.score || (score === best.score && anchor.degree > best.degree)) {
        best = { node: anchor.node, score, degree: anchor.degree };
      }
    });
    const anchor = best && best.score > 0 ? best.node : fallback;
    if (!anchor || anchor.entity_key === node.entity_key) return;
    output.push({
      source: anchor.entity_key,
      target: node.entity_key,
      weight: 0.15,
      relation_type: "visual-orphan-link",
      provisional: true,
    });
  });
  return output;
}

function normalizeFullGraph(nodes, links) {
  const keep = new Set(nodes.map((n) => n.entity_key));
  const explicit = links.filter((l) => keep.has(l.source) && keep.has(l.target));
  return { nodes, links: addProvisionalOrphanLinks(nodes, explicit) };
}

function neighborhoodGraph(result, seedKey) {
  const seed = result.seed;
  const entities = result.entities || [];
  const relationships = result.relationships || [];
  const byId = {};
  const nodes = [];
  const seen = new Set();
  const addNode = (entity) => {
    if (!entity || !entity.entity_key || seen.has(entity.entity_key)) return;
    seen.add(entity.entity_key);
    byId[entity.id] = entity;
    nodes.push({
      entity_key: entity.entity_key,
      canonical_name: entity.canonical_name || entity.entity_key,
      entity_type: entity.entity_type || "entity",
      degree: entity.entity_key === seedKey ? 4 : 1,
    });
  };
  addNode(seed);
  entities.forEach((e) => addNode(e.entity || e));
  const links = [];
  relationships.forEach((r) => {
    const rel = r.relationship || r;
    const src = byId[rel.subject_entity_id];
    const dst = byId[rel.object_entity_id];
    if (!src || !dst) return;
    links.push({
      source: src.entity_key,
      target: dst.entity_key,
      weight: rel.weight || 1,
      relation_type: rel.relation_type || "related",
      provisional: false,
    });
  });
  return normalizeFullGraph(nodes, links);
}

async function loadFullGraph() {
  $("#graph-hint").style.display = "none";
  const meta = $("#graph-meta");
  meta.textContent = "Loading graph…";
  try {
    const body = await api("graph-full", {});
    const data = body.result || {};
    let nodes = (data.nodes || []).slice().sort((a, b) => b.degree - a.degree);
    const graphNodeCount = nodes.length;
    const keepMemorySubjects = new Set(nodes.map((n) => n.entity_key));
    let addedMemorySubjects = 0;
    memorySubjectNodes().forEach((n) => {
      if (keepMemorySubjects.has(n.entity_key)) return;
      keepMemorySubjects.add(n.entity_key);
      nodes.push(n);
      addedMemorySubjects += 1;
    });
    const total = nodes.length;
    const capped = nodes.length > FULL_GRAPH_MAX_NODES;
    if (capped) nodes = nodes.slice(0, FULL_GRAPH_MAX_NODES);
    const keep = new Set(nodes.map((n) => n.entity_key));
    let links = (data.links || []).filter((l) => keep.has(l.source) && keep.has(l.target));
    const normalized = normalizeFullGraph(nodes, links);
    nodes = normalized.nodes;
    links = normalized.links;
    graph.selectedKey = null;
    renderSvgGraph(nodes, links);
    const provisionalCount = links.filter((l) => l.provisional).length;
    meta.textContent =
      `${nodes.length}${capped ? ` of ${total}` : ""} entities · ${links.length} links` +
      (addedMemorySubjects ? ` · ${addedMemorySubjects} memory subjects` : "") +
      (provisionalCount ? ` · ${provisionalCount} red orphan links` : "") +
      (graphNodeCount === 0 && addedMemorySubjects ? " · graph fallback" : "") +
      ` · click a node for its memories · ${body.elapsed_ms}ms`;
  } catch (err) {
    meta.textContent = "";
    toast(err.message);
  }
}

async function evidenceMemoriesForEntity(entityKey) {
  const body = await api("graph-context", { entity_key: entityKey, depth: 1, max_edges: 80 });
  const ids = [...new Set([...(body.result.entity_memory_ids || []), ...(body.result.evidence_memory_ids || [])])];
  const memories = [];
  for (const id of ids.slice(0, 50)) {
    try {
      const item = await api("get", { id });
      if (item.result && item.result.memory) memories.push(memoryRecordToListItem(item.result.memory));
    } catch (_) {}
  }
  return memories;
}

// Node click: show direct memories first, then relationship evidence memories.
async function entityMemories(entityKey) {
  try {
    const body = await api("memory-list", { filters: { entity_keys: [entityKey] }, limit: 50 });
    let results = body.result.results || [];
    let source = "direct";
    if (!results.length) {
      results = await evidenceMemoriesForEntity(entityKey);
      source = "evidence";
    }
    browser.items = results;
    renderResultList(results);
    renderGraphMemoryList(results);
    const label = source === "evidence" ? "evidence memor(ies)" : "memor(ies)";
    if ($("#browser-meta")) $("#browser-meta").textContent = `${results.length} ${label} for ${entityKey}`;
    $("#graph-meta").textContent = `${results.length} ${label} for ${entityKey}`;
  } catch (err) {
    toast(err.message);
  }
}

function focusGraphMemory(item, liNode) {
  document.querySelectorAll("#graph-memory-list .result-item").forEach((n) => n.classList.remove("is-selected"));
  if (liNode) liNode.classList.add("is-selected");
  if (!item.entity_key) return;
  focusSvgNode(item.entity_key);
}

function mergeSvgGraph(nodes, links) {
  const byKey = new Map(graph.nodes.map((n) => [n.entity_key, n]));
  nodes.forEach((node) => {
    if (byKey.has(node.entity_key)) return;
    byKey.set(node.entity_key, seedNodePosition(node, byKey.size));
  });
  const edgeKey = (link) => `${link.source}->${link.target}:${link.relation_type || ""}:${link.provisional ? "p" : "e"}`;
  const mergedLinks = new Map(graph.links.map((link) => [edgeKey(link), link]));
  links.forEach((link) => mergedLinks.set(edgeKey(link), link));
  renderSvgGraph(Array.from(byKey.values()), addProvisionalOrphanLinks(Array.from(byKey.values()), Array.from(mergedLinks.values())), graph.selectedKey, false);
}

function seedNodePosition(node, index, total = 12) {
  if (node.x != null && node.y != null) return node;
  const angle = (Math.PI * 2 * index) / Math.max(1, total);
  const radius = 90 + (index % 5) * 26;
  return { ...node, x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, vx: 0, vy: 0 };
}

function renderSvgGraph(nodes, links, seedKey = null, resetView = true) {
  ensureSvg();
  if (graph.raf) cancelAnimationFrame(graph.raf);
  graph.raf = null;
  graph.nodes = nodes.map((node, index) => seedNodePosition(node, index, nodes.length));
  graph.links = links;
  graph.nodeByKey = new Map(graph.nodes.map((node) => [node.entity_key, node]));
  graph.links.forEach((link) => {
    link.sourceNode = graph.nodeByKey.get(link.source);
    link.targetNode = graph.nodeByKey.get(link.target);
  });
  if (resetView) resetGraphTransform();
  drawSvgGraph(seedKey);
  scheduleGraphTicks(160);
}

function drawSvgGraph(seedKey = graph.selectedKey) {
  graph.linkLayer.innerHTML = "";
  graph.nodeLayer.innerHTML = "";
  graph.links.forEach((link, index) => {
    const line = svgEl("line", {
      class: link.provisional ? "memory-link is-provisional" : "memory-link",
      "data-index": index,
    });
    graph.linkLayer.appendChild(line);
    link.el = line;
  });
  graph.nodes.forEach((node) => {
    const group = svgEl("g", {
      class: "memory-node" + (node.entity_key === seedKey ? " is-seed" : ""),
      tabindex: "0",
      "data-key": node.entity_key,
    });
    const degree = graph.links.filter((link) => link.source === node.entity_key || link.target === node.entity_key).length;
    const radius = Math.max(6, Math.min(15, 7 + Math.sqrt(degree + Number(node.degree || 0)) * 2));
    group.appendChild(svgEl("circle", { r: radius + 8, class: "memory-node-glow" }));
    group.appendChild(svgEl("circle", { r: radius, class: "memory-node-core" }));
    const title = svgEl("title");
    title.textContent = node.canonical_name || node.entity_key;
    group.appendChild(title);
    group.addEventListener("click", () => {
      graph.selectedKey = node.entity_key;
      focusSvgNode(node.entity_key, false);
      entityMemories(node.entity_key);
    });
    group.addEventListener("dblclick", () => expandEntity(node.entity_key));
    group.addEventListener("pointerdown", (event) => {
      event.stopPropagation();
      graph.dragging = node;
      const point = screenToGraph(event.clientX, event.clientY);
      node.fx = point.x;
      node.fy = point.y;
      group.setPointerCapture(event.pointerId);
    });
    graph.nodeLayer.appendChild(group);
    node.el = group;
  });
  updateSvgPositions();
}

function updateSvgPositions() {
  graph.links.forEach((link) => {
    if (!link.el || !link.sourceNode || !link.targetNode) return;
    link.el.setAttribute("x1", link.sourceNode.x);
    link.el.setAttribute("y1", link.sourceNode.y);
    link.el.setAttribute("x2", link.targetNode.x);
    link.el.setAttribute("y2", link.targetNode.y);
  });
  graph.nodes.forEach((node) => {
    if (!node.el) return;
    node.el.setAttribute("transform", `translate(${node.x} ${node.y})`);
  });
}

function scheduleGraphTicks(count) {
  let remaining = count;
  const tick = () => {
    runForceTick();
    updateSvgPositions();
    remaining -= 1;
    if (remaining > 0) graph.raf = requestAnimationFrame(tick);
    else graph.raf = null;
  };
  if (!graph.raf) graph.raf = requestAnimationFrame(tick);
}

function runForceTick() {
  const nodes = graph.nodes;
  for (let i = 0; i < nodes.length; i += 1) {
    for (let j = i + 1; j < nodes.length; j += 1) {
      const a = nodes[i];
      const b = nodes[j];
      let dx = b.x - a.x;
      let dy = b.y - a.y;
      let distance2 = dx * dx + dy * dy || 0.01;
      const force = Math.min(6, 1000 / distance2);
      const distance = Math.sqrt(distance2);
      dx /= distance;
      dy /= distance;
      a.vx = (a.vx || 0) - dx * force;
      a.vy = (a.vy || 0) - dy * force;
      b.vx = (b.vx || 0) + dx * force;
      b.vy = (b.vy || 0) + dy * force;
    }
  }
  graph.links.forEach((link) => {
    const a = link.sourceNode;
    const b = link.targetNode;
    if (!a || !b) return;
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    const distance = Math.sqrt(dx * dx + dy * dy) || 1;
    const target = link.provisional ? 170 : 105;
    const strength = link.provisional ? 0.015 : 0.045;
    const force = (distance - target) * strength;
    dx /= distance;
    dy /= distance;
    a.vx = (a.vx || 0) + dx * force;
    a.vy = (a.vy || 0) + dy * force;
    b.vx = (b.vx || 0) - dx * force;
    b.vy = (b.vy || 0) - dy * force;
  });
  nodes.forEach((node) => {
    if (node.fx != null && node.fy != null) {
      node.x = node.fx;
      node.y = node.fy;
      return;
    }
    node.vx = ((node.vx || 0) - node.x * 0.004) * 0.86;
    node.vy = ((node.vy || 0) - node.y * 0.004) * 0.86;
    node.x += node.vx;
    node.y += node.vy;
  });
}

function focusSvgNode(entityKey, center = true) {
  const node = graph.nodeByKey && graph.nodeByKey.get(entityKey);
  if (!node || !graph.svg) return;
  graph.nodeLayer.querySelectorAll(".memory-node").forEach((elNode) => elNode.classList.remove("is-seed"));
  if (node.el) node.el.classList.add("is-seed");
  if (!center) return;
  const rect = graph.svg.getBoundingClientRect();
  graph.transform.x = rect.width / 2 - node.x * graph.transform.scale;
  graph.transform.y = rect.height / 2 - node.y * graph.transform.scale;
  applySvgTransform();
}

$("#entity-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const query = $("#entity-input").value.trim();
  searchEntities(query);
});

// ---- Boot ----------------------------------------------------------------
async function boot() {
  try {
    const body = await api("stats", {});
    const stats = body.result || {};
    $("#store-info").textContent =
      `${body.store.path} · ${stats.active_count ?? stats.memory_count ?? "?"} memories`;
  } catch (err) {
    $("#store-info").textContent = "store unavailable";
    toast("stats: " + err.message);
  }
  await loadRecent();
  showView("graph");
}
boot();
