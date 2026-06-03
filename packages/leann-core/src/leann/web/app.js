const state = {
  indexes: [],
  selectedIndex: "",
  searching: false,
};

const indexList = document.querySelector("#indexList");
const indexCount = document.querySelector("#indexCount");
const statusText = document.querySelector("#statusText");
const refreshButton = document.querySelector("#refreshButton");
const searchForm = document.querySelector("#searchForm");
const searchButton = document.querySelector("#searchButton");
const queryInput = document.querySelector("#queryInput");
const topKInput = document.querySelector("#topKInput");
const complexityInput = document.querySelector("#complexityInput");
const grepInput = document.querySelector("#grepInput");
const resultMeta = document.querySelector("#resultMeta");
const results = document.querySelector("#results");

function setStatus(text) {
  statusText.textContent = text;
}

function clearNode(node) {
  while (node.firstChild) {
    node.removeChild(node.firstChild);
  }
}

function renderMessage(kind, text) {
  const message = document.createElement("div");
  message.className = kind;
  message.textContent = text;
  return message;
}

function formatSize(sizeMb) {
  const numericSize = Number(sizeMb || 0);
  if (numericSize >= 1024) {
    return `${(numericSize / 1024).toFixed(2)} GB`;
  }
  return `${numericSize.toFixed(2)} MB`;
}

function renderIndexes() {
  clearNode(indexList);
  indexCount.textContent = String(state.indexes.length);

  if (state.indexes.length === 0) {
    indexList.appendChild(renderMessage("empty", "No indexes found."));
    return;
  }

  for (const index of state.indexes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "index-item";
    if (index.name === state.selectedIndex) {
      button.classList.add("active");
    }

    const name = document.createElement("span");
    name.className = "index-name";
    name.textContent = index.name || "Unnamed index";

    const detail = document.createElement("span");
    detail.className = "index-detail";
    detail.textContent = `${index.type || "cli"} · ${index.status || "unknown"} · ${formatSize(
      index.size_mb,
    )}`;

    button.append(name, detail);
    button.addEventListener("click", () => {
      state.selectedIndex = index.name;
      renderIndexes();
      resultMeta.textContent = `Selected ${index.name}`;
      clearNode(results);
    });
    indexList.appendChild(button);
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch {
      detail = response.statusText;
    }
    throw new Error(detail);
  }
  return response.json();
}

async function loadIndexes() {
  refreshButton.disabled = true;
  setStatus("Loading indexes...");
  clearNode(results);
  resultMeta.textContent = "";

  try {
    state.indexes = await fetchJson("/indexes");
    if (!state.indexes.some((index) => index.name === state.selectedIndex)) {
      state.selectedIndex = state.indexes[0]?.name || "";
    }
    renderIndexes();
    setStatus(state.indexes.length ? "Ready" : "No indexes in this project");
  } catch (error) {
    state.indexes = [];
    state.selectedIndex = "";
    renderIndexes();
    setStatus("Unable to load indexes");
    results.appendChild(renderMessage("error", error.message));
  } finally {
    refreshButton.disabled = false;
  }
}

function renderSearchResults(items, elapsedMs) {
  clearNode(results);
  resultMeta.textContent = `${items.length} result${items.length === 1 ? "" : "s"} · ${elapsedMs} ms`;

  if (items.length === 0) {
    results.appendChild(renderMessage("empty", "No results."));
    return;
  }

  for (const item of items) {
    const card = document.createElement("article");
    card.className = "result";

    const head = document.createElement("div");
    head.className = "result-head";

    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `Score ${Number(item.score || 0).toFixed(3)}`;

    const source = document.createElement("span");
    source.className = "source";
    source.textContent = item.metadata?.source || item.id || "";

    const snippet = document.createElement("div");
    snippet.className = "snippet";
    snippet.textContent = item.text || "";

    head.append(score, source);
    card.append(head, snippet);

    const metadataEntries = Object.entries(item.metadata || {}).filter(
      ([key]) => key !== "source",
    );
    if (metadataEntries.length > 0) {
      const metadata = document.createElement("div");
      metadata.className = "metadata";
      metadata.textContent = JSON.stringify(Object.fromEntries(metadataEntries), null, 2);
      card.appendChild(metadata);
    }

    results.appendChild(card);
  }
}

async function runSearch(event) {
  event.preventDefault();
  if (!state.selectedIndex) {
    resultMeta.textContent = "";
    clearNode(results);
    results.appendChild(renderMessage("error", "Select an index first."));
    return;
  }

  const query = queryInput.value.trim();
  if (!query || state.searching) {
    return;
  }

  state.searching = true;
  searchButton.disabled = true;
  resultMeta.textContent = "Searching...";
  clearNode(results);

  const startedAt = performance.now();
  try {
    const payload = {
      query,
      top_k: Number(topKInput.value || 5),
      complexity: Number(complexityInput.value || 64),
      use_grep: Boolean(grepInput.checked),
    };
    const items = await fetchJson(`/indexes/${encodeURIComponent(state.selectedIndex)}/search`, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderSearchResults(items, Math.round(performance.now() - startedAt));
  } catch (error) {
    resultMeta.textContent = "";
    results.appendChild(renderMessage("error", error.message));
  } finally {
    state.searching = false;
    searchButton.disabled = false;
  }
}

refreshButton.addEventListener("click", loadIndexes);
searchForm.addEventListener("submit", runSearch);

loadIndexes();
