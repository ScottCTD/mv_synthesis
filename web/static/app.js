const state = {
  catalog: null,
  assets: { dataset_prefix: "/assets/dataset/" },
  sessionId: `sess_${Math.random().toString(36).slice(2, 10)}`,
  song: null,
  leftMethod: null,
  rightMethod: null,
  leftEntries: new Map(),
  rightEntries: new Map(),
  lineOrder: [],
  currentPos: 0,
  leftManifestPath: null,
  rightManifestPath: null,
  isSubmitting: false,
  username: "",
  candidateAudioEnabled: false,
  config: {
    default_left_method: null,
    default_right_method: null,
    hide_method_select: false,
    anonymize_methods: false,
  },
  candidateExpanded: {},
  currentCandidates: {},
  preloadContainer: null,
};

const elements = {
  songTitle: document.getElementById("songTitle"),
  songAudio: document.getElementById("songAudio"),
  songAudioRow: document.getElementById("songAudioRow"),
  lineText: document.getElementById("lineText"),
  lineAudio: document.getElementById("lineAudio"),
  lineAudioRow: document.getElementById("lineAudioRow"),
  leftMethodName: document.getElementById("leftMethodName"),
  rightMethodName: document.getElementById("rightMethodName"),
  leftSelected: document.getElementById("leftSelected"),
  rightSelected: document.getElementById("rightSelected"),
  leftVideoCandidates: document.getElementById("leftVideoCandidates"),
  rightVideoCandidates: document.getElementById("rightVideoCandidates"),
  leftVibeCandidates: document.getElementById("leftVibeCandidates"),
  rightVibeCandidates: document.getElementById("rightVibeCandidates"),
  progressText: document.getElementById("progressText"),
  statusText: document.getElementById("statusText"),
  startStatus: document.getElementById("startStatus"),
  toggleSession: document.getElementById("toggleSession"),
  startScreen: document.getElementById("startScreen"),
  mainContent: document.getElementById("mainContent"),
  songSelect: document.getElementById("songSelect"),
  leftMethodSelect: document.getElementById("leftMethodSelect"),
  rightMethodSelect: document.getElementById("rightMethodSelect"),
  leftMethodLabel: document.getElementById("leftMethodLabel"),
  rightMethodLabel: document.getElementById("rightMethodLabel"),
  leftMethodField: document.getElementById("leftMethodField"),
  rightMethodField: document.getElementById("rightMethodField"),
  loadSession: document.getElementById("loadSession"),
  playToggle: document.getElementById("playToggle"),
  usernameInput: document.getElementById("usernameInput"),
  candidateAudioToggle: document.getElementById("candidateAudioToggle"),
};

document.addEventListener("DOMContentLoaded", () => {
  init();
});

async function init() {
  setStatus("Loading catalog...");
  try {
    const catalog = await fetchJSON("/api/catalog");
    const config = await fetchJSON("/api/config");
    state.catalog = catalog;
    state.config = { ...state.config, ...config };
    populateSelectors(catalog);
    wireControls();
    applyInitialSelection();
    if (elements.usernameInput) {
      state.username = elements.usernameInput.value.trim();
      updateControlState();
    }
    updateCandidateAudioButton();
    applyConfigVisibility();
    showStartScreen();
  } catch (error) {
    setStatus(`Failed to load catalog: ${error.message}`);
  }
}

function wireControls() {
  elements.toggleSession.addEventListener("click", () => {
    showStartScreen();
    pauseAllMedia();
  });

  if (elements.usernameInput) {
    elements.usernameInput.addEventListener("input", () => {
      state.username = elements.usernameInput.value.trim();
      updateControlState();
    });
  }

  elements.loadSession.addEventListener("click", async () => {
    const success = await loadSession();
    if (success) {
      hideStartScreen();
    }
  });

  elements.leftMethodSelect.addEventListener("change", () => {
    updateSongOptions();
    updateControlState();
  });

  elements.rightMethodSelect.addEventListener("change", () => {
    updateSongOptions();
    updateControlState();
  });

  elements.songSelect.addEventListener("change", () => {
    updateControlState();
  });

  if (elements.playToggle) {
    elements.playToggle.addEventListener("click", () => playAllOnce());
  }

  if (elements.candidateAudioToggle) {
    elements.candidateAudioToggle.addEventListener("click", () => toggleCandidateAudio());
  }

  document.querySelectorAll(".control-buttons button").forEach((button) => {
    const decision = button.dataset.decision;
    if (!decision) return;
    button.addEventListener("click", () => handleDecision(decision));
  });

  document.addEventListener("keydown", (event) => {
    const active = document.activeElement?.tagName;
    if (active === "INPUT" || active === "SELECT" || active === "TEXTAREA") {
      return;
    }
    if (event.key === "ArrowLeft") handleDecision("left");
    if (event.key === "ArrowRight") handleDecision("right");
    if (event.key.toLowerCase() === "t") handleDecision("tie");
    if (event.key.toLowerCase() === "n") handleDecision("next");
  });
}

function ensurePreloadContainer() {
  if (state.preloadContainer) return state.preloadContainer;
  const container = document.createElement("div");
  container.className = "preload-container";
  container.setAttribute("aria-hidden", "true");
  container.style.display = "none";
  document.body.appendChild(container);
  state.preloadContainer = container;
  return container;
}

function populateSelectors(catalog) {
  elements.leftMethodSelect.innerHTML = "";
  elements.rightMethodSelect.innerHTML = "";

  const methods = Object.keys(catalog.methods);
  methods.forEach((method) => {
    const optionLeft = document.createElement("option");
    optionLeft.value = method;
    optionLeft.textContent = method;
    elements.leftMethodSelect.appendChild(optionLeft);

    const optionRight = document.createElement("option");
    optionRight.value = method;
    optionRight.textContent = method;
    elements.rightMethodSelect.appendChild(optionRight);
  });
}

function applyInitialSelection() {
  const params = new URLSearchParams(window.location.search);
  if (params.get("song")) elements.songSelect.value = params.get("song");
  if (params.get("left")) elements.leftMethodSelect.value = params.get("left");
  if (params.get("right")) elements.rightMethodSelect.value = params.get("right");

  const methods = Object.keys(state.catalog.methods);
  const defaultLeft = state.config.default_left_method;
  const defaultRight = state.config.default_right_method;
  if (!elements.leftMethodSelect.value) {
    if (defaultLeft && methods.includes(defaultLeft)) {
      elements.leftMethodSelect.value = defaultLeft;
    } else {
      elements.leftMethodSelect.value = methods[0] || "";
    }
  }
  if (!elements.rightMethodSelect.value) {
    if (defaultRight && methods.includes(defaultRight)) {
      elements.rightMethodSelect.value = defaultRight;
    } else {
      elements.rightMethodSelect.value = methods[1] || methods[0] || "";
    }
  }

  updateSongOptions();
  updateControlState();
}

function updateSongOptions() {
  const leftMethod = elements.leftMethodSelect.value;
  const rightMethod = elements.rightMethodSelect.value;
  const songs = intersectSongs(leftMethod, rightMethod);
  const current = elements.songSelect.value;

  elements.songSelect.innerHTML = "";
  songs.forEach((song) => {
    const option = document.createElement("option");
    option.value = song;
    option.textContent = song;
    elements.songSelect.appendChild(option);
  });

  if (songs.includes(current)) {
    elements.songSelect.value = current;
  } else if (songs.length) {
    elements.songSelect.value = songs[0];
  }

  if (!songs.length) {
    setStatus("No overlapping songs for the selected methods.");
  }

  updateControlState();
}

function applyConfigVisibility() {
  const anonymize =
    Boolean(state.config.anonymize_methods) || Boolean(state.config.hide_method_select);
  if (!anonymize) {
    elements.leftMethodLabel.textContent = "Left method";
    elements.rightMethodLabel.textContent = "Right method";
    elements.leftMethodSelect.disabled = false;
    elements.rightMethodSelect.disabled = false;
    return;
  }

  elements.leftMethodLabel.textContent = "Method 1";
  elements.rightMethodLabel.textContent = "Method 2";
  elements.leftMethodSelect.disabled = true;
  elements.rightMethodSelect.disabled = true;

  const leftValue = elements.leftMethodSelect.value;
  const rightValue = elements.rightMethodSelect.value;
  if (leftValue) {
    elements.leftMethodSelect.innerHTML = "";
    const option = document.createElement("option");
    option.value = leftValue;
    option.textContent = "Method 1";
    elements.leftMethodSelect.appendChild(option);
    elements.leftMethodSelect.value = leftValue;
  }
  if (rightValue) {
    elements.rightMethodSelect.innerHTML = "";
    const option = document.createElement("option");
    option.value = rightValue;
    option.textContent = "Method 2";
    elements.rightMethodSelect.appendChild(option);
    elements.rightMethodSelect.value = rightValue;
  }
}

function intersectSongs(leftMethod, rightMethod) {
  const leftSongs = new Set(state.catalog.methods[leftMethod]?.songs || []);
  const rightSongs = new Set(state.catalog.methods[rightMethod]?.songs || []);
  if (!leftSongs.size || !rightSongs.size) {
    return [];
  }
  return [...leftSongs].filter((song) => rightSongs.has(song)).sort();
}

function showStartScreen() {
  if (elements.startScreen) {
    elements.startScreen.classList.remove("is-hidden");
  }
  if (elements.mainContent) {
    elements.mainContent.classList.add("is-hidden");
  }
}

function hideStartScreen() {
  if (elements.startScreen) {
    elements.startScreen.classList.add("is-hidden");
  }
  if (elements.mainContent) {
    elements.mainContent.classList.remove("is-hidden");
  }
}

async function loadSession() {
  const song = elements.songSelect.value;
  const leftMethod = elements.leftMethodSelect.value;
  const rightMethod = elements.rightMethodSelect.value;

  if (!state.username) {
    setStatus("Enter a username to start.");
    return false;
  }
  if (!song || !leftMethod || !rightMethod) {
    setStatus("Select a song and two methods.");
    return false;
  }
  if (leftMethod === rightMethod) {
    setStatus("Left and right methods must be different.");
    return false;
  }

  updateControlState();

  setStatus("Loading manifests...");
  let leftData;
  let rightData;
  try {
    [leftData, rightData] = await Promise.all([
      fetchJSON(
        `/api/manifest?method=${encodeURIComponent(leftMethod)}&song=${encodeURIComponent(song)}`
      ),
      fetchJSON(
        `/api/manifest?method=${encodeURIComponent(rightMethod)}&song=${encodeURIComponent(song)}`
      ),
    ]);
  } catch (error) {
    setStatus(`Failed to load manifests: ${error.message}`);
    return false;
  }

  state.song = song;
  state.leftMethod = leftMethod;
  state.rightMethod = rightMethod;
  state.assets = leftData.assets || state.assets;
  state.leftEntries = mapEntries(leftData.entries || []);
  state.rightEntries = mapEntries(rightData.entries || []);
  state.leftManifestPath =
    state.catalog.methods[leftMethod]?.manifests?.[song] || null;
  state.rightManifestPath =
    state.catalog.methods[rightMethod]?.manifests?.[song] || null;

  const leftIndices = new Set(state.leftEntries.keys());
  const shared = [];
  state.rightEntries.forEach((_value, key) => {
    if (leftIndices.has(key)) shared.push(key);
  });
  shared.sort((a, b) => a - b);
  state.lineOrder = shared;
  state.currentPos = 0;

  elements.songTitle.textContent = song;
  const anonymize =
    Boolean(state.config.anonymize_methods) || Boolean(state.config.hide_method_select);
  elements.leftMethodName.textContent = anonymize ? "Method 1" : leftMethod;
  elements.rightMethodName.textContent = anonymize ? "Method 2" : rightMethod;

  if (leftData.song_audio_url) {
    elements.songAudio.src = leftData.song_audio_url;
    elements.songAudioRow.hidden = false;
  } else {
    elements.songAudioRow.hidden = true;
  }

  renderCurrentLine();
  setStatus("Ready.");
  return true;
}

function mapEntries(entries) {
  const map = new Map();
  entries.forEach((entry) => {
    if (entry && entry.index !== undefined) {
      map.set(entry.index, entry);
    }
  });
  return map;
}

function renderCurrentLine() {
  if (!state.lineOrder.length) {
    setStatus("No overlapping lyric lines found.");
    return;
  }

  resetCandidateExpansion();

  const lineIndex = state.lineOrder[state.currentPos];
  const leftEntry = state.leftEntries.get(lineIndex);
  const rightEntry = state.rightEntries.get(lineIndex);
  const lineText = leftEntry?.lyric_text || rightEntry?.lyric_text || "";
  const lineAudio = leftEntry?.line_audio_url || rightEntry?.line_audio_url || "";

  elements.lineText.textContent = lineText || "No lyric text.";
  if (lineAudio) {
    elements.lineAudio.src = lineAudio;
    elements.lineAudioRow.hidden = false;
  } else {
    elements.lineAudio.removeAttribute("src");
    elements.lineAudio.load();
    elements.lineAudioRow.hidden = true;
  }

  const leftSelectedVideo = renderSelected(elements.leftSelected, leftEntry, "left");
  const rightSelectedVideo = renderSelected(
    elements.rightSelected,
    rightEntry,
    "right"
  );

  state.currentCandidates = {
    leftVideo: leftEntry?.video_candidates || [],
    rightVideo: rightEntry?.video_candidates || [],
    leftVibe: leftEntry?.vibe_candidates || [],
    rightVibe: rightEntry?.vibe_candidates || [],
  };

  renderCandidateGroup("leftVideo", elements.leftVideoCandidates, "video", "left");
  renderCandidateGroup("rightVideo", elements.rightVideoCandidates, "video", "right");
  renderCandidateGroup("leftVibe", elements.leftVibeCandidates, "vibe", "left");
  renderCandidateGroup("rightVibe", elements.rightVibeCandidates, "vibe", "right");

  elements.progressText.textContent = `Line ${state.currentPos + 1} / ${
    state.lineOrder.length
  } (Index ${lineIndex})`;

  setupPreloadNextSelected(leftSelectedVideo, rightSelectedVideo);
}

function updateControlState() {
  const hasUser = Boolean(state.username);
  document.querySelectorAll(".control-buttons button").forEach((button) => {
    button.disabled = !hasUser;
  });
  if (elements.loadSession) {
    const leftMethod = elements.leftMethodSelect.value;
    const rightMethod = elements.rightMethodSelect.value;
    const song = elements.songSelect.value;
    const ready =
      hasUser && song && leftMethod && rightMethod && leftMethod !== rightMethod;
    elements.loadSession.disabled = !ready;
  }
}

function updateCandidateAudioButton() {
  if (!elements.candidateAudioToggle) return;
  const label = state.candidateAudioEnabled ? "Candidate audio: on" : "Candidate audio: off";
  elements.candidateAudioToggle.textContent = label;
  elements.candidateAudioToggle.setAttribute(
    "aria-pressed",
    state.candidateAudioEnabled ? "true" : "false"
  );
}

function setCandidateAudioEnabled(enabled) {
  state.candidateAudioEnabled = enabled;
  updateCandidateAudioButton();
  document.querySelectorAll("video.candidate-video").forEach((video) => {
    video.muted = !state.candidateAudioEnabled;
  });
}

function renderSelected(container, entry, side) {
  container.innerHTML = "";
  if (!entry?.selected_display_url) {
    container.innerHTML = '<div class="empty-state">No selected clip.</div>';
    return null;
  }

  const video = document.createElement("video");
  video.controls = true;
  video.preload = "metadata";
  video.playsInline = true;
  video.src = entry.selected_display_url;
  container.appendChild(video);

  const meta = document.createElement("div");
  meta.className = "candidate-meta";
  meta.appendChild(makeBadge("selected", side));
  if (entry.selected?.segment_id) {
    meta.appendChild(makeMetaItem("id", entry.selected.segment_id));
  }
  if (entry.selected?.strategy) {
    meta.appendChild(makeMetaItem("strategy", entry.selected.strategy));
  }
  container.appendChild(meta);
  return video;
}

function resetCandidateExpansion() {
  state.candidateExpanded = {
    leftVideo: false,
    rightVideo: false,
    leftVibe: false,
    rightVibe: false,
  };
}

function renderCandidateGroup(groupKey, container, source, side) {
  const candidates = state.currentCandidates[groupKey] || [];
  container.innerHTML = "";
  if (!candidates.length) {
    container.innerHTML = '<div class="empty-state">No candidates.</div>';
    return;
  }

  if (!state.candidateExpanded[groupKey]) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-toggle";
    button.textContent = `Show ${source} candidates (${candidates.length})`;
    button.addEventListener("click", () => {
      state.candidateExpanded[groupKey] = true;
      renderCandidateGroup(groupKey, container, source, side);
    });
    container.appendChild(button);
    return;
  }

  const hideButton = document.createElement("button");
  hideButton.type = "button";
  hideButton.className = "candidate-toggle";
  hideButton.textContent = `Hide ${source} candidates`;
  hideButton.addEventListener("click", () => {
    state.candidateExpanded[groupKey] = false;
    container.innerHTML = "";
    renderCandidateGroup(groupKey, container, source, side);
  });
  container.appendChild(hideButton);

  candidates.forEach((candidate) => {
    if (!candidate?.segment_path) return;

    const card = document.createElement("div");
    const hasVibe = Boolean(candidate.vibe_card);
    card.className = `candidate-card side-${side}${hasVibe ? " has-vibe" : ""}`;

    const media = document.createElement("div");
    media.className = "candidate-media";

    const video = document.createElement("video");
    video.controls = true;
    video.preload = "metadata";
    video.playsInline = true;
    video.muted = !state.candidateAudioEnabled;
    video.classList.add("candidate-video");
    video.src = `${state.assets.dataset_prefix}${candidate.segment_path}`;
    media.appendChild(video);

    const meta = document.createElement("div");
    meta.className = "candidate-meta";
    meta.appendChild(makeBadge(source, side));
    if (candidate.segment_id) meta.appendChild(makeMetaItem("id", candidate.segment_id));
    if (candidate.duration !== undefined) {
      meta.appendChild(makeMetaItem("dur", `${candidate.duration.toFixed(2)}s`));
    }
    if (candidate.video_score !== undefined && candidate.video_score !== null) {
      meta.appendChild(makeMetaItem("v", candidate.video_score.toFixed(3)));
    }
    if (
      candidate.vibe_card_score !== undefined &&
      candidate.vibe_card_score !== null
    ) {
      meta.appendChild(makeMetaItem("vibe", candidate.vibe_card_score.toFixed(3)));
    }
    media.appendChild(meta);
    card.appendChild(media);

    if (hasVibe) {
      const panel = document.createElement("pre");
      panel.className = "vibe-panel";
      panel.textContent = candidate.vibe_card;
      card.appendChild(panel);
    } else {
      const spacer = document.createElement("div");
      spacer.className = "vibe-spacer";
      spacer.setAttribute("aria-hidden", "true");
      card.appendChild(spacer);
    }

    container.appendChild(card);
  });
}

function setupPreloadNextSelected(leftVideo, rightVideo) {
  const targets = [leftVideo, rightVideo].filter(Boolean);
  if (!targets.length) {
    preloadNextSelected();
    return;
  }

  let triggered = false;
  const trigger = () => {
    if (triggered) return;
    triggered = true;
    preloadNextSelected();
  };

  let remaining = targets.length;
  const onLoaded = () => {
    remaining -= 1;
    if (remaining <= 0) {
      trigger();
    }
  };

  targets.forEach((video) => {
    video.addEventListener("loadeddata", onLoaded, { once: true });
  });

  setTimeout(trigger, 2000);
}

function preloadNextSelected() {
  const nextIndex = state.lineOrder[state.currentPos + 1];
  const container = ensurePreloadContainer();
  container.innerHTML = "";
  if (!nextIndex) {
    return;
  }
  const leftEntry = state.leftEntries.get(nextIndex);
  const rightEntry = state.rightEntries.get(nextIndex);
  const urls = [
    leftEntry?.selected_display_url,
    rightEntry?.selected_display_url,
  ].filter(Boolean);
  urls.forEach((url) => {
    const video = document.createElement("video");
    video.preload = "metadata";
    video.muted = true;
    video.src = url;
    container.appendChild(video);
  });
}

async function handleDecision(decision) {
  if (state.isSubmitting) return;
  if (!state.lineOrder.length) return;
  if (!state.username) {
    setStatus("Enter a username to enable voting.");
    return;
  }

  if (decision === "next") {
    await submitDecision("skip");
    advanceLine();
    return;
  }

  await submitDecision(decision);
  advanceLine();
}

function gatherMediaElements() {
  const media = Array.from(document.querySelectorAll("video"));
  if (elements.lineAudio?.src) {
    media.unshift(elements.lineAudio);
  }
  return media;
}

async function playAllMedia() {
  const media = gatherMediaElements();
  await Promise.all(
    media.map((item) => item.play().catch(() => null))
  );
}

function pauseAllMedia() {
  gatherMediaElements().forEach((item) => {
    item.pause();
    try {
      item.currentTime = 0;
    } catch (_error) {
      // Some media elements may not be seekable yet.
    }
  });
}

async function playAllOnce() {
  if (!state.username) {
    setStatus("Enter a username to enable playback.");
    return;
  }
  gatherMediaElements().forEach((item) => {
    try {
      item.currentTime = 0;
    } catch (_error) {
      // Some media elements may not be seekable yet.
    }
  });
  await playAllMedia();
}

function toggleCandidateAudio() {
  if (!state.username) {
    setStatus("Enter a username to enable playback.");
    return;
  }
  setCandidateAudioEnabled(!state.candidateAudioEnabled);
}

async function submitDecision(decision) {
  const lineIndex = state.lineOrder[state.currentPos];
  const leftEntry = state.leftEntries.get(lineIndex);
  const rightEntry = state.rightEntries.get(lineIndex);

  const winner =
    decision === "left"
      ? state.leftMethod
      : decision === "right"
      ? state.rightMethod
      : null;

  const record = {
    timestamp: new Date().toISOString(),
    session_id: state.sessionId,
    username: state.username,
    song: state.song,
    line_index: lineIndex,
    line_position: state.currentPos + 1,
    decision,
    winner,
    lyric_text: leftEntry?.lyric_text || rightEntry?.lyric_text || null,
    line_audio_url: leftEntry?.line_audio_url || rightEntry?.line_audio_url || null,
    song_audio_url: elements.songAudio?.src || null,
    left: {
      method: state.leftMethod,
      manifest_path: state.leftManifestPath,
      entry: leftEntry || null,
    },
    right: {
      method: state.rightMethod,
      manifest_path: state.rightManifestPath,
      entry: rightEntry || null,
    },
  };

  const payload = {
    username: state.username,
    song: state.song,
    left_method: state.leftMethod,
    right_method: state.rightMethod,
    record,
  };

  state.isSubmitting = true;
  setStatus("Saving...");
  try {
    await fetchJSON("/api/results", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: { "Content-Type": "application/json" },
    });
    setStatus("Saved.");
  } catch (error) {
    setStatus(`Save failed: ${error.message}`);
  } finally {
    state.isSubmitting = false;
  }
}

function advanceLine() {
  if (state.currentPos < state.lineOrder.length - 1) {
    state.currentPos += 1;
    renderCurrentLine();
  } else {
    setStatus("Reached end of song.");
  }
}

function makeBadge(label, side) {
  const span = document.createElement("span");
  span.className = `badge ${side}`;
  span.textContent = label;
  return span;
}

function makeMetaItem(label, value) {
  const span = document.createElement("span");
  span.textContent = `${label}: ${value}`;
  return span;
}

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json();
}

function setStatus(message) {
  if (elements.statusText) {
    elements.statusText.textContent = message;
  }
  if (elements.startStatus) {
    elements.startStatus.textContent = message;
  }
}
