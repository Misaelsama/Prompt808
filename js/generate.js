/**
 * Prompt808 — Generate page.
 */

import { app } from "../../scripts/app.js";
import { api as comfyApi } from "../../scripts/api.js";
import * as api from "./api.js";
import { $el, helpButton, spinner, toast } from "./utils.js";

const QUANTIZATIONS = ["FP16", "FP8", "8-bit", "4-bit"];
const ENRICHMENTS = ["Baseline", "Vivid", "Expressive", "Poetic", "Lyrical", "Freeform"];

// NSFW styles/moods hidden when Prompt808.General.NSFW setting is off
const NSFW_STYLES = new Set(["Boudoir", "Erotica"]);
const NSFW_MOODS = new Set(["Sensual", "Provocative"]);

let _container = null;
let _options = null;
let _results = [];
let _activeResult = 0;
let _loading = false;
let _cancelled = false;
let _batchProgress = "";

// Form state
let _seed = -1, _promptType = "Any", _archetypeId = "Any", _mood = "Any";
let _modelName = "None", _quantization = "FP16", _enrichment = "Vivid";
let _temperature = 0.7, _maxTokens = 1024, _keepLoaded = false;
let _prefix = "", _suffix = "", _batchCount = 1;
let _pendingUnload = false;

// DOM references
let _resultsArea, _controlsForm;

// Listen for generation results from node execution (or any server-side generation)
comfyApi.addEventListener("prompt808.generation_result", ({ detail }) => {
  if (_loading) return;  // sidebar is generating — ignore broadcast
  _results = [detail];
  _activeResult = 0;
  if (_resultsArea) _renderResults();
});

// Settings persistence
let _saveTimer = null;

function _collectSettings() {
  return {
    prompt_type: _promptType,
    archetype: _archetypeId,
    mood: _mood,
    llm_model: _modelName,
    quantization: _quantization,
    enrichment: _enrichment,
    temperature: _temperature,
    max_tokens: _maxTokens,
    keep_model_loaded: _keepLoaded,
    debug: app.ui.settings.getSettingValue("Prompt808.Troubleshooting.Debug") ?? false,
  };
}

function _saveSettingsDebounced() {
  if (_saveTimer) clearTimeout(_saveTimer);
  _saveTimer = setTimeout(() => {
    api.saveGenerateSettings(_collectSettings()).catch(() => {});
  }, 300);
}

async function _loadSavedSettings() {
  try {
    const saved = await api.getGenerateSettings();
    if (saved.prompt_type !== undefined) _promptType = saved.prompt_type;
    if (saved.archetype !== undefined) _archetypeId = saved.archetype;
    if (saved.mood !== undefined) _mood = saved.mood;
    if (saved.llm_model !== undefined) _modelName = saved.llm_model;
    if (saved.quantization !== undefined) _quantization = saved.quantization;
    if (saved.enrichment !== undefined) _enrichment = saved.enrichment;
    if (saved.temperature !== undefined) _temperature = saved.temperature;
    if (saved.max_tokens !== undefined) _maxTokens = saved.max_tokens;
    if (saved.keep_model_loaded !== undefined) _keepLoaded = saved.keep_model_loaded;
  } catch {
    // First use — no saved settings yet
  }
}

export function render(container) {
  _container = container;
  container.innerHTML = "";

  const page = $el("div", {}, [
    $el("div.p8-page-header", {}, [
      $el("h2.p8-page-title", { textContent: "Generate Prompt" }),
      helpButton("Generate", [
        "This page builds prompts from your element library. Each generation picks elements, applies an archetype template, and optionally rewrites through an LLM.",
        "Archetype selects a composition recipe (subject + lighting + mood grouping). Mood biases element selection toward a particular feeling. Prompt Type sets the style (Cinematic, Documentary, Fine Art, etc.).",
        "LLM Rewriting passes the assembled prompt through a local text model. Enrichment level controls how aggressively the model rewrites and adjusts temperature accordingly: Baseline (0.7) stays faithful, Vivid (0.8) adds sensory detail, Expressive and above give the model progressively more creative freedom up to Lyrical (1.0).",
        "Quantization sets the model precision (FP16 is full quality, FP8/8-bit/4-bit trade quality for VRAM savings). Temperature sets the base value that enrichment levels adjust from.",
        "Prefix and Suffix are prepended/appended verbatim to the final prompt — useful for LoRA trigger words or quality tags. Batch Count generates multiple variations using consecutive seeds.",
      ], { marginLeft: "auto" }),
    ]),
    $el("div.p8-gen-layout", {}, [
      _controlsForm = _buildControls(),
      _resultsArea = $el("div.p8-gen-results", {}, [
        $el("p.p8-placeholder", { textContent: "Configure options and click Generate to create a prompt." }),
      ]),
    ]),
  ]);

  container.appendChild(page);
  _initSettings();
}

async function _initSettings() {
  await _loadSavedSettings();
  await _loadOptions();
  _rebuildControls();
}

export function onActivated() {
  _loadOptions();
}

export function onDataVersionChanged() {
  _loadOptions();
}

async function _loadOptions() {
  try {
    _options = await api.getGenerateOptions();
    _rebuildSelects();
  } catch {
    // Silently ignore — selects will use defaults
  }
}

function _rebuildSelects() {
  if (!_controlsForm) return;
  const nsfw = app.ui.settings.getSettingValue("Prompt808.General.NSFW");

  const update = (name, values, current) => {
    const sel = _controlsForm.querySelector(`[data-sf="${name}"]`);
    if (!sel || !values?.length) return;
    // Filter NSFW items when setting is off
    const filtered = nsfw ? values : values.filter(v =>
      name === "promptType" ? !NSFW_STYLES.has(v) :
      name === "mood" ? !NSFW_MOODS.has(v) : true
    );
    sel.innerHTML = "";
    for (const v of filtered) {
      sel.appendChild($el("option", { value: v, textContent: v, selected: v === current }));
    }
  };

  update("promptType", _options?.prompt_types, _promptType);
  update("archetype", _options?.archetypes, _archetypeId);
  update("mood", _options?.moods, _mood);
  update("model", _options?.models, _modelName);
}

function _rebuildControls() {
  if (!_controlsForm || !_controlsForm.parentNode) return;
  const parent = _controlsForm.parentNode;
  const newForm = _buildControls();
  parent.replaceChild(newForm, _controlsForm);
  _controlsForm = newForm;
  _rebuildSelects();
}

function _settingChanged(setter) {
  return (v) => { setter(v); _saveSettingsDebounced(); };
}

function _buildControls() {
  return $el("div.p8-gen-controls", {}, [
    // Scene section
    $el("div.p8-section-label", { textContent: "Scene" }),
    _field("Seed", $el("div.p8-gen-seed-row", {}, [
      $el("input.p8-input", {
        type: "text", inputMode: "numeric", value: _seed,
        title: "-1 = randomize every generation",
        onInput: (e) => {
          const v = e.target.value;
          if (v === "" || v === "-") { _seed = v === "-" ? -1 : 0; return; }
          const n = parseInt(v, 10);
          if (!isNaN(n) && n >= -1) _seed = n;
        },
      }),
      $el("button.p8-btn.p8-btn--secondary", {
        textContent: "Randomize",
        onClick: () => {
          _seed = Math.floor(Math.random() * 1000000);
          const inp = _controlsForm.querySelector("[data-sf='seed']");
          if (inp) inp.value = _seed;
        },
      }),
    ]), "seed"),

    $el("div.p8-field-row", {}, [
      _selectField("Prompt Type", "promptType", _options?.prompt_types || ["Any"], _promptType, _settingChanged(v => _promptType = v)),
      _selectField("Archetype", "archetype", _options?.archetypes || ["Any"], _archetypeId, _settingChanged(v => _archetypeId = v)),
    ]),
    _selectField("Mood", "mood", _options?.moods || ["Any"], _mood, _settingChanged(v => _mood = v)),

    // LLM section
    $el("div.p8-section-label", { textContent: "LLM Rewriting" }),
    _selectField("Text Model", "model", _options?.models || ["None"], _modelName, _settingChanged(v => _modelName = v)),
    $el("div.p8-field-row", {}, [
      _selectField("Enrichment", "enrichment", ENRICHMENTS, _enrichment, _settingChanged(v => _enrichment = v)),
      _selectField("Quantization", "quantization", QUANTIZATIONS, _quantization, _settingChanged(v => _quantization = v)),
    ]),
    $el("div.p8-field-row", {}, [
      _numberField("Temperature", _temperature, 0.1, 1.5, 0.1, _settingChanged(v => _temperature = v)),
      _numberField("Max Tokens", _maxTokens, 128, 2048, 64, _settingChanged(v => _maxTokens = v)),
    ]),

    // Output section
    $el("div.p8-section-label", { textContent: "Output" }),
    _textField("Prefix", _prefix, "Prepended to prompt (e.g. LoRA trigger)", v => _prefix = v),
    _textField("Suffix", _suffix, "Appended to prompt (e.g. quality tags)", v => _suffix = v),
    _numberField("Batch Count", _batchCount, 1, 50, 1, v => _batchCount = Math.max(1, Math.min(50, v))),

    // Toggles
    _checkbox("Keep model loaded", _keepLoaded, _settingChanged(v => {
      _keepLoaded = v;
      if (!v) {
        if (!_loading) {
          api.unloadLLM().catch(() => {});
        } else {
          _pendingUnload = true;
        }
      }
    })),

    // Buttons
    $el("div.p8-gen-btn-row", {}, [
      $el("button.p8-btn.p8-btn--primary.p8-btn--large", {
        textContent: "Generate",
        style: { flex: "1" },
        dataset: { sf: "generateBtn" },
        onClick: _handleGenerate,
      }),
    ]),
  ]);
}

async function _handleGenerate() {
  const actualSeed = _seed === -1 ? Math.floor(Math.random() * 1000000) : _seed;
  _cancelled = false;
  _loading = true;
  _results = [];
  _activeResult = 0;
  _batchProgress = "";

  const btn = _controlsForm.querySelector("[data-sf='generateBtn']");
  btn.disabled = true;
  btn.innerHTML = "";
  btn.appendChild(spinner(14));
  btn.appendChild(document.createTextNode(" Generating..."));

  // Add stop button for batch
  let stopBtn = null;
  if (_batchCount > 1) {
    stopBtn = $el("button.p8-btn.p8-btn--secondary", {
      textContent: "Stop",
      onClick: () => { _cancelled = true; },
    });
    btn.parentElement.appendChild(stopBtn);
  }

  _resultsArea.innerHTML = "";

  try {
    for (let i = 0; i < _batchCount; i++) {
      if (_cancelled) break;
      if (_batchCount > 1) {
        btn.innerHTML = "";
        btn.appendChild(spinner(14));
        btn.appendChild(document.createTextNode(` Generating ${i + 1} / ${_batchCount}...`));
      }

      const params = {
        seed: actualSeed + i,
        archetype_id: _archetypeId,
        style: _promptType,
        mood: _mood,
        model_name: _modelName,
        quantization: _quantization,
        enrichment: _enrichment,
        temperature: _temperature,
        max_tokens: _maxTokens,
        keep_model_loaded: _keepLoaded,
        prefix: _prefix,
        suffix: _suffix,
        batch_count: 1,
        debug: app.ui.settings.getSettingValue("Prompt808.Troubleshooting.Debug") ?? false,
      };

      const data = await api.generatePrompt(params);
      _results.push(data);
      _activeResult = i;
    }
  } catch (e) {
    toast("Generation failed: " + e.message, "error");
  }

  _loading = false;
  if (stopBtn) stopBtn.remove();
  btn.disabled = false;
  btn.textContent = _batchCount > 1 ? `Generate ${_batchCount} Prompts` : "Generate";

  if (_pendingUnload) {
    _pendingUnload = false;
    api.unloadLLM().catch(() => {});
  }

  _renderResults();
}

function _renderResults() {
  _resultsArea.innerHTML = "";
  if (_results.length === 0) {
    _resultsArea.appendChild($el("p.p8-placeholder", { textContent: "No results yet." }));
    return;
  }

  // Batch tabs
  if (_results.length > 1) {
    const tabs = $el("div.p8-gen-batch-tabs", {},
      _results.map((_, i) =>
        $el("button", {
          classList: ["p8-gen-batch-tab", i === _activeResult ? "p8-gen-batch-tab--active" : ""],
          textContent: `#${i + 1}`,
          onClick: () => { _activeResult = i; _renderResults(); },
        })
      ).concat([
        $el("button.p8-btn.p8-btn--small.p8-btn--secondary", {
          textContent: "Copy All",
          style: { marginLeft: "auto" },
          onClick: () => {
            const all = _results.map((r, i) => `--- Prompt #${i + 1} ---\n${r.prompt}`).join("\n\n");
            _copyText(all, "All prompts");
          },
        }),
      ])
    );
    _resultsArea.appendChild(tabs);
  }

  const result = _results[_activeResult];
  if (!result) return;

  // Prompt card
  _resultsArea.appendChild(_resultCard("Prompt", result.prompt, "p8-prompt-text"));
  _resultsArea.appendChild(_resultCard("Negative Prompt", result.negative_prompt, "p8-negative-text"));

  // Meta
  _resultsArea.appendChild($el("div.p8-result-meta", {}, [
    $el("span", {}, ["Status: ", $el("strong", { textContent: result.status })]),
    $el("span", {}, ["Seed: ", $el("strong", { textContent: String(result.seed) })]),
    $el("span", {}, ["Archetype: ", $el("strong", { textContent: result.archetype_used })]),
    $el("span", {}, ["Elements: ", $el("strong", { textContent: String(result.elements_used?.length ?? 0) })]),
  ]));
}

function _resultCard(title, text, textClass) {
  return $el("div.p8-result-card", {}, [
    $el("div.p8-result-header", {}, [
      $el("h3", { textContent: title }),
      $el("button.p8-btn.p8-btn--small.p8-btn--secondary", {
        textContent: "Copy",
        onClick: () => _copyText(text, title),
      }),
    ]),
    $el("p", { classList: [textClass], textContent: text }),
  ]);
}

function _copyText(text, label) {
  navigator.clipboard.writeText(text).then(
    () => toast(`${label} copied`, "success"),
    () => toast("Copy failed", "error"),
  );
}

// --- Field helpers ---

function _field(label, el, sfKey) {
  const wrap = $el("div.p8-field", {}, [
    $el("label.p8-label", { textContent: label }),
    el,
  ]);
  if (sfKey) {
    const inp = el.querySelector("input") || el;
    if (inp.setAttribute) inp.dataset.sf = sfKey;
  }
  return wrap;
}

function _selectField(label, sfKey, options, current, onChange) {
  const sel = $el("select.p8-select", {
    dataset: { sf: sfKey },
    onChange: (e) => onChange(e.target.value),
  }, options.map(v => $el("option", { value: v, textContent: v, selected: v === current })));
  return $el("div.p8-field", {}, [$el("label.p8-label", { textContent: label }), sel]);
}

function _numberField(label, current, min, max, step, onChange) {
  return $el("div.p8-field", {}, [
    $el("label.p8-label", { textContent: label }),
    $el("input.p8-input", {
      type: "number", value: current, min, max, step,
      onInput: (e) => onChange(Number(e.target.value)),
    }),
  ]);
}

function _textField(label, current, placeholder, onChange) {
  return $el("div.p8-field", {}, [
    $el("label.p8-label", { textContent: label }),
    $el("input.p8-input", {
      type: "text", value: current, placeholder,
      onInput: (e) => onChange(e.target.value),
    }),
  ]);
}

function _checkbox(label, current, onChange) {
  return $el("label.p8-check", {}, [
    $el("input", {
      type: "checkbox", checked: current,
      onChange: (e) => onChange(e.target.checked),
    }),
    label,
  ]);
}
