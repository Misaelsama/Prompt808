/**
 * Prompt808 — Library page (element browser).
 */

import * as api from "./api.js";
import { $el, helpButton, Prompt808Dialog, elementCard, spinner, toast } from "./utils.js";

const PAGE_SIZE = 50;

let _container = null;
let _elements = [];
let _total = 0;
let _offset = 0;
let _categories = [];
let _counts = {};
let _activeCategory = null;
let _stats = null;
let _loading = true;

let _filtersEl, _gridEl, _paginationEl, _headerStats;

export function render(container) {
  _container = container;
  container.innerHTML = "";

  const page = $el("div", {}, [
    $el("div.p8-page-header", {}, [
      $el("h2.p8-page-title", { textContent: "Element Library" }),
      _headerStats = $el("span.p8-page-count"),
      helpButton("Element Library", [
        "This page shows every visual element extracted from your analyzed photos. Elements are the atomic building blocks — individual descriptions of subjects, lighting, colors, composition, mood, and more — that the generator assembles into prompts.",
        "Use the category filter chips to narrow by type (e.g. only Lighting elements, only Color Palette). The count next to each chip shows how many elements exist in that category.",
        "Click Edit on any card to refine its description or tags. Good descriptions produce better prompts, so editing is encouraged — especially for subjects where the vision model may have been imprecise.",
        "Click Delete to permanently remove an element. Deleting an element removes it from any archetypes that reference it.",
      ], { marginLeft: "auto" }),
    ]),
    _filtersEl = $el("div.p8-filters"),
    _gridEl = $el("div"),
    _paginationEl = $el("div"),
  ]);

  container.appendChild(page);
  _fetchAll();
}

export function onActivated() { _fetchAll(); }
export function onDataVersionChanged() { _fetchAll(); }

async function _fetchAll() {
  await Promise.all([_fetchElements(), _fetchMeta()]);
}

async function _fetchElements() {
  _loading = true;
  _renderGrid();
  try {
    const data = await api.getElements({ category: _activeCategory, offset: _offset, limit: PAGE_SIZE });
    _elements = data.elements;
    _total = data.total;
  } catch (e) {
    toast("Failed to load elements: " + e.message, "error");
  }
  _loading = false;
  _renderGrid();
  _renderPagination();
}

async function _fetchMeta() {
  try {
    const [catData, statsData] = await Promise.all([api.getCategories(), api.getStats()]);
    _categories = catData.categories || [];
    _counts = catData.counts || {};
    _stats = statsData;
    _renderFilters();
    _headerStats.textContent = _stats ? `${_stats.elements} elements across ${_stats.categories} categories` : "";
  } catch { /* silent */ }
}

function _renderFilters() {
  _filtersEl.innerHTML = "";
  // "All" chip
  _filtersEl.appendChild($el("button", {
    classList: ["p8-cat-chip", !_activeCategory ? "p8-cat-chip--active" : ""],
    textContent: `All (${_stats?.elements ?? "..."})`,
    onClick: () => { _activeCategory = null; _offset = 0; _fetchElements(); _renderFilters(); },
  }));
  for (const cat of _categories) {
    _filtersEl.appendChild($el("button", {
      classList: ["p8-cat-chip", _activeCategory === cat ? "p8-cat-chip--active" : ""],
      textContent: `${cat} (${_counts[cat] ?? 0})`,
      onClick: () => { _activeCategory = cat; _offset = 0; _fetchElements(); _renderFilters(); },
    }));
  }
}

function _renderGrid() {
  _gridEl.innerHTML = "";
  if (_loading) {
    _gridEl.appendChild($el("div.p8-center", {}, [spinner(24)]));
    return;
  }
  if (_elements.length === 0) {
    _gridEl.appendChild($el("p.p8-empty", { textContent: "No elements found. Upload photos on the Analyze page to build your library." }));
    return;
  }
  const grid = $el("div.p8-elem-grid");
  for (const elem of _elements) {
    grid.appendChild(elementCard(elem, {
      onEdit: (el) => _showEditDialog(el),
      onDelete: (el) => _handleDelete(el),
    }));
  }
  _gridEl.appendChild(grid);
}

function _renderPagination() {
  _paginationEl.innerHTML = "";
  const totalPages = Math.ceil(_total / PAGE_SIZE);
  if (totalPages <= 1) return;
  const currentPage = Math.floor(_offset / PAGE_SIZE) + 1;

  _paginationEl.appendChild($el("div.p8-pagination", {}, [
    $el("button.p8-btn.p8-btn--secondary.p8-btn--small", {
      textContent: "Prev", disabled: _offset === 0,
      onClick: () => { _offset = Math.max(0, _offset - PAGE_SIZE); _fetchElements(); },
    }),
    $el("span", { textContent: `Page ${currentPage} of ${totalPages}` }),
    $el("button.p8-btn.p8-btn--secondary.p8-btn--small", {
      textContent: "Next", disabled: _offset + PAGE_SIZE >= _total,
      onClick: () => { _offset += PAGE_SIZE; _fetchElements(); },
    }),
  ]));
}

async function _handleDelete(elem) {
  if (!confirm(`Delete element "${elem.id}"?`)) return;
  try {
    await api.deleteElement(elem.id);
    toast("Element deleted", "success");
    _fetchAll();
  } catch (e) {
    toast("Delete failed: " + e.message, "error");
  }
}

function _showEditDialog(elem) {
  let editDesc = elem.desc;
  let editTags = elem.tags.join(", ");

  const body = $el("div.p8-edit-form", {}, [
    $el("div.p8-field", {}, [
      $el("label.p8-label", { textContent: "Description" }),
      $el("textarea.p8-textarea", {
        value: editDesc, rows: 3,
        onInput: (e) => editDesc = e.target.value,
      }),
    ]),
    $el("div.p8-field", {}, [
      $el("label.p8-label", { textContent: "Tags (comma separated)" }),
      $el("input.p8-input", {
        value: editTags,
        onInput: (e) => editTags = e.target.value,
      }),
    ]),
    $el("div.p8-edit-actions", {}, [
      $el("button.p8-btn.p8-btn--secondary", {
        textContent: "Cancel",
        onClick: () => dlg.close(),
      }),
      $el("button.p8-btn.p8-btn--primary", {
        textContent: "Save",
        onClick: async () => {
          const updates = {};
          if (editDesc !== elem.desc) updates.desc = editDesc;
          const newTags = editTags.split(",").map(t => t.trim()).filter(Boolean);
          if (JSON.stringify(newTags) !== JSON.stringify(elem.tags)) updates.tags = newTags;
          if (Object.keys(updates).length === 0) { dlg.close(); return; }
          try {
            await api.updateElement(elem.id, updates);
            toast("Element updated", "success");
            dlg.close();
            _fetchElements();
          } catch (e) {
            toast("Update failed: " + e.message, "error");
          }
        },
      }),
    ]),
  ]);

  const dlg = new Prompt808Dialog("Edit Element", body);
  dlg.show();
}
