/**
 * Prompt808 — Archetypes page.
 */

import * as api from "./api.js";
import { $el, helpButton, spinner, tagList, toast } from "./utils.js";

let _container = null;
let _archetypes = [];
let _loading = true;
let _regenerating = false;
let _expanded = {};

let _listEl, _headerCount;

export function render(container) {
  _container = container;
  container.innerHTML = "";

  const page = $el("div", {}, [
    $el("div.p8-page-header", {}, [
      $el("h2.p8-page-title", { textContent: "Archetypes" }),
      _headerCount = $el("span.p8-page-count"),
      helpButton("Archetypes", [
        "Archetypes are composition templates automatically generated from your element library. Each archetype groups a set of compatible elements — such as a subject paired with matching lighting, mood, color palette, and composition — into a reusable recipe.",
        "When generating a prompt, the generator selects an archetype to determine which elements work well together, ensuring coherent compositions rather than random element combinations.",
        "Expand any archetype to see its compatible tags (the stylistic dimensions it was built around), negative hints (things to avoid), and the specific element IDs it references.",
        "\"Regenerate All\" discards existing archetypes and rebuilds them from scratch using the current element library. Do this after adding or removing a significant number of elements. You can also delete individual archetypes that don't produce good results.",
      ]),
      $el("button.p8-btn.p8-btn--primary", {
        textContent: "Regenerate All",
        style: { marginLeft: "auto" },
        dataset: { sf: "regenBtn" },
        onClick: _handleRegenerate,
      }),
    ]),
    _listEl = $el("div"),
  ]);

  container.appendChild(page);
  _fetchArchetypes();
}

export function onActivated() { _fetchArchetypes(); }
export function onDataVersionChanged() { _fetchArchetypes(); }

async function _fetchArchetypes() {
  _loading = true;
  _renderList();
  try {
    const data = await api.getArchetypes();
    _archetypes = data.archetypes || [];
  } catch (e) {
    toast("Failed to load archetypes: " + e.message, "error");
  }
  _loading = false;
  _headerCount.textContent = `${_archetypes.length} archetype(s)`;
  _renderList();
}

async function _handleRegenerate() {
  if (!confirm("Regenerate all archetypes from current library?")) return;
  _regenerating = true;
  const btn = _container.querySelector("[data-sf='regenBtn']");
  if (btn) { btn.disabled = true; btn.innerHTML = ""; btn.appendChild(spinner(14)); btn.appendChild(document.createTextNode(" Regenerating...")); }

  try {
    const result = await api.regenerateArchetypes();
    toast(`Regenerated ${result.count} archetype(s): ${result.archetypes.join(", ")}`, "success");
    _fetchArchetypes();
  } catch (e) {
    toast("Regeneration failed: " + e.message, "error");
  }

  _regenerating = false;
  if (btn) { btn.disabled = false; btn.textContent = "Regenerate All"; }
}

function _renderList() {
  _listEl.innerHTML = "";
  if (_loading) {
    _listEl.appendChild($el("div.p8-center", {}, [spinner(24)]));
    return;
  }
  if (_archetypes.length === 0) {
    _listEl.appendChild($el("p.p8-empty", { textContent: "No archetypes yet. Analyze photos to build your library, then archetypes will be generated automatically." }));
    return;
  }

  const list = $el("div.p8-arch-list");
  for (const arch of _archetypes) {
    const isOpen = _expanded[arch.id];
    const card = $el("div.p8-arch-card", {}, [
      $el("div.p8-arch-header", {
        onClick: () => { _expanded[arch.id] = !_expanded[arch.id]; _renderList(); },
      }, [
        $el("div.p8-arch-title", {}, [
          $el("span.p8-arch-arrow", { textContent: isOpen ? "\u25BC" : "\u25B6" }),
          $el("h3", { textContent: arch.name }),
          $el("span.p8-arch-elem-count", { textContent: `${arch.element_ids?.length ?? 0} elements` }),
        ]),
        $el("button.p8-btn.p8-btn--small.p8-btn--danger-outline", {
          textContent: "Delete",
          onClick: (e) => { e.stopPropagation(); _handleDelete(arch); },
        }),
      ]),
    ]);

    if (isOpen) {
      const details = $el("div.p8-arch-details");

      // Compatible tags
      if (arch.compatible && Object.keys(arch.compatible).length > 0) {
        const section = $el("div.p8-arch-section", {}, [
          $el("h4.p8-arch-section-title", { textContent: "Compatible Tags" }),
        ]);
        for (const [key, tags] of Object.entries(arch.compatible)) {
          section.appendChild($el("div.p8-arch-tag-group", {}, [
            $el("span.p8-arch-tag-group-label", { textContent: key + ":" }),
            tagList(tags),
          ]));
        }
        details.appendChild(section);
      }

      // Negative hints
      if (arch.negative_hints?.length > 0) {
        details.appendChild($el("div.p8-arch-section", {}, [
          $el("h4.p8-arch-section-title", { textContent: "Negative Hints" }),
          tagList(arch.negative_hints),
        ]));
      }

      // Element IDs
      if (arch.element_ids?.length > 0) {
        details.appendChild($el("div.p8-arch-section", {}, [
          $el("h4.p8-arch-section-title", { textContent: "Element IDs" }),
          $el("div.p8-arch-elem-ids", {},
            arch.element_ids.map(id => $el("code.p8-arch-elem-id", { textContent: id }))
          ),
        ]));
      }

      card.appendChild(details);
    }

    list.appendChild(card);
  }
  _listEl.appendChild(list);
}

async function _handleDelete(arch) {
  if (!confirm(`Delete archetype "${arch.name}"?`)) return;
  try {
    await api.deleteArchetype(arch.id);
    toast("Archetype deleted", "success");
    _fetchArchetypes();
  } catch (e) {
    toast("Delete failed: " + e.message, "error");
  }
}
