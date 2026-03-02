/**
 * Prompt808 — Photos page.
 */

import * as api from "./api.js";
import { $el, helpButton, Prompt808Dialog, confirmDialog, spinner, tagList, toast } from "./utils.js";

let _container = null;
let _photos = [];
let _loading = true;

let _gridEl, _headerCount;

export function render(container) {
  _container = container;
  container.innerHTML = "";

  const page = $el("div", {}, [
    $el("div.p8-page-header", {}, [
      $el("h2.p8-page-title", { textContent: "Photos" }),
      _headerCount = $el("span.p8-page-count"),
      helpButton("Photos", [
        "This page shows every photo you have analyzed, organized as a thumbnail grid. Each card shows the photo, its element count, detected subject type, and the categories of elements extracted from it.",
        "Click any photo to open its detail view, which shows the full-size image and a scrollable list of every element that was extracted from it — useful for reviewing or verifying what the vision model found.",
        "The delete button (x overlay on hover) removes the photo record and all of its associated elements from the library. This is a permanent action that cannot be undone.",
        "Photos are stored as thumbnails only — the original image files are not kept after analysis.",
      ], { marginLeft: "auto" }),
    ]),
    _gridEl = $el("div"),
  ]);

  container.appendChild(page);
  _fetchPhotos();
}

export function onActivated() { _fetchPhotos(); }
export function onDataVersionChanged() { _fetchPhotos(); }

async function _fetchPhotos() {
  _loading = true;
  _renderGrid();
  try {
    const data = await api.getPhotos();
    _photos = data.photos || [];
  } catch (e) {
    toast("Failed to load photos: " + e.message, "error");
  }
  _loading = false;
  _headerCount.textContent = `${_photos.length} photo${_photos.length !== 1 ? "s" : ""} analyzed`;
  _renderGrid();
}

function _renderGrid() {
  _gridEl.innerHTML = "";
  if (_loading) {
    _gridEl.appendChild($el("div.p8-center", {}, [spinner(24)]));
    return;
  }
  if (_photos.length === 0) {
    _gridEl.appendChild($el("p.p8-empty", { textContent: "No photos analyzed yet. Upload photos on the Analyze page." }));
    return;
  }

  const grid = $el("div.p8-photo-grid");
  for (const photo of _photos) {
    grid.appendChild($el("div.p8-photo-card", {
      onClick: () => _showDetail(photo),
    }, [
      $el("div.p8-photo-img-wrap", {}, [
        $el("img.p8-photo-img", {
          src: api.getThumbnailUrl(photo.thumbnail),
          alt: photo.source_photo || photo.thumbnail,
          loading: "lazy",
        }),
        $el("button.p8-photo-delete-overlay", {
          innerHTML: "&times;",
          title: "Delete photo and all its elements",
          onClick: (e) => { e.stopPropagation(); _confirmDelete(photo); },
        }),
      ]),
      $el("div.p8-photo-info", {}, [
        $el("span.p8-photo-el-count", { textContent: `${photo.element_count} element${photo.element_count !== 1 ? "s" : ""}` }),
        photo.subject_type ? $el("span.p8-photo-subject", { textContent: photo.subject_type }) : null,
      ].filter(Boolean)),
      photo.categories.length > 0 ? $el("div.p8-photo-categories", {},
        photo.categories.map(cat => $el("span.p8-photo-cat-tag", { textContent: cat }))
      ) : null,
    ].filter(Boolean)));
  }
  _gridEl.appendChild(grid);
}

async function _showDetail(photo) {
  const loadingEl = $el("div.p8-center", {}, [spinner(20)]);
  const body = $el("div", {}, [
    $el("img.p8-photo-detail-img", { src: api.getThumbnailUrl(photo.thumbnail), alt: "Photo" }),
    $el("div.p8-photo-detail-meta", {}, [
      photo.subject_type ? $el("span", { textContent: photo.subject_type, style: { color: "var(--p8-accent)", fontWeight: "500" } }) : null,
      photo.added ? $el("span", { textContent: photo.added, style: { fontSize: "12px", color: "var(--p8-text-muted)" } }) : null,
    ].filter(Boolean)),
    $el("h4", { textContent: `Elements (${photo.element_count})`, style: { fontSize: "14px", fontWeight: "600", color: "var(--p8-text-secondary)", margin: "8px 0" } }),
    loadingEl,
    $el("div", { style: { display: "flex", justifyContent: "flex-end", marginTop: "10px" } }, [
      $el("button.p8-btn.p8-btn--danger-outline", {
        textContent: "Delete Photo & Elements",
        onClick: () => { dlg.close(); _confirmDelete(photo); },
      }),
    ]),
  ]);

  const dlg = new Prompt808Dialog(
    photo.source_photo?.split(/[\\/]/).pop() || "Photo Details",
    body,
    { width: "560px" },
  );
  dlg.show();

  // Load elements
  try {
    const data = await api.getPhotoElements(photo.thumbnail);
    const elems = data.elements || [];
    const list = $el("div.p8-elem-list", {},
      elems.map(elem => $el("div.p8-elem-item", {}, [
        $el("div.p8-elem-item-header", {}, [
          $el("span.p8-elem-item-category", { textContent: elem.category }),
          $el("span.p8-elem-item-id", { textContent: elem.id }),
        ]),
        $el("p.p8-elem-item-desc", { textContent: elem.desc }),
        elem.tags?.length ? tagList(elem.tags) : null,
      ].filter(Boolean)))
    );
    loadingEl.replaceWith(list);
  } catch (e) {
    loadingEl.replaceWith($el("p", { textContent: "Failed to load elements: " + e.message, style: { color: "var(--p8-error)" } }));
  }
}

async function _confirmDelete(photo) {
  const ok = await confirmDialog(
    "Delete Photo",
    `This will permanently delete the photo and all ${photo.element_count} associated element${photo.element_count !== 1 ? "s" : ""} from your library. This cannot be undone.`,
  );
  if (!ok) return;

  try {
    const data = await api.deletePhoto(photo.thumbnail);
    toast(`Photo removed: ${data.elements_removed} element${data.elements_removed !== 1 ? "s" : ""} deleted`, "success");
    _fetchPhotos();
  } catch (e) {
    toast("Delete failed: " + e.message, "error");
  }
}
