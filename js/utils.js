/**
 * Prompt808 UI utilities — DOM builder, toast, dialog, spinner.
 */

/**
 * DOM element builder (matches ComfyUI-Manager's $el pattern).
 * @param {string} tag - Tag name, optionally with .class or #id
 * @param {Object} attrs - Attributes/properties/events
 * @param {Array|string|Node} children
 * @returns {HTMLElement}
 */
export function $el(tag, attrs = {}, children = []) {
  // Parse tag.class#id
  let tagName = tag;
  let classes = [];
  let id = null;
  const classMatch = tag.match(/\.([^.#]+)/g);
  if (classMatch) classes = classMatch.map(c => c.slice(1));
  const idMatch = tag.match(/#([^.#]+)/);
  if (idMatch) id = idMatch[1];
  tagName = tag.split(/[.#]/)[0] || "div";

  const el = document.createElement(tagName);
  if (id) el.id = id;
  if (classes.length) el.classList.add(...classes);

  for (const [key, val] of Object.entries(attrs)) {
    if (key === "style" && typeof val === "object") {
      Object.assign(el.style, val);
    } else if (key === "classList" && Array.isArray(val)) {
      el.classList.add(...val.filter(Boolean));
    } else if (key === "dataset" && typeof val === "object") {
      Object.assign(el.dataset, val);
    } else if (key.startsWith("on") && typeof val === "function") {
      el.addEventListener(key.slice(2).toLowerCase(), val);
    } else if (key === "textContent" || key === "innerHTML" || key === "value" || key === "checked" || key === "disabled" || key === "src" || key === "href" || key === "htmlFor" || key === "type" || key === "placeholder" || key === "title" || key === "alt" || key === "name" || key === "min" || key === "max" || key === "step" || key === "rows" || key === "maxLength" || key === "inputMode" || key === "accept" || key === "multiple" || key === "loading" || key === "readOnly" || key === "selected" || key === "open") {
      el[key] = val;
    } else {
      el.setAttribute(key, val);
    }
  }

  if (!Array.isArray(children)) children = [children];
  for (const child of children) {
    if (child == null || child === false) continue;
    if (typeof child === "string" || typeof child === "number") {
      el.appendChild(document.createTextNode(String(child)));
    } else if (child instanceof Node) {
      el.appendChild(child);
    }
  }

  return el;
}

// ---------------------------------------------------------------------------
// Toast notification system
// ---------------------------------------------------------------------------

let _toastContainer = null;

function _ensureToastContainer() {
  if (_toastContainer && document.body.contains(_toastContainer)) return;
  _toastContainer = $el("div.p8-toast-container");
  document.body.appendChild(_toastContainer);
}

/**
 * Show a toast notification.
 * @param {string} message
 * @param {"info"|"success"|"error"|"warning"} type
 * @param {number} duration - ms
 */
export function toast(message, type = "info", duration = 4000) {
  _ensureToastContainer();
  const el = $el("div", {
    classList: ["p8-toast", `p8-toast--${type}`],
    textContent: message,
  });
  _toastContainer.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateX(20px)";
    setTimeout(() => el.remove(), 200);
  }, duration);
}

// ---------------------------------------------------------------------------
// Modal dialog (native <dialog>)
// ---------------------------------------------------------------------------

export class Prompt808Dialog {
  constructor(title, contentEl, { onClose, width } = {}) {
    this.onClose = onClose;
    this.element = $el("dialog.p8-dialog", {}, [
      $el("div.p8-dialog__header", {}, [
        $el("span.p8-dialog__title", { textContent: title }),
        $el("button.p8-dialog__close", {
          textContent: "\u00d7",
          onClick: () => this.close(),
        }),
      ]),
      $el("div.p8-dialog__body", {}, [contentEl]),
    ]);
    if (width) this.element.style.maxWidth = width;
    // Close on backdrop click
    this.element.addEventListener("click", (e) => {
      if (e.target === this.element) this.close();
    });
    document.body.appendChild(this.element);
  }

  show() {
    this.element.showModal();
  }

  close() {
    this.element.close();
    this.element.remove();
    this.onClose?.();
  }

  /** Update body content */
  setBody(contentEl) {
    const body = this.element.querySelector(".p8-dialog__body");
    body.innerHTML = "";
    body.appendChild(contentEl);
  }
}

/**
 * Show a quick confirmation dialog. Returns a Promise<boolean>.
 */
export function confirmDialog(title, message) {
  return new Promise((resolve) => {
    const body = $el("div", {}, [
      $el("p.p8-dialog__message", { textContent: message }),
      $el("div.p8-dialog__actions", {}, [
        $el("button.p8-btn.p8-btn--secondary", {
          textContent: "Cancel",
          onClick: () => { dlg.close(); resolve(false); },
        }),
        $el("button.p8-btn.p8-btn--danger", {
          textContent: "Confirm",
          onClick: () => { dlg.close(); resolve(true); },
        }),
      ]),
    ]);
    const dlg = new Prompt808Dialog(title, body);
    dlg.show();
  });
}

// ---------------------------------------------------------------------------
// Spinner
// ---------------------------------------------------------------------------

export function spinner(size = 20) {
  const el = $el("span.p8-spinner", {
    style: { width: size + "px", height: size + "px" },
  });
  return el;
}

// ---------------------------------------------------------------------------
// Help button
// ---------------------------------------------------------------------------

export function helpButton(title, contentLines, style) {
  const btn = $el("button.p8-help-btn", {
    textContent: "?",
    title: "Help",
    onClick: () => {
      const body = $el("div.p8-help-body", {}, contentLines.map(line =>
        $el("p", { textContent: line })
      ));
      new Prompt808Dialog(title, body).show();
    },
  });
  if (style) Object.assign(btn.style, style);
  return btn;
}

// ---------------------------------------------------------------------------
// Tag list renderer
// ---------------------------------------------------------------------------

export function tagList(tags, onRemove) {
  return $el("div.p8-tags", {}, tags.map((tag, i) =>
    $el("span.p8-tag", {}, [
      tag,
      onRemove ? $el("button.p8-tag__remove", {
        textContent: "\u00d7",
        onClick: () => onRemove(tag, i),
      }) : null,
    ].filter(Boolean))
  ));
}

// ---------------------------------------------------------------------------
// Element card builder
// ---------------------------------------------------------------------------

export function elementCard(elem, { onEdit, onDelete, thumbnailBase } = {}) {
  const parts = [];

  if (elem.thumbnail) {
    parts.push($el("img.p8-elem-card__thumb", {
      src: `${thumbnailBase || "/prompt808/thumbnails"}/${elem.thumbnail}`,
      alt: elem.desc || "",
      loading: "lazy",
    }));
  }

  parts.push($el("div.p8-elem-card__header", {}, [
    $el("span.p8-elem-card__category", { textContent: elem.category }),
    elem.subject_type ? $el("span.p8-elem-card__subject", { textContent: elem.subject_type }) : null,
  ].filter(Boolean)));

  parts.push($el("p.p8-elem-card__desc", { textContent: elem.desc }));

  if (elem.tags?.length) {
    parts.push(tagList(elem.tags));
  }

  if (onEdit || onDelete) {
    const actions = [];
    if (onEdit) actions.push($el("button.p8-btn.p8-btn--small.p8-btn--secondary", {
      textContent: "Edit", onClick: () => onEdit(elem),
    }));
    if (onDelete) actions.push($el("button.p8-btn.p8-btn--small.p8-btn--danger-outline", {
      textContent: "Delete", onClick: () => onDelete(elem),
    }));
    parts.push($el("div.p8-elem-card__actions", {}, actions));
  }

  return $el("div.p8-elem-card", {}, parts);
}
