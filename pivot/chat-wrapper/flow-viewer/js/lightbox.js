/**
 * Full-size image lightbox: double-click any card image to inspect it at
 * full resolution and open/download the original file.
 */
export function initLightbox() {
  if (typeof document === "undefined") return;
  let box = document.getElementById("flow-lightbox");
  if (!box) {
    box = document.createElement("div");
    box.id = "flow-lightbox";
    box.className = "flow-lightbox";
    box.hidden = true;
    box.innerHTML = `
      <div class="flow-lightbox-panel">
        <div class="flow-lightbox-bar">
          <span class="flow-lightbox-title"></span>
          <span class="flow-lightbox-actions">
            <a class="flow-lightbox-open reui-btn sm" target="_blank" rel="noopener">Open</a>
            <a class="flow-lightbox-dl reui-btn sm" download>Download</a>
            <button class="flow-lightbox-close reui-btn sm" type="button">Close</button>
          </span>
        </div>
        <img class="flow-lightbox-img" alt="">
      </div>`;
    document.body.appendChild(box);
    box.addEventListener("click", (e) => {
      if (e.target === box || e.target.closest(".flow-lightbox-close")) box.hidden = true;
    });
  }

  const open = (src, label) => {
    const img = box.querySelector(".flow-lightbox-img");
    img.src = src;
    const name = String(label || "preview").replace(/[^\w.\-]+/g, "_").slice(0, 48) || "preview";
    box.querySelector(".flow-lightbox-title").textContent = label || "Preview";
    const op = box.querySelector(".flow-lightbox-open");
    if (op) op.href = src;
    const dl = box.querySelector(".flow-lightbox-dl");
    if (dl) {
      dl.href = src;
      dl.download = /\.(png|jpe?g|webp|ply|obj|glb)$/i.test(src) ? src.split("/").pop().split("?")[0] : name + ".png";
    }
    box.hidden = false;
  };

  if (!document.__flowLightboxBound) {
    document.__flowLightboxBound = true;
    document.addEventListener("dblclick", (e) => {
      const img = e.target.closest && e.target.closest("img.flow-node-img, .inspector-visual img, .flow-lightbox-img");
      if (!img || img.classList.contains("flow-lightbox-img")) return;
      const src = img.currentSrc || img.src;
      if (!src || src.startsWith("data:image/svg")) return;
      e.preventDefault();
      e.stopPropagation();
      open(src, img.alt || "Preview");
    }, true);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") box.hidden = true;
    });
  }
}
