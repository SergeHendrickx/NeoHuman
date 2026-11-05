/**
 * Application menubar: File / View / Help over the docked viewer.
 *
 * The View menu's Panels checklist is the way back for closed dock tabs
 * (the only recovery used to be Reset layout). Re-added panels land in
 * their home group via homeRefFor. Pure parts are node-tested.
 */

const HOME_GROUPS = [
  ["world", "camera", "performance", "crowd", "actors"],
  ["pose", "identity", "expression", "eyes", "mouth", "materials", "clothing", "animation", "ai"],
  ["telemetry", "crowdstats"],
];

/**
 * Reference panel for re-adding `paneId`: first open member of its home
 * group, or null when the group (or id) has nowhere to dock within.
 * isOpen: (id) => boolean.
 */
export function homeRefFor(paneId, isOpen) {
  if (paneId === "viewport") return null;
  const group = HOME_GROUPS.find((g) => g.includes(paneId));
  if (!group) return null;
  return group.find((id) => id !== paneId && isOpen(id)) ?? null;
}

export function initMenubar({ dockApi = null, panes = [], actions = {} }) {
  const bar = document.getElementById("menubar");
  if (!bar) return null;
  const menus = [...bar.querySelectorAll(".menu")];

  function closeAll(except = null) {
    for (const menu of menus) {
      if (menu === except) continue;
      menu.querySelector(".menu-pop").hidden = true;
      menu.querySelector(".menu-top").setAttribute("aria-expanded", "false");
    }
  }

  function togglePanel(id) {
    if (!dockApi) return;
    const existing = dockApi.getPanel(id);
    if (existing) {
      const openCount = panes.filter((p) => dockApi.getPanel(p.id)).length;
      if (openCount <= 1) {
        actions.setStatus?.("Cannot close the last panel.", "error");
        rebuildPanels();
        return;
      }
      existing.api.close();
      rebuildPanels();
      return;
    }
    const meta = panes.find((p) => p.id === id);
    if (!meta) return;
    const ref = homeRefFor(id, (pid) => Boolean(dockApi.getPanel(pid)))
      ?? dockApi.activePanel?.id
      ?? null;
    dockApi.addPanel({
      id: meta.id,
      title: meta.title,
      component: meta.id,
      ...(ref ? { position: { direction: "within", referencePanel: ref } } : {}),
    });
    dockApi.getPanel(id)?.api.setActive();
    rebuildPanels();
  }

  function rebuildPanels() {
    const box = document.getElementById("menu-panels");
    if (!box) return;
    box.replaceChildren();
    if (!dockApi) {
      const note = document.createElement("div");
      note.className = "quiet";
      note.textContent = "Dock unavailable (static layout).";
      box.append(note);
      return;
    }
    for (const pane of panes) {
      const label = document.createElement("label");
      label.className = "menu-check";
      const box_ = document.createElement("input");
      box_.type = "checkbox";
      box_.checked = Boolean(dockApi.getPanel(pane.id));
      box_.addEventListener("change", () => togglePanel(pane.id));
      label.append(box_, document.createTextNode(pane.title));
      box.append(label);
    }
  }

  for (const menu of menus) {
    const top = menu.querySelector(".menu-top");
    const pop = menu.querySelector(".menu-pop");
    top.addEventListener("click", () => {
      const opening = pop.hidden;
      closeAll(menu);
      pop.hidden = !opening;
      top.setAttribute("aria-expanded", String(opening));
      if (opening && menu.dataset.menu === "view") {
        rebuildPanels();
        const headerBtn = pop.querySelector('[data-act="header"]');
        if (headerBtn) headerBtn.textContent = actions.isHeaderHidden?.() ? "Show header" : "Hide header";
      }
    });
    pop.addEventListener("click", (event) => {
      const item = event.target.closest("[data-act]");
      if (!item) return;
      const act = item.dataset.act;
      if (act === "snapshot") actions.snapshot?.();
      else if (act === "fullscreen") actions.fullscreen?.();
      else if (act === "header") actions.toggleHeader?.();
      else if (act === "layout") actions.resetLayout?.();
      else if (act === "shortcuts") actions.shortcuts?.();
      else if (act === "about") actions.about?.();
      closeAll();
    });
  }

  document.addEventListener("click", (event) => {
    if (!event.target.closest("#menubar")) closeAll();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeAll();
  });

  return { closeAll, rebuildPanels };
}
