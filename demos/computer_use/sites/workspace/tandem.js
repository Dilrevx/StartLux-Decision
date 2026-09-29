// Tandem: a small mock team workspace for the computer-use demo.  Invitations live in localStorage so the recorder
// can check the result.
const $ = s => document.querySelector(s);
const params = new URLSearchParams(location.search);
const store = {
  get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
};
const TEAMS = {
  engineering: {name: "Engineering", n: 14}, design: {name: "Design", n: 6}, marketing: {name: "Marketing", n: 5},
  support: {name: "Support", n: 9}, finance: {name: "Finance", n: 3},
};
const DESIGN = [["Maya Chen", "maya@harborline.example", "Admin"], ["Leo Park", "leo@harborline.example", "Editor"],
  ["Ines Duarte", "ines@harborline.example", "Editor"], ["Sam Okafor", "sam@harborline.example", "Viewer"],
  ["Ruth Klein", "ruth@harborline.example", "Viewer"], ["Tom Weller", "tom@harborline.example", "Editor"]];
const PEOPLE = [["Dana Whitfield", "dana@harborline.example"], ["Dan Ortiz", "dan.ortiz@harborline.example"],
  ["Diana Moss", "diana@harborline.example"], ["Dante Rossi", "dante@harborline.example"]];

function shell(active, body) {
  const nav = [["Home", "index.html"], ["Projects", "projects.html"], ["Teams", "teams.html"], ["Calendar", "projects.html?v=calendar"],
    ["Reports", "projects.html?v=reports"], ["Settings", "settings.html"], ["Help", "settings.html?v=help"]];
  return `<div class="app"><aside class="side"><div class="brand">Tandem</div>
      ${nav.map(([l, h]) => `<a class="${l === active ? "on" : ""}" href="${h}">${l}</a>`).join("")}</aside>
    <div class="main"><div class="top"><input placeholder="Search Tandem" aria-label="Search Tandem">
      <button class="btn">New project</button><button class="icon" aria-label="Notifications">Notifications</button>
      <span class="me">Maya Chen</span></div><div class="content">${body}</div></div></div>`;
}

function home() {
  return shell("Home", `<h1>Good morning, Maya</h1><div class="cols">
    <div class="card"><h2>Recent projects</h2>${["Website refresh", "Q4 launch plan", "Mobile onboarding"]
      .map(p => `<a class="line" href="projects.html">${p}</a>`).join("")}</div>
    <div class="card"><h2>Your tasks</h2>${["Review homepage mockups", "Approve October budget", "Write launch announcement"]
      .map(p => `<a class="line" href="projects.html">${p}</a>`).join("")}</div></div>`);
}

function projects() {
  return shell("Projects", `<h1>Projects</h1><div class="card">${["Website refresh", "Q4 launch plan", "Mobile onboarding", "Brand guidelines"]
    .map(p => `<a class="line" href="projects.html">${p}</a>`).join("")}</div>`);
}

function settings() {
  return shell("Settings", `<h1>Workspace settings</h1><div class="card"><h2>General</h2>
    <label class="field">Workspace name <input value="Harborline"></label>
    <label class="field">Time zone <input value="Pacific Time (US and Canada)"></label>
    <button class="btn">Save changes</button></div>`);
}

function teams() {
  return shell("Teams", `<div class="head"><h1>Teams</h1><button class="btn">Create team</button></div><div class="card">
    ${Object.entries(TEAMS).map(([id, t]) => `<div class="trow"><div><b>${t.name}</b><div class="muted">${t.n} members</div></div>
      <a class="btn alt" href="team.html?id=${id}" aria-label="Open ${t.name}">Open</a></div>`).join("")}</div>`);
}

function team() {
  const id = params.get("id") || "design", t = TEAMS[id] || TEAMS.design;
  const invites = store.get("invites", []).filter(x => x.team === id);
  const rows = (id === "design" ? DESIGN : DESIGN.slice(0, 3)).map(([n, e, r]) => [n, e, r, ""])
    .concat(invites.map(x => [x.name, x.email, x.role, "Invited"]));
  return shell("Teams", `<div class="crumbs"><a href="teams.html">Teams</a> › ${t.name}</div>
    <div class="head"><h1>${t.name}</h1><button class="btn" id="invite">Invite people</button></div>
    <div class="tabs"><a class="on" href="#">Members</a><a href="projects.html">Projects</a><a href="settings.html">Team settings</a></div>
    <div class="card"><table><tr><th>Name</th><th>Email</th><th>Role</th><th></th></tr>
      ${rows.map(([n, e, r, s]) => `<tr><td>${n}</td><td class="muted">${e}</td><td>${r}${s ? ` <span class="tag">${s}</span>` : ""}</td>
        <td style="text-align:right"><button class="icon" aria-label="More actions for ${n}">⋯</button></td></tr>`).join("")}</table></div>
    <div class="modal" id="modal"><div class="dlg"><div class="head"><h2>Invite people to ${t.name}</h2>
        <button class="icon" id="close" aria-label="Close">×</button></div>
      <label class="field">Email or name<div class="pick"><div id="chips"></div>
        <input id="who" placeholder="Type an email or name" autocomplete="off" aria-label="Email or name" aria-expanded="false"></div></label>
      <div class="sugg" id="sugg"><div class="muted" style="padding:6px 10px">People in Harborline</div>
        ${PEOPLE.map(([n, e]) => `<a href="#" data-person="${n}|${e}">${n}, ${e}</a>`).join("")}</div>
      <div class="field">Role
        ${[["viewer", "Viewer, can view and comment"], ["editor", "Editor, can edit projects"], ["admin", "Admin, can manage the team"]]
          .map(([v, l]) => `<label class="opt"><input type="radio" name="role" value="${v}" ${v === "viewer" ? "checked" : ""}> ${l}</label>`).join("")}</div>
      <div class="actions"><button class="btn alt" id="cancel">Cancel</button><button class="btn" id="send">Send invite</button></div>
    </div></div><div class="toast" id="toast"></div>`);
}

const PAGES = {home, projects, settings, teams, team};
document.getElementById("app").innerHTML = PAGES[document.body.dataset.page]();
let picked = null;
const modal = $("#modal");
if (modal) {
  const open = () => modal.classList.add("open"), shut = () => { modal.classList.remove("open"); $("#sugg").classList.remove("open"); };
  $("#invite").addEventListener("click", open);
  $("#close").addEventListener("click", shut);
  $("#cancel").addEventListener("click", shut);
  $("#who").addEventListener("focus", () => { $("#sugg").classList.add("open"); $("#who").setAttribute("aria-expanded", "true"); });
  document.querySelectorAll("[data-person]").forEach(a => a.addEventListener("click", e => {
    e.preventDefault();
    const [n, em] = a.dataset.person.split("|");
    picked = {name: n, email: em};
    $("#chips").innerHTML = `<span class="chip">${n}</span>`;
    $("#who").value = "";
    $("#sugg").classList.remove("open");
    $("#who").setAttribute("aria-expanded", "false");
  }));
  $("#send").addEventListener("click", () => {
    if (!picked) { $("#toast").textContent = "Add at least one person."; $("#toast").classList.add("show"); return; }
    const role = document.querySelector("input[name=role]:checked").value;
    const invites = store.get("invites", []);
    invites.push({team: params.get("id") || "design", name: picked.name, email: picked.email,
                  role: role[0].toUpperCase() + role.slice(1)});
    store.set("invites", invites);
    location.href = `team.html?id=${params.get("id") || "design"}&sent=${encodeURIComponent(picked.email)}`;
  });
  if (params.get("sent")) { $("#toast").textContent = "Invitation sent to " + params.get("sent"); $("#toast").classList.add("show"); }
}
