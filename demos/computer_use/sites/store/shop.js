// Kestrel Market: a small mock store for the computer-use demo.  Everything runs in the page; cart and order live in
// localStorage so the recorder can check the final order.
const CATALOG = [
  {id: "p1", t: "Voltmax AA Alkaline Batteries, 24-pack", p: 14.99, fee: 0, r: 4.5, n: "2,947", pack: 24, fam: "voltmax-aa"},
  {id: "p2", t: "Duracore AA Alkaline Batteries, 8-pack", p: 7.29, fee: 0, r: 4.8, n: "5,102", pack: 8, fam: "duracore-aa"},
  {id: "p3", t: "Voltmax AA Alkaline Batteries, 8-pack", p: 5.99, fee: 4.99, r: 4.5, n: "2,311", pack: 8, fam: "voltmax-aa"},
  {id: "p4", t: "Everlite AAA Alkaline Batteries, 8-pack", p: 5.79, fee: 0, r: 4.4, n: "932", pack: 8, fam: "everlite-aaa"},
  {id: "p5", t: "Everlite AA Alkaline Batteries, 8-pack", p: 6.49, fee: 0, r: 4.6, n: "1,874", pack: 8, fam: "everlite-aa"},
  {id: "p6", t: "Duracore AA Rechargeable Batteries, 4-pack with charger", p: 24.99, fee: 0, r: 4.7, n: "3,020", pack: 4, fam: "duracore-rc", rc: true},
  {id: "p7", t: "Everlite AA Alkaline Batteries, 4-pack", p: 3.49, fee: 4.99, r: 4.5, n: "611", pack: 4, fam: "everlite-aa"},
  {id: "p8", t: "Everlite AA Lithium Batteries, 8-pack", p: 12.49, fee: 0, r: 4.8, n: "744", pack: 8, fam: "everlite-aa-li"},
  {id: "h1", t: "Quietline Wireless Headphones", p: 89.00, fee: 0, r: 4.6, n: "4,410"},
  {id: "h2", t: "Brightway LED Bulbs, 6-pack", p: 12.99, fee: 0, r: 4.7, n: "8,025"},
  {id: "h3", t: "Oakmill Burr Coffee Grinder", p: 49.00, fee: 0, r: 4.5, n: "1,203"},
];
const byId = Object.fromEntries(CATALOG.map(x => [x.id, x]));
const $ = s => document.querySelector(s);
const money = v => "$" + v.toFixed(2);
const params = new URLSearchParams(location.search);
const store = {
  get: (k, d) => { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch (e) { return d; } },
  set: (k, v) => localStorage.setItem(k, JSON.stringify(v)),
};
const cartCount = () => store.get("cart", []).reduce((a, x) => a + x.qty, 0);
const delivery = x => x.fee ? `<span class="fee">Delivery ${money(x.fee)}</span>` : `<span class="free">Free delivery</span>`;

function header(withCats) {
  return `<header><div class="bar">
    <a class="logo" href="index.html">Kestrel <span>Market</span></a>
    <div class="search" id="search"><input id="q" placeholder="Search Kestrel Market" aria-label="Search" aria-expanded="false" autocomplete="off">
      <div class="sugg">${["AA batteries", "AAA batteries", "rechargeable batteries", "phone charger", "LED bulbs"]
        .map(s => `<a href="search.html?q=${encodeURIComponent(s)}">${s}</a>`).join("")}</div></div>
    <nav class="links"><a href="search.html?q=deals">Deals</a><a href="search.html?q=orders">Orders</a>
      <a href="cart.html">Cart (${cartCount()})</a></nav></div>
    ${withCats ? `<div class="cats">${["Electronics", "Home & Kitchen", "Garden", "Toys", "Grocery", "Office"]
      .map(c => `<a href="search.html?q=${encodeURIComponent(c.toLowerCase())}">${c}</a>`).join("")}</div>` : ""}
  </header>`;
}

function wireSearch() {
  const box = $("#search"), q = $("#q");
  q.addEventListener("focus", () => { box.classList.add("open"); q.setAttribute("aria-expanded", "true"); });
  q.addEventListener("keydown", e => { if (e.key === "Enter") location.href = "search.html?q=" + encodeURIComponent(q.value); });
  document.addEventListener("click", e => { if (!box.contains(e.target)) { box.classList.remove("open"); q.setAttribute("aria-expanded", "false"); } });
}

function home() {
  const feat = ["h1", "h2", "h3", "p4"].map(id => byId[id]);
  return header(true) + `<main>
    <div class="hero"><div><h1>Autumn deals are here</h1><div class="muted">Up to 30% off home essentials this week.</div></div>
      <a class="btn" href="search.html?q=deals">Shop deals</a></div>
    <h2 style="font-size:17px;margin:22px 0 0">Popular right now</h2>
    <div class="grid">${feat.map(x => `<div class="card"><div class="img"></div>
      <a class="ttl" href="product.html?id=${x.id}">${x.t}</a>
      <div class="price">${money(x.p)}</div>${delivery(x)}
      <div style="margin-top:8px"><button class="btn alt" data-add="${x.id}">Add to cart</button></div></div>`).join("")}</div></main>`;
}

function search() {
  const q = (params.get("q") || "").toLowerCase();
  let rows = CATALOG.filter(x => x.id.startsWith("p"));
  if (q.includes("aaa")) rows = rows.filter(x => x.t.includes("AAA"));
  else if (q.includes("recharge")) rows = rows.filter(x => x.rc);
  else if (!q.includes("batter")) rows = [];
  const free = params.get("free") === "1", pack = params.get("pack"), sort = params.get("sort") || "featured";
  if (free) rows = rows.filter(x => !x.fee);
  if (pack) rows = rows.filter(x => String(x.pack) === pack);
  if (sort === "price-asc") rows.sort((a, b) => a.p - b.p);
  if (sort === "price-desc") rows.sort((a, b) => b.p - a.p);
  if (sort === "rating") rows.sort((a, b) => b.r - a.r);
  const link = (k, v) => { const u = new URLSearchParams(params); if (u.get(k) === v) u.delete(k); else u.set(k, v); return "search.html?" + u; };
  const box = (k, v, label) => `<label><input type="checkbox" ${params.get(k) === v ? "checked" : ""} onchange="location.href='${link(k, v)}'"> ${label}</label>`;
  return header(false) + `<main><div class="layout"><aside class="filters">
      <h3>Delivery</h3>${box("free", "1", "Free delivery")}
      <h3>Pack size</h3>${box("pack", "4", "4-pack")}${box("pack", "8", "8-pack")}${box("pack", "24", "24-pack")}</aside>
    <section><div class="toolbar"><span>${rows.length} results for "${params.get("q") || ""}"</span>
      <label>Sort by <select id="sort" aria-label="Sort by">${[["featured", "Featured"], ["price-asc", "Price: low to high"],
        ["price-desc", "Price: high to low"], ["rating", "Customer rating"]]
        .map(([v, l]) => `<option value="${v}" ${v === sort ? "selected" : ""}>${l}</option>`).join("")}</select></label></div>
      ${rows.length ? rows.map(x => `<div class="item"><div class="img"></div>
        <div><a class="ttl" href="product.html?id=${x.id}">${x.t}</a><div class="muted">★ ${x.r} (${x.n})</div></div>
        <div class="right"><div class="price">${money(x.p)}</div>${delivery(x)}</div></div>`).join("")
        : `<div class="panel">No results. Try another search.</div>`}
    </section></div></main>`;
}

function product() {
  const x = byId[params.get("id")] || byId.p5;
  const sib = CATALOG.filter(y => y.fam && y.fam === x.fam).sort((a, b) => a.pack - b.pack);
  return header(true) + `<main><div class="crumbs"><a href="index.html">Home</a> › <a href="search.html?q=AA%20batteries">Batteries</a></div>
    <div class="pp"><div class="img"></div><div>
      <h1 style="font-size:22px;margin:0 0 6px">${x.t}</h1><div class="muted">★ ${x.r} (${x.n} ratings)</div>
      <div class="price" style="font-size:22px;margin-top:10px">${money(x.p)}</div>${delivery(x)}
      ${sib.length > 1 ? `<div class="packs">${sib.map(y => `<button class="pack ${y.id === x.id ? "on" : ""}" aria-pressed="${y.id === x.id}" onclick="location.href='product.html?id=${y.id}'">${y.pack}-pack</button>`).join("")}</div>` : `<div style="height:12px"></div>`}
      <div class="row"><button class="btn" data-add="${x.id}" data-go="cart.html">Add to cart</button>
        <button class="btn alt" data-add="${x.id}" data-go="checkout.html">Buy now</button></div>
    </div></div></main>`;
}

function cart() {
  const items = store.get("cart", []);
  const sub = items.reduce((a, it) => a + byId[it.id].p * it.qty, 0);
  return header(false) + `<main><div class="panel"><h2>Your cart</h2>
    ${items.length ? items.map(it => `<div class="item" style="grid-template-columns:84px 1fr 160px"><div class="img"></div>
      <div><a class="ttl" href="product.html?id=${it.id}">${byId[it.id].t}</a><div class="muted">Quantity ${it.qty}</div></div>
      <div class="right"><div class="price">${money(byId[it.id].p * it.qty)}</div><a href="#" data-remove="${it.id}">Remove</a></div></div>`).join("")
      : `<div class="muted">Your cart is empty.</div>`}
    <div class="row" style="justify-content:space-between"><a href="index.html">Continue shopping</a>
      <span>Subtotal <b>${money(sub)}</b> <a class="btn" href="checkout.html" style="margin-left:12px">Proceed to checkout</a></span></div></div></main>`;
}

function checkout() {
  const items = store.get("cart", []);
  const radio = (name, v, label, on) => `<label class="opt"><input type="radio" name="${name}" value="${v}" ${on ? "checked" : ""}> ${label}</label>`;
  return header(false) + `<main><div class="layout" style="grid-template-columns:1fr 300px"><section>
    <div class="panel"><h2>Shipping address</h2>
      ${radio("addr", "work", "Work: 400 Pine Street, Seattle, WA 98101", true)}
      ${radio("addr", "home", "Home: 12 Harbor Lane, Portland, OR 97209", false)}
      <a href="#">Add a new address</a></div>
    <div class="panel"><h2>Delivery speed</h2>
      ${radio("speed", "standard", "Standard, free, arrives Oct 3 to Oct 5", true)}
      ${radio("speed", "express", "Express, $9.99, arrives tomorrow", false)}</div>
    <div class="panel"><h2>Payment</h2>${radio("pay", "visa", "Visa ending in 4242", true)}</div></section>
    <aside class="panel" style="align-self:start"><h2>Order summary</h2>
      ${items.map(it => `<div class="muted">${byId[it.id].t} × ${it.qty}</div>`).join("")}
      <button class="btn" id="place" style="width:100%;margin-top:14px">Place order</button>
      <div style="margin-top:10px"><a href="cart.html">Back to cart</a></div></aside></div></main>`;
}

function done() {
  const o = store.get("order", null);
  return header(false) + `<main><div class="ok"><h2 style="margin:0 0 6px">Thank you, your order is placed.</h2>
    ${o ? o.items.map(it => `<div>${byId[it.id].t} × ${it.qty}, ${money(byId[it.id].p * it.qty)}</div>`).join("") +
      `<div class="muted" style="margin-top:6px">Shipping to ${o.addressLabel}. ${o.speedLabel}.</div>` : ""}</div></main>`;
}

const PAGES = {home, search, product, cart, checkout, done};
document.getElementById("app").innerHTML = PAGES[document.body.dataset.page]();
wireSearch();
document.querySelectorAll("[data-add]").forEach(b => b.addEventListener("click", () => {
  const items = store.get("cart", []), id = b.dataset.add, it = items.find(x => x.id === id);
  if (it) it.qty += 1; else items.push({id, qty: 1});
  store.set("cart", items);
  location.href = b.dataset.go || "cart.html";
}));
document.querySelectorAll("[data-remove]").forEach(a => a.addEventListener("click", e => {
  e.preventDefault();
  store.set("cart", store.get("cart", []).filter(x => x.id !== a.dataset.remove));
  location.reload();
}));
const sortSel = $("#sort");
if (sortSel) sortSel.addEventListener("change", () => { const u = new URLSearchParams(params); u.set("sort", sortSel.value); location.href = "search.html?" + u; });
const place = $("#place");
if (place) place.addEventListener("click", () => {
  const addr = document.querySelector("input[name=addr]:checked"), speed = document.querySelector("input[name=speed]:checked");
  store.set("order", {items: store.get("cart", []), address: addr.value, speed: speed.value,
    addressLabel: addr.parentElement.textContent.trim(), speedLabel: speed.parentElement.textContent.trim()});
  location.href = "done.html";
});
