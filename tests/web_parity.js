// Vérifie que web/engine.js reproduit le moteur Python (heuristic.py) sur les cas générés par skyjo.assist.
// Usage : node tests/web_parity.js <engine.js> <cases.json> <profiles.json>
const fs = require("fs");
const [enginePath, casesPath, profilesPath] = process.argv.slice(2);
const E = require(require("path").resolve(enginePath));
const cases = JSON.parse(fs.readFileSync(casesPath, "utf8"));
const profiles = JSON.parse(fs.readFileSync(profilesPath, "utf8"));
const TOL = 1e-9, TIE = 1e-5;
let checks = 0;
const errors = [];
const near = (a, b, tol = TOL) => (a === null && b === null) || Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
const key = (o) => `${o.kind}:${o.pos[0]},${o.pos[1]}`;

function expect(cond, msg) { checks++; if (!cond && errors.length < 25) errors.push(msg); }

function compareOptions(label, py, js) {
  const m = new Map(py.map((o) => [key(o), o.value]));
  expect(py.length === js.length, `${label}: ${py.length} options Python, ${js.length} JS`);
  for (const o of js) expect(m.has(key(o)) && near(m.get(key(o)), o.value), `${label}: option ${key(o)} JS ${o.value} Python ${m.get(key(o))}`);
}

function compareCtx(label, py, js) {
  for (const k of ["hidden_value", "hidden_place", "my_est", "opp_best", "tempo", "unknown"])
    expect(near(py[k], js[k]), `${label}: ctx.${k} Python ${py[k]} JS ${js[k]}`);
  expect(py.last_turn === js.last_turn, `${label}: dernier tour`);
}

cases.forEach((c, i) => {
  for (const [name, py] of Object.entries(c.expected)) {
    const H = new E.Heuristic(profiles[name]);
    const label = `cas ${i} / ${name}`;
    if (c.drawn === null) {
      const view = E.buildView({ me: c.me, opponents: c.opponents, discard: c.discard });
      const js = H.discardChoice(view);
      compareCtx(label, py.ctx, js.ctx);
      compareOptions(label, py.options, js.options);
      expect(py.blocked === js.blocked, `${label}: blocked`);
      if (py.best) expect(near(py.best.total, js.best.total), `${label}: meilleur placement ${py.best.total} vs ${js.best.total}`);
      if (py.draw_ev !== null) expect(near(py.draw_ev, js.draw_ev, TIE), `${label}: draw_ev ${py.draw_ev} vs ${js.draw_ev}`);
      const margin = py.draw_ev !== null ? (py.best ? py.best.total - py.draw_ev : -1) : (py.best ? py.best.total - profiles[name].discard_min_gain : -1);
      if (Math.abs(margin) > TIE) expect(py.take === js.take, `${label}: prendre la défausse Python ${py.take} JS ${js.take}`);
      if (py.take && js.take) {
        const v2 = E.buildView({ me: c.me, opponents: c.opponents, discard: null, inHand: c.discard });
        const p2 = H.placeDiscard(v2, c.discard);
        expect(near(py.placement.chosen.total, p2.chosen.total), `${label}: pose de la défausse ${py.placement.chosen.total} vs ${p2.chosen.total}`);
      }
    } else {
      const view = E.buildView({ me: c.me, opponents: c.opponents, discard: c.discard, inHand: c.drawn });
      const js = H.drawn(view, c.drawn);
      compareCtx(label, py.ctx, js.ctx);
      compareOptions(label, py.options, js.options);
      if (py.place && js.place && !py.forced) expect(near(py.place.total, js.place.total), `${label}: placement ${py.place.total} vs ${js.place.total}`);
      if (py.reveal && js.reveal) expect(near(py.reveal.total, js.reveal.total), `${label}: retournement ${py.reveal.total} vs ${js.reveal.total}`);
      if (py.place && py.reveal && Math.abs(py.place.total - py.reveal.total) > TIE)
        expect(py.replace === js.replace, `${label}: poser/retourner Python ${py.replace} JS ${js.replace}`);
    }
  }
});
console.log(`${checks} vérifications, ${errors.length} écart(s)`);
if (errors.length) { console.log(errors.join("\n")); process.exit(1); }
