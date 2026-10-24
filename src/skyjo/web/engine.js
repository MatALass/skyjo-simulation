/* Moteur de décision de Skyjo — portage JavaScript de skyjo/strategies/heuristic.py.
 *
 * Python reste la référence : les profils viennent de PROFILES (exportés depuis Python au build) et
 * tests/test_web_engine.py vérifie, sur des centaines de situations aléatoires, que ce moteur calcule
 * les mêmes valeurs que heuristic.py. Toute modification de l'un doit être reportée dans l'autre.
 *
 * Plateau (grid) : tableau de 4 colonnes ; colonne = null (retirée) ou [v0, v1, v2] avec v = nombre
 * (carte visible) ou null (carte cachée). Position = [colonne, ligne].
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.SkyjoEngine = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const ROWS = 3, COLS = 4;
  const DISTRIBUTION = { "-2": 5, "-1": 10, "0": 15 };
  for (let v = 1; v <= 12; v++) DISTRIBUTION[String(v)] = 10;
  const VALUES = Object.keys(DISTRIBUTION).map(Number).sort((a, b) => a - b);

  // ------------------------------------------------------------------ croyance sur les cartes inconnues
  function makeBelief(counts) {
    const c = {};
    let total = 0, s1 = 0, s2 = 0;
    for (const v of VALUES) {
      const n = counts[v] || 0;
      if (n < 0) throw new Error("Trop de cartes de valeur " + v);
      if (n > 0) { c[v] = n; total += n; s1 += v * n; s2 += v * v * n; }
    }
    let expected = 0, variance = 0;
    if (total) {
      expected = s1 / total;
      variance = Math.max(s2 / total - expected * expected, 0);
    } else {
      expected = NAIVE_EXPECTED;
    }
    return {
      total, expected, variance,
      prob: (v) => (total ? (c[v] || 0) / total : 0),
      // même ordre d'itération que le dict Python (valeurs croissantes de la distribution d'origine)
      items: () => VALUES.filter((v) => c[v]).map((v) => [v, c[v] / total]),
    };
  }
  const NAIVE_EXPECTED = VALUES.reduce((a, v) => a + v * DISTRIBUTION[v], 0) / 150;
  const NAIVE = makeBelief(Object.fromEntries(VALUES.map((v) => [v, DISTRIBUTION[v]])));

  // ------------------------------------------------------------------ plateaux
  const positions = (g) => {
    const out = [];
    g.forEach((col, c) => { if (col) for (let r = 0; r < ROWS; r++) out.push([c, r]); });
    return out;
  };
  const peek = (g, [c, r]) => g[c][r];
  const columnOthers = (g, [c, r]) => g[c].filter((_, i) => i !== r);
  const visibleScore = (g) => positions(g).reduce((a, p) => a + (peek(g, p) ?? 0), 0);
  const hiddenCount = (g) => positions(g).filter((p) => peek(g, p) === null).length;
  const visibleValues = (g) => positions(g).map((p) => peek(g, p)).filter((v) => v !== null);
  const same = (a, b) => a[0] === b[0] && a[1] === b[1];

  /** Vue d'une décision. `inHand` : carte piochée (déjà sortie du paquet). */
  function buildView({ me, opponents, discard, inHand = null, discardPile = null }) {
    const counts = { ...DISTRIBUTION };
    const known = [...visibleValues(me), ...opponents.flatMap(visibleValues)];
    for (const v of discardPile ?? (discard === null || discard === undefined ? [] : [discard])) known.push(v);
    if (inHand !== null) known.push(inHand);
    for (const v of known) counts[v] -= 1;
    const isLastTurn = opponents.some((g) => hiddenCount(g) === 0);
    return { board: me, opponents, topDiscard: discard ?? null, belief: makeBelief(counts), isLastTurn };
  }

  // ------------------------------------------------------------------ loi normale (comme math.erfc)
  // Chebyshev (Numerical Recipes, 3e éd.) : précision ~1e-16, identique à math.erfc à l'arrondi près.
  const ERFC_COF = [-1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2, -9.561514786808631e-3,
    -9.46595344482036e-4, 3.66839497852761e-4, 4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
    -2.27365122e-10, 9.6467911e-11, 2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13,
    3.81e-16, 7.106e-15, -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17];
  function erfccheb(z) {
    let d = 0, dd = 0;
    const t = 2 / (2 + z), ty = 4 * t - 2;
    for (let j = ERFC_COF.length - 1; j > 0; j--) { const tmp = d; d = ty * d - dd + ERFC_COF[j]; dd = tmp; }
    return t * Math.exp(-z * z + 0.5 * (ERFC_COF[0] + ty * d) - dd);
  }
  const erfc = (x) => (x >= 0 ? erfccheb(x) : 2 - erfccheb(-x));

  // ------------------------------------------------------------------ stratégie paramétrable
  class Heuristic {
    constructor(profile) { this.p = profile; }

    ctx(view) {
      const p = this.p;
      const belief = p.card_counting ? view.belief : NAIVE;
      const e = belief.expected;
      const pos = positions(view.board);
      const cell = new Map(pos.map((q) => [q.join(","), peek(view.board, q)]));
      const hidden = pos.filter((q) => cell.get(q.join(",")) === null);
      let myEst = 0;
      for (const v of cell.values()) if (v !== null) myEst += v;
      myEst += hidden.length * e;
      let oppBest = Infinity;
      for (const g of view.opponents) oppBest = Math.min(oppBest, visibleScore(g) + hiddenCount(g) * e);
      let tempo;
      if (view.isLastTurn) tempo = 0;
      else {
        const lead = Math.max(-1, Math.min(1, (oppBest - myEst) / 10));
        tempo = p.reveal_bonus + p.tempo_weight * lead;
      }
      const hiddenPlace = view.isLastTurn ? e : e - p.hidden_discount;
      return { belief, hiddenValue: e, hiddenPlace, myEst, oppBest, tempo, positions: pos, hidden, cell,
               denyCache: new Map(), nextBoard: null };
    }

    pLowest(view, ctx, myFinal) {
      const v = ctx.belief.variance;
      let prob = 1;
      for (const g of view.opponents) {
        const h = hiddenCount(g);
        const mu = visibleScore(g) + h * ctx.hiddenValue - this.p.opp_last_turn_gain;
        const sd = Math.sqrt(h * v + 4);
        const z = (myFinal + 0.5 - mu) / sd;
        prob *= 0.5 * erfc(z / Math.SQRT2);
      }
      return prob;
    }

    finishValue(view, ctx, myFinal) {
      const p = this.p;
      if (p.finish_policy === "always") return 0;
      if (p.finish_policy === "if_lowest") {
        if (myFinal > 0 && myFinal >= ctx.oppBest - p.finish_margin) return -myFinal;
        return p.finish_bonus;
      }
      const pl = this.pLowest(view, ctx, myFinal);
      const cost = myFinal > 0 ? (1 - pl) * myFinal : 0;
      return pl * p.finish_bonus - cost;
    }

    finishDetail(view, ctx, myFinal) {
      const p = this.p;
      const d = { policy: p.finish_policy, my_final: myFinal, opp_best: ctx.oppBest, margin: p.finish_margin, bonus: p.finish_bonus };
      if (p.finish_policy === "probabilistic") d.p_lowest = this.pLowest(view, ctx, myFinal);
      return d;
    }

    pairValue(card) { return this.p.pair_bonus + this.p.pair_scale * Math.max(card, 0); }

    spread(others) {
      const hiddenInCol = 1 + others.filter((v) => v === null).length;
      return (this.p.reveal_spread * (hiddenInCol - 1)) / 2;
    }

    denyCost(view, ctx, card) {
      if (!this.p.deny_weight) return 0;
      if (ctx.denyCache.has(card)) return ctx.denyCache.get(card);
      if (ctx.nextBoard === null) {
        const nb = view.opponents[0];
        ctx.nextBoard = positions(nb).map((q) => [peek(nb, q), columnOthers(nb, q)]);
      }
      let best = 0;
      for (const [current, others] of ctx.nextBoard) {
        const cur = current === null ? ctx.hiddenValue : current;
        const gain = others[0] === card && others[1] === card ? cur + 2 * card : cur - card;
        if (gain > best) best = gain;
      }
      const cost = this.p.deny_weight * best;
      ctx.denyCache.set(card, cost);
      return cost;
    }

    placeValue(view, ctx, pos, card, enforceLimit, parts = null) {
      const p = this.p;
      const current = ctx.cell.get(pos.join(","));
      const hidden = current === null;
      if (hidden && enforceLimit && p.hidden_max !== null && card > p.hidden_max) {
        if (parts) parts.blocked = true;
        return null;
      }
      const cur = hidden ? ctx.hiddenPlace : current;
      const others = columnOthers(view.board, pos);
      const completes = others[0] === card && others[1] === card;
      const delta = completes ? cur + 2 * card : cur - card;
      let value = delta;
      if (parts) Object.assign(parts, { current, hidden, cur_value: cur, completes, gain: delta });
      if ((p.pair_bonus || p.pair_scale) && !completes) {
        if (others.includes(card) && card >= p.pair_min) {
          const pv = this.pairValue(card);
          value += pv;
          if (parts) parts.pair = pv;
        }
        if (!hidden && others.includes(current) && current >= p.pair_min) {
          const pv = this.pairValue(current);
          value -= pv;
          if (parts) parts.break_pair = -pv;
        }
      }
      if (hidden) {
        value += ctx.tempo;
        if (parts && ctx.tempo) parts.tempo = ctx.tempo;
        if (p.reveal_spread) {
          const sp = this.spread(others);
          value += sp;
          if (parts && sp) parts.spread = sp;
        }
        if (ctx.hidden.length === 1 && !view.isLastTurn) {
          const fv = this.finishValue(view, ctx, ctx.myEst - delta);
          value += fv;
          if (parts) { parts.finish = fv; parts.finish_detail = this.finishDetail(view, ctx, ctx.myEst - delta); }
        }
      } else {
        const dc = this.denyCost(view, ctx, current);
        value -= dc;
        if (parts && dc) parts.deny = -dc;
      }
      if (parts) parts.total = value;
      return value;
    }

    revealValue(view, ctx, pos, drawn, parts = null) {
      const dc = this.denyCost(view, ctx, drawn);
      let value = ctx.tempo - dc;
      if (parts) { if (ctx.tempo) parts.tempo = ctx.tempo; if (dc) parts.deny = -dc; }
      const others = columnOthers(view.board, pos);
      if (this.p.reveal_spread) {
        const sp = this.spread(others);
        value += sp;
        if (parts && sp) parts.spread = sp;
      }
      if (others[0] !== null && others[0] === others[1]) {
        const x = others[0];
        const chance = ctx.belief.prob(x) * 3 * x;
        value += chance;
        if (parts) { parts.column_chance = chance; parts.column_value = x; }
      }
      if (ctx.hidden.length === 1 && !view.isLastTurn) {
        const fv = this.finishValue(view, ctx, ctx.myEst);
        value += fv;
        if (parts) { parts.finish = fv; parts.finish_detail = this.finishDetail(view, ctx, ctx.myEst); }
      }
      if (parts) parts.total = value;
      return value;
    }

    // Départage déterministe (Python tire au sort entre coups strictement égaux).
    bestPlace(view, ctx, card, enforceLimit = true) {
      let bestPos = null, bestVal = -Infinity;
      for (const pos of ctx.positions) {
        const v = this.placeValue(view, ctx, pos, card, enforceLimit);
        if (v !== null && v > bestVal) { bestPos = pos; bestVal = v; }
      }
      return [bestPos, bestVal];
    }

    bestReveal(view, ctx, drawn) {
      let bestPos = null, bestVal = -Infinity;
      for (const pos of ctx.hidden) {
        const v = this.revealValue(view, ctx, pos, drawn);
        if (v > bestVal) { bestPos = pos; bestVal = v; }
      }
      return [bestPos, bestVal];
    }

    drawEv(view, ctx) {
      let ev = 0;
      for (const [card, prob] of ctx.belief.items()) {
        const vp = this.bestPlace(view, ctx, card)[1];
        const vr = this.bestReveal(view, ctx, card)[1];
        ev += prob * Math.max(vp, vr);
      }
      return ev;
    }

    options(view, ctx, card, withReveal) {
      const opts = [];
      for (const pos of ctx.positions) {
        const v = this.placeValue(view, ctx, pos, card, true);
        if (v !== null) opts.push({ kind: "place", pos, current: ctx.cell.get(pos.join(",")), value: v });
      }
      if (withReveal) for (const pos of ctx.hidden) opts.push({ kind: "reveal", pos, current: null, value: this.revealValue(view, ctx, pos, card) });
      return opts.sort((a, b) => b.value - a.value);
    }

    ctxSummary(view, ctx) {
      return { hidden_value: ctx.hiddenValue, hidden_place: ctx.hiddenPlace, counting: this.p.card_counting,
               unknown: ctx.belief.total, my_est: ctx.myEst, opp_best: ctx.oppBest, tempo: ctx.tempo, last_turn: view.isLastTurn };
    }

    params() {
      const p = this.p;
      return { hidden_max: p.hidden_max, discard_min_gain: p.discard_min_gain, lookahead: p.draw_lookahead,
               finish_policy: p.finish_policy, finish_margin: p.finish_margin };
    }

    explainPlace(view, ctx, pos, card) {
      const parts = { kind: "place", pos };
      this.placeValue(view, ctx, pos, card, false, parts);
      return parts;
    }

    explainReveal(view, ctx, pos, card) {
      const parts = { kind: "reveal", pos };
      this.revealValue(view, ctx, pos, card, parts);
      return parts;
    }

    /** Début de tour : prendre la défausse ? Renvoie l'explication (comme last_explanation en Python). */
    discardChoice(view) {
      const ctx = this.ctx(view);
      const card = view.topDiscard;
      const [pos, value] = this.bestPlace(view, ctx, card);
      let take, drawEv = null;
      if (pos === null) take = false;
      else if (this.p.draw_lookahead) { drawEv = this.drawEv(view, ctx); take = value > drawEv; }
      else take = value >= this.p.discard_min_gain;
      return {
        step: "discard_choice", card, take, blocked: pos === null,
        best: pos === null ? null : this.explainPlace(view, ctx, pos, card),
        draw_ev: drawEv, ctx: this.ctxSummary(view, ctx), params: this.params(),
        options: this.options(view, ctx, card, false),
      };
    }

    /** Défausse prise : où la poser. */
    placeDiscard(view, card) {
      const ctx = this.ctx(view);
      let [pos] = this.bestPlace(view, ctx, card);
      const forced = pos === null;
      if (pos === null) [pos] = this.bestPlace(view, ctx, card, false);
      return { step: "place_discard", card, forced, chosen: this.explainPlace(view, ctx, pos, card),
               options: this.options(view, ctx, card, false), ctx: this.ctxSummary(view, ctx), params: this.params() };
    }

    /** Carte piochée : la poser (où) ou la défausser et retourner (quoi). */
    drawn(view, card) {
      const ctx = this.ctx(view);
      let [posP, vp] = this.bestPlace(view, ctx, card);
      const [posR, vr] = this.bestReveal(view, ctx, card);
      let replace, forced = false;
      if (posR === null || (posP !== null && vp >= vr)) {
        forced = posP === null;
        if (posP === null) [posP] = this.bestPlace(view, ctx, card, false);
        replace = true;
      } else replace = false;
      return {
        step: "drawn", card, replace, forced,
        place: posP === null ? null : this.explainPlace(view, ctx, posP, card),
        reveal: posR === null ? null : this.explainReveal(view, ctx, posR, card),
        options: this.options(view, ctx, card, true), ctx: this.ctxSummary(view, ctx), params: this.params(),
      };
    }
  }

  return { ROWS, COLS, DISTRIBUTION, VALUES, NAIVE_EXPECTED, makeBelief, buildView, Heuristic, erfc,
           positions, hiddenCount, visibleScore, same };
});
