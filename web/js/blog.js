/**
 * ChessFM Interactive Research Blog — Data-Driven Visuals & Interactive Widgets
 * Pure zero-dependency ES module backed 100% by ./data/blog_data.json.
 */

const PALETTE = {
  paper: '#FAF8F5',
  surface: '#FFFFFF',
  warm: '#F3EFE7',
  rule: '#E3DDD1',
  ink: '#171717',
  ink2: '#4A4842',
  muted: '#6E6A60',
  emerald: '#147A53',
  emeraldSoft: '#E6F4EE',
  vermilion: '#C83E2B',
  vermilionSoft: '#FBECE9',
  cobalt: '#1D5B9E',
  cobaltSoft: '#E8F1FA',
  amber: '#B86B14',
  amberSoft: '#FDF3E4',
  slate: '#738CA8',
  boardLight: '#EEEED2',
  boardDark: '#769656',
};

const PIECE_GLYPHS = {
  K: '♔', Q: '♕', R: '♖', B: '♗', N: '♘', P: '♙',
  k: '♚', q: '♛', r: '♜', b: '♝', n: '♞', p: '♟',
};

const FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];

export function escapeHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

export function parseFenToSquares(fen) {
  const boardPart = String(fen || '').trim().split(/\s+/)[0] || '';
  const ranks = boardPart.split('/');
  const map = {};
  if (ranks.length !== 8) return map;
  for (let r = 0; r < 8; r += 1) {
    let file = 0;
    for (const ch of ranks[r]) {
      if (ch >= '1' && ch <= '8') {
        file += Number(ch);
      } else if (PIECE_GLYPHS[ch] && file < 8) {
        map[`${FILES[file]}${8 - r}`] = ch;
        file += 1;
      }
    }
  }
  return map;
}

function sqToCoords(sq, size = 44) {
  const f = FILES.indexOf(sq[0]);
  const rank = Number(sq[1]);
  if (f < 0 || Number.isNaN(rank) || rank < 1 || rank > 8) return null;
  const r = 8 - rank;
  return { x: f * size, y: r * size, cx: f * size + size / 2, cy: r * size + size / 2 };
}

/**
 * Render an interactive tournament-grade SVG chessboard with optional move arrows.
 */
export function renderChessboardSvg(fen, { highlights = {}, arrows = [], size = 44 } = {}) {
  const total = size * 8;
  const pieces = parseFenToSquares(fen);
  const out = [];
  out.push(
    `<svg viewBox="0 0 ${total} ${total}" role="img" aria-label="Chess position ${escapeHtml(fen)}">`,
    '<defs>',
    `<marker id="cb-arrow-v" viewBox="0 0 10 10" refX="7.2" refY="5" markerWidth="5.2" markerHeight="5.2" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="${PALETTE.vermilion}"/></marker>`,
    `<marker id="cb-arrow-e" viewBox="0 0 10 10" refX="7.2" refY="5" markerWidth="5.2" markerHeight="5.2" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="${PALETTE.emerald}"/></marker>`,
    `<marker id="cb-arrow-c" viewBox="0 0 10 10" refX="7.2" refY="5" markerWidth="5.2" markerHeight="5.2" orient="auto-start-reverse"><path d="M 0 1 L 10 5 L 0 9 z" fill="${PALETTE.cobalt}"/></marker>`,
    '</defs>',
  );

  for (let r = 0; r < 8; r += 1) {
    for (let f = 0; f < 8; f += 1) {
      const sq = `${FILES[f]}${8 - r}`;
      const isLight = (r + f) % 2 === 0;
      const baseFill = isLight ? PALETTE.boardLight : PALETTE.boardDark;
      const x = f * size;
      const y = r * size;
      out.push(`<rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${baseFill}"/>`);
      if (highlights[sq]) {
        out.push(`<rect x="${x}" y="${y}" width="${size}" height="${size}" fill="${highlights[sq]}" opacity="0.34"/>`);
      }
      if (f === 0) {
        const lblColor = isLight ? '#5C7740' : '#E5E7C3';
        out.push(`<text x="${x + 3}" y="${y + 11}" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${lblColor}">${8 - r}</text>`);
      }
      if (r === 7) {
        const lblColor = isLight ? '#5C7740' : '#E5E7C3';
        out.push(`<text x="${x + size - 8}" y="${y + size - 3}" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${lblColor}">${FILES[f]}</text>`);
      }
      const pc = pieces[sq];
      if (pc) {
        const isWhite = pc === pc.toUpperCase();
        const glyph = PIECE_GLYPHS[pc];
        const fill = isWhite ? '#FFFFFF' : '#181818';
        const stroke = isWhite ? '#1D1D1D' : '#F6F5F0';
        const sw = isWhite ? '1.3' : '0.6';
        out.push(
          `<text x="${x + size / 2}" y="${y + size * 0.77}" text-anchor="middle" font-size="${Math.round(size * 0.78)}" fill="${fill}" stroke="${stroke}" stroke-width="${sw}" paint-order="stroke" style="user-select:none">${glyph}</text>`,
        );
      }
    }
  }

  for (const arrow of arrows) {
    const from = sqToCoords(arrow.from, size);
    const to = sqToCoords(arrow.to, size);
    if (!from || !to) continue;
    const dx = to.cx - from.cx;
    const dy = to.cy - from.cy;
    const dist = Math.hypot(dx, dy) || 1;
    const padStart = size * 0.14;
    const padEnd = size * 0.22;
    const x1 = from.cx + (dx / dist) * padStart;
    const y1 = from.cy + (dy / dist) * padStart;
    const x2 = to.cx - (dx / dist) * padEnd;
    const y2 = to.cy - (dy / dist) * padEnd;
    const markerId = arrow.color === PALETTE.vermilion ? 'cb-arrow-v' : arrow.color === PALETTE.cobalt ? 'cb-arrow-c' : 'cb-arrow-e';
    out.push(
      `<line class="anim-arrow" x1="${x1.toFixed(1)}" y1="${y1.toFixed(1)}" x2="${x2.toFixed(1)}" y2="${y2.toFixed(1)}" stroke="${arrow.color}" stroke-width="5.2" stroke-linecap="round" marker-end="url(#${markerId})"/>`,
    );
  }

  out.push('</svg>');
  return out.join('');
}

export function highlightSlopTerms(text) {
  let safe = escapeHtml(text);
  const slopPatterns = [
    /(tension in the center)/gi,
    /(central tension)/gi,
    /(resolving the tension[^,.]*)/gi,
    /(improve the placement of your minor pieces)/gi,
  ];
  for (const re of slopPatterns) {
    safe = safe.replace(re, '<span class="hl-slop">$1</span>');
  }
  return safe;
}

export function buildPassAtKSvg(passAtK) {
  const W = 490;
  const H = 270;
  const pad = { l: 44, r: 96, t: 24, b: 36 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const ks = ['pass@1', 'pass@2', 'pass@4', 'pass@8'];
  const xAt = (i) => pad.l + (i / (ks.length - 1)) * cw;
  const yAt = (v) => pad.t + ch - (v / 55) * ch;

  const series = [
    { key: 'E2a', label: 'E2a SFT', color: PALETTE.cobalt, dash: '' },
    { key: 'E3a', label: 'E3a Hack', color: PALETTE.vermilion, dash: '5 4' },
    { key: 'E3b', label: 'E3b Fix', color: PALETTE.emerald, dash: '' },
  ];

  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Stratified pass@1 to pass@8 curves">`];
  for (const pct of [0, 10, 20, 30, 40, 50]) {
    const y = yAt(pct);
    out.push(`<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${W - pad.r}" y2="${y.toFixed(1)}" stroke="${PALETTE.rule}" stroke-width="1"/>`);
    out.push(`<text x="${pad.l - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-family="ui-monospace,monospace" font-size="10" fill="${PALETTE.muted}">${pct}%</text>`);
  }

  for (let i = 0; i < ks.length; i += 1) {
    out.push(`<text x="${xAt(i).toFixed(1)}" y="${H - 12}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="10.5" font-weight="600" fill="${PALETTE.ink}">${ks[i]}</text>`);
  }

  // Shaded CI ribbons first
  for (const s of series) {
    const m = passAtK.models[s.key].pass_at_k;
    const upper = ks.map((k, i) => `${xAt(i).toFixed(1)},${yAt(m[k].ci95_pct[1]).toFixed(1)}`).join(' ');
    const lower = ks.map((k, i) => `${xAt(i).toFixed(1)},${yAt(m[k].ci95_pct[0]).toFixed(1)}`).reverse().join(' ');
    out.push(`<polygon points="${upper} ${lower}" fill="${s.color}" opacity="0.11"/>`);
  }

  // Lines & points
  for (const s of series) {
    const m = passAtK.models[s.key].pass_at_k;
    const pts = ks.map((k, i) => `${xAt(i).toFixed(1)},${yAt(m[k].mean_pct).toFixed(1)}`).join(' ');
    out.push(`<polyline fill="none" stroke="${s.color}" stroke-width="2.6" ${s.dash ? `stroke-dasharray="${s.dash}"` : ''} points="${pts}"/>`);
    ks.forEach((k, i) => {
      const v = m[k].mean_pct;
      const cx = xAt(i).toFixed(1);
      const cy = yAt(v).toFixed(1);
      out.push(`<circle cx="${cx}" cy="${cy}" r="4.2" fill="#FFFFFF" stroke="${s.color}" stroke-width="2.3"><title>${s.label} ${k}: ${v.toFixed(1)}%</title></circle>`);
    });
    const lastV = m['pass@8'].mean_pct;
    const ly = yAt(lastV) + (s.key === 'E3b' ? 7 : s.key === 'E3a' ? -2 : 3);
    out.push(`<text x="${W - pad.r + 8}" y="${ly.toFixed(1)}" font-family="ui-monospace,monospace" font-size="10.5" font-weight="700" fill="${s.color}">${s.label} ${lastV.toFixed(1)}%</text>`);
  }

  out.push('</svg>');
  return out.join('');
}

export function buildProbeTrajectorySvg(interp1) {
  const W = 500;
  const H = 280;
  const pad = { l: 44, r: 118, t: 26, b: 34 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const layers = interp1.layers;
  const xAt = (i) => pad.l + (i / (layers.length - 1)) * cw;
  const yAt = (v) => pad.t + ch - ((v - 58) / 17) * ch;

  const models = [
    { key: 'E1a_PieceList', label: 'E1a PL', color: PALETTE.cobalt, dash: '', w: 2.5 },
    { key: 'E3b_SharpWDL', label: 'E3b RLVR', color: PALETTE.emerald, dash: '', w: 2.7 },
    { key: 'E2a_r64', label: 'E2a r=64', color: '#4B8E78', dash: '', w: 1.9 },
    { key: 'E0_ASCII', label: 'E0 ASCII', color: PALETTE.amber, dash: '5 4', w: 2.1 },
    { key: 'Base_ASCII', label: 'Base ASCII', color: '#7D7970', dash: '3 3', w: 2.0 },
  ];

  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="64-square linear board probe accuracy across layers 0 to 35">`];

  // Highlight L26-L29 Piece-Tracker & L30-35 Rebound zone
  const xL24 = xAt(4);
  const xL30 = xAt(5);
  const xL35 = xAt(6);
  out.push(`<rect x="${(xL24 + (xL30 - xL24) * 0.35).toFixed(1)}" y="${pad.t}" width="${(xL35 - (xL24 + (xL30 - xL24) * 0.35)).toFixed(1)}" height="${ch}" fill="${PALETTE.cobaltSoft}" opacity="0.75" rx="4"/>`);
  out.push(`<text x="${((xL30 + xL35) / 2).toFixed(1)}" y="${pad.t + 14}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${PALETTE.cobalt}">L26-35 REBOUND</text>`);

  for (const pct of [60, 64, 68, 72]) {
    const y = yAt(pct);
    out.push(`<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${W - pad.r}" y2="${y.toFixed(1)}" stroke="${PALETTE.rule}" stroke-width="1"/>`);
    out.push(`<text x="${pad.l - 8}" y="${(y + 4).toFixed(1)}" text-anchor="end" font-family="ui-monospace,monospace" font-size="10" fill="${PALETTE.muted}">${pct}%</text>`);
  }

  layers.forEach((l, i) => {
    out.push(`<text x="${xAt(i).toFixed(1)}" y="${H - 12}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="10.5" font-weight="600" fill="${PALETTE.ink}">L${l}</text>`);
  });

  const labelOffsets = { E1a_PieceList: 0, E3b_SharpWDL: -3, E2a_r64: 10, E0_ASCII: -2, Base_ASCII: 10 };
  for (const m of models) {
    const hm = interp1.heatmap[m.key].layers;
    const pts = layers.map((l, i) => `${xAt(i).toFixed(1)},${yAt(hm[String(l)].occupied_pct).toFixed(1)}`).join(' ');
    out.push(`<polyline fill="none" stroke="${m.color}" stroke-width="${m.w}" ${m.dash ? `stroke-dasharray="${m.dash}"` : ''} points="${pts}"/>`);
    layers.forEach((l, i) => {
      const v = hm[String(l)].occupied_pct;
      out.push(`<circle cx="${xAt(i).toFixed(1)}" cy="${yAt(v).toFixed(1)}" r="3.4" fill="#FFFFFF" stroke="${m.color}" stroke-width="2"><title>${m.label} L${l}: ${v.toFixed(1)}%</title></circle>`);
    });
    const lastV = hm['35'].occupied_pct;
    out.push(`<text x="${W - pad.r + 7}" y="${(yAt(lastV) + 3 + (labelOffsets[m.key] || 0)).toFixed(1)}" font-family="ui-monospace,monospace" font-size="9.8" font-weight="700" fill="${m.color}">${m.label} ${lastV.toFixed(1)}%</text>`);
  }

  out.push('</svg>');
  return out.join('');
}

export function buildHeadMatrixSvg(interp4, mode = 'intact') {
  const W = 450;
  const H = 220;
  const padL = 32;
  const padT = 22;
  const cellW = 11.2;
  const cellH = 5.4;
  const topSet = new Map(interp4.top5_heads.map((h) => [`${h.layer}:${h.head}`, h]));
  const randSet = new Set(interp4.random5_heads.map((h) => `${h.layer}:${h.head}`));

  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="36 by 32 attention head circuit matrix">`];
  // Column labels
  for (let l = 0; l < 36; l += 5) {
    out.push(`<text x="${(padL + l * cellW + 5).toFixed(1)}" y="14" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8.5" fill="${PALETTE.muted}">L${l}</text>`);
  }
  out.push(`<text x="${(padL + 35 * cellW + 5).toFixed(1)}" y="14" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8.5" fill="${PALETTE.muted}">L35</text>`);
  out.push(`<text x="12" y="${(padT + 16 * cellH).toFixed(1)}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="8.5" fill="${PALETTE.muted}" transform="rotate(-90 12 ${(padT + 16 * cellH).toFixed(1)})">32 Heads</text>`);

  // Highlight L26-L29 strip
  out.push(`<rect x="${(padL + 26 * cellW - 1.5).toFixed(1)}" y="${padT - 2}" width="${(4 * cellW + 2).toFixed(1)}" height="${(32 * cellH + 3).toFixed(1)}" fill="${PALETTE.cobaltSoft}" rx="3"/>`);

  for (let l = 0; l < 36; l += 1) {
    for (let h = 0; h < 32; h += 1) {
      const key = `${l}:${h}`;
      const isTop = topSet.has(key);
      const isRand = randSet.has(key);
      let fill = '#E9E4D9';
      let title = `L${l}.H${h}`;
      if (isTop) {
        const info = topSet.get(key);
        fill = mode === 'top5' ? PALETTE.vermilion : PALETTE.cobalt;
        title = `${info.label} (Piece-Tracker, sel=+${info.mean_selectivity.toFixed(4)})`;
      } else if (isRand) {
        fill = mode === 'random5' ? PALETTE.amber : '#D4CBB8';
        title = `L${l}.H${h} (Layer-matched random control)`;
      }
      out.push(`<rect x="${(padL + l * cellW).toFixed(1)}" y="${(padT + h * cellH).toFixed(1)}" width="${(cellW - 2.2).toFixed(1)}" height="${(cellH - 1.4).toFixed(1)}" rx="1.2" fill="${fill}"><title>${title}</title></rect>`);
    }
  }
  out.push('</svg>');
  return out.join('');
}

export function buildSvdSpectrumSvg(interp2, pair = 'play') {
  const W = 540;
  const H = 250;
  const pad = { l: 48, r: 22, t: 28, b: 32 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;

  const isPlay = pair === 'play';
  const aProf = interp2.profiles[isPlay ? 'e3a_apawn_hack' : 'arm_c_v1_grpo'];
  const bProf = interp2.profiles[isPlay ? 'e3b_sharp_wdl' : 'arm_a_v2_grpo'];
  const aVals = aProf.per_layer.map((r) => r.total_frobenius);
  const bVals = bProf.per_layer.map((r) => r.total_frobenius);

  const minY = isPlay ? 0.175 : 0.12;
  const maxY = isPlay ? 0.255 : 0.165;
  const xAt = (l) => pad.l + (l / 35) * cw;
  const yAt = (v) => pad.t + ch - ((v - minY) / (maxY - minY)) * ch;

  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="36-layer exact LoRA QR-SVD Frobenius norm profile">`];

  if (isPlay) {
    // Highlight L34-L35 Hack Spike & L26-L29 Piece-Tracker band
    out.push(`<rect x="${xAt(26).toFixed(1)}" y="${pad.t}" width="${(xAt(29) - xAt(26)).toFixed(1)}" height="${ch}" fill="${PALETTE.cobaltSoft}" opacity="0.7"/>`);
    out.push(`<rect x="${xAt(33.6).toFixed(1)}" y="${pad.t}" width="${(xAt(35) - xAt(33.6) + 5).toFixed(1)}" height="${ch}" fill="${PALETTE.vermilionSoft}" opacity="0.9"/>`);
  } else {
    out.push(`<rect x="${xAt(16).toFixed(1)}" y="${pad.t}" width="${(xAt(20) - xAt(16)).toFixed(1)}" height="${ch}" fill="${PALETTE.emeraldSoft}" opacity="0.7"/>`);
  }

  const ticks = isPlay ? [0.18, 0.20, 0.22, 0.24] : [0.12, 0.135, 0.15, 0.16];
  for (const t of ticks) {
    const y = yAt(t);
    out.push(`<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${W - pad.r}" y2="${y.toFixed(1)}" stroke="${PALETTE.rule}" stroke-width="1"/>`);
    out.push(`<text x="${pad.l - 7}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-family="ui-monospace,monospace" font-size="9.5" fill="${PALETTE.muted}">${t.toFixed(2)}</text>`);
  }
  for (const l of [0, 5, 10, 15, 20, 25, 30, 35]) {
    out.push(`<text x="${xAt(l).toFixed(1)}" y="${H - 10}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" fill="${PALETTE.muted}">L${l}</text>`);
  }

  const ptsA = aVals.map((v, l) => `${xAt(l).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ');
  const ptsB = bVals.map((v, l) => `${xAt(l).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ');
  out.push(`<polyline fill="none" stroke="${PALETTE.vermilion}" stroke-width="2.4" points="${ptsA}"/>`);
  out.push(`<polyline fill="none" stroke="${PALETTE.emerald}" stroke-width="2.4" points="${ptsB}"/>`);

  if (isPlay) {
    out.push(`<circle cx="${xAt(35).toFixed(1)}" cy="${yAt(aVals[35]).toFixed(1)}" r="4.5" fill="${PALETTE.vermilion}"/>`);
    out.push(`<text x="${(xAt(35) - 8).toFixed(1)}" y="${(yAt(aVals[35]) - 8).toFixed(1)}" text-anchor="end" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${PALETTE.vermilion}">E3a Peak L35 (${aVals[35].toFixed(3)})</text>`);
    out.push(`<circle cx="${xAt(17).toFixed(1)}" cy="${yAt(bVals[17]).toFixed(1)}" r="4.5" fill="${PALETTE.emerald}"/>`);
    out.push(`<text x="${xAt(17).toFixed(1)}" y="${(yAt(bVals[17]) - 10).toFixed(1)}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${PALETTE.emerald}">E3b Peak L17</text>`);
  } else {
    out.push(`<text x="${xAt(18).toFixed(1)}" y="${pad.t + 14}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${PALETTE.emerald}">Arm A V2: stronger L16-L20</text>`);
    out.push(`<circle cx="${xAt(31).toFixed(1)}" cy="${yAt(aVals[31]).toFixed(1)}" r="4.5" fill="${PALETTE.vermilion}"/>`);
    out.push(`<text x="${xAt(31).toFixed(1)}" y="${(yAt(aVals[31]) - 9).toFixed(1)}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" font-weight="700" fill="${PALETTE.vermilion}">Arm C Peak L31</text>`);
  }

  out.push('</svg>');
  return out.join('');
}

export function buildLogitLensSvg(interp3) {
  const W = 440;
  const H = 190;
  const pad = { l: 42, r: 24, t: 22, b: 30 };
  const cw = W - pad.l - pad.r;
  const ch = H - pad.t - pad.b;
  const ll = interp3.logit_lens;
  const labels = ['0%', '25%', '50%', '75%', '100%', 'Post'];
  const vals = [
    ll.mean_prob_by_fraction['0.0'],
    ll.mean_prob_by_fraction['0.25'],
    ll.mean_prob_by_fraction['0.5'],
    ll.mean_prob_by_fraction['0.75'],
    ll.mean_prob_by_fraction['1.0'],
    ll.post_think_mean_prob,
  ];
  const xAt = (i) => pad.l + (i / (labels.length - 1)) * cw;
  const yAt = (v) => pad.t + ch - v * ch;

  const out = [`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Logit lens probability trajectory across think tokens">`];
  for (const t of [0, 0.5, 1.0]) {
    const y = yAt(t);
    out.push(`<line x1="${pad.l}" y1="${y.toFixed(1)}" x2="${W - pad.r}" y2="${y.toFixed(1)}" stroke="${PALETTE.rule}" stroke-width="1"/>`);
    out.push(`<text x="${pad.l - 7}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" font-family="ui-monospace,monospace" font-size="9.5" fill="${PALETTE.muted}">${t.toFixed(2)}</text>`);
  }
  const area = `${pad.l},${yAt(0).toFixed(1)} ` + vals.map((v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ') + ` ${xAt(5).toFixed(1)},${yAt(0).toFixed(1)}`;
  out.push(`<polygon points="${area}" fill="${PALETTE.emerald}" opacity="0.12"/>`);
  out.push(`<polyline fill="none" stroke="${PALETTE.emerald}" stroke-width="2.6" points="${vals.map((v, i) => `${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ')}"/>`);

  vals.forEach((v, i) => {
    const cx = xAt(i).toFixed(1);
    const cy = yAt(v).toFixed(1);
    out.push(`<circle cx="${cx}" cy="${cy}" r="4.2" fill="#FFFFFF" stroke="${PALETTE.emerald}" stroke-width="2.3"/>`);
    out.push(`<text x="${cx}" y="${H - 10}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="9.5" font-weight="600" fill="${PALETTE.ink}">${labels[i]}</text>`);
  });
  out.push(`<text x="${(xAt(5) - 10).toFixed(1)}" y="${(yAt(vals[5]) + 4).toFixed(1)}" text-anchor="end" font-family="ui-monospace,monospace" font-size="10.5" font-weight="700" fill="${PALETTE.emerald}">0.99997</text>`);
  out.push('</svg>');
  return out.join('');
}

/* ==========================================================================
   DOM Interactive Hydration (runs only in browser)
   ========================================================================== */

function setupScrollChoreography() {
  const bar = document.getElementById('reading-progress');
  const sections = Array.from(document.querySelectorAll('section.chapter[id]'));
  const tocLinks = new Map(
    Array.from(document.querySelectorAll('.toc-list a[href^="#"]')).map((a) => [a.getAttribute('href').slice(1), a]),
  );

  const onScroll = () => {
    const doc = document.documentElement;
    const max = Math.max(doc.scrollHeight - doc.clientHeight, 1);
    const pct = Math.min(100, Math.max(0, (window.scrollY / max) * 100));
    if (bar) bar.style.width = `${pct.toFixed(2)}%`;

    let current = sections[0]?.id;
    for (const s of sections) {
      if (s.getBoundingClientRect().top <= 140) current = s.id;
    }
    for (const [id, link] of tocLinks) {
      link.classList.toggle('active', id === current);
    }
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) {
          if (e.isIntersecting) {
            e.target.classList.add('visible');
            io.unobserve(e.target);
          }
        }
      },
      { threshold: 0.08, rootMargin: '0px 0px -30px 0px' },
    );
    document.querySelectorAll('.reveal').forEach((el) => io.observe(el));
  } else {
    document.querySelectorAll('.reveal').forEach((el) => el.classList.add('visible'));
  }
}

function hydrateRepresentationAndLadder(data) {
  const promptEl = document.getElementById('rep-prompt-preview');
  const statsEl = document.getElementById('rep-stat-chips');
  const asciiSample = `8 r . b q k . . r\n7 p p p . . p p p\n6 . . n . . . . .\n5 . . . n p . . .\n4 . b B . . . . .\n3 . . N P . N . .\n2 P P P . . P P P\n1 R . B Q K . . R`;
  const plSample = `White: Ke1, Qd1, Ra1, Rh1, Bc1, Bc4, Nc3, Nf3, pawns a2 b2 c2 d3 f2 g2 h2\nBlack: Ke8, Qd8, Ra8, Rh8, Bc8, Bb4, Nc6, Nd5, pawns a7 b7 c7 e5 f7 g7 h7`;

  const setRepMode = (mode) => {
    document.querySelectorAll('[data-rep-mode]').forEach((b) => {
      b.classList.toggle('active', b.dataset.repMode === mode);
    });
    if (mode === 'ascii') {
      promptEl.textContent = asciiSample;
      statsEl.innerHTML =
        '<span class="stat-chip vermilion">0-shot piece_on_square: 12.5%</span>' +
        '<span class="stat-chip">E0 SFT Strict Legal: 21.0%</span>';
    } else {
      promptEl.textContent = plSample;
      statsEl.innerHTML =
        '<span class="stat-chip emerald">0-shot piece_on_square: 40.0% (+27.5 pp)</span>' +
        '<span class="stat-chip emerald">E1a SFT Strict Legal: 63.5% (3.0x)</span>' +
        '<span class="stat-chip cobalt">E1a piece_on_square: 80.5%</span>';
    }
  };

  document.querySelectorAll('[data-rep-mode]').forEach((btn) => {
    btn.addEventListener('click', () => setRepMode(btn.dataset.repMode));
  });
  setRepMode('piecelist');

  // Interactive Ablation Ladder Bar Chart
  const barWrap = document.getElementById('ladder-bar-list');
  const metricConfigs = {
    strict_legal_pct: { max: 80, suffix: '%', ids: ['Base_ASCII', 'E0_SFT', 'E0_GRPO', 'E1a', 'E1b', 'E2a', 'E2b'] },
    puzzles_900_pct: { max: 26, suffix: '%', ids: ['E1a', 'E1b', 'E2a', 'E2b', 'Gemma_26B'] },
    chessqa_1000_pct: { max: 52, suffix: '%', ids: ['E1a', 'E1b', 'E2a', 'E2b', 'Gemma_26B'] },
    strict_acpl: { max: 460, suffix: ' cp', ids: ['E0_SFT', 'E1a', 'E1b', 'E2a', 'E2b'] },
  };
  const byId = new Map(data.sft_ladder.map((r) => [r.id, r]));

  const renderLadder = (metric) => {
    document.querySelectorAll('[data-ladder-metric]').forEach((b) => {
      b.classList.toggle('active', b.dataset.ladderMetric === metric);
    });
    const cfg = metricConfigs[metric];
    barWrap.innerHTML = cfg.ids
      .map((id) => {
        const row = byId.get(id);
        if (!row) return '';
        const val = row[metric];
        const pct = Math.min(100, Math.max(2, (val / cfg.max) * 100));
        const isWinner = id === 'E2a';
        const isFT = id === 'E2b';
        const is26B = id === 'Gemma_26B';
        const color = isWinner ? PALETTE.emerald : isFT ? PALETTE.amber : is26B ? '#9A948A' : PALETTE.cobalt;
        const extraNote = is26B && metric === 'puzzles_900_pct' ? ' (10.8% lenient)' : '';
        return `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(row.label)}<small>${escapeHtml(row.method)}</small></div>
            <div class="bar-track"><div class="bar-fill" style="width:${pct.toFixed(1)}%;background:${color};"></div></div>
            <div class="bar-val" style="color:${color}">${val.toFixed(1)}${cfg.suffix}${extraNote}</div>
          </div>`;
      })
      .join('');
  };

  document.querySelectorAll('[data-ladder-metric]').forEach((btn) => {
    btn.addEventListener('click', () => renderLadder(btn.dataset.ladderMetric));
  });
  renderLadder('puzzles_900_pct');
}

function hydrateAPawnHackWidget(data) {
  const examples = data.rlvr_apawn_hack.board_examples;
  const tabsEl = document.getElementById('apawn-pos-tabs');
  const boardEl = document.getElementById('apawn-board-svg');
  const pillsEl = document.getElementById('apawn-move-pills');
  const diversityEl = document.getElementById('apawn-diversity-bars');

  let posIdx = 0;
  let viewMode = 'both';

  tabsEl.innerHTML = examples
    .map((ex, i) => `<button type="button" class="ctrl-btn ${i === 0 ? 'active' : ''}" data-pos-idx="${i}">${i + 1}. ${escapeHtml(ex.title)}</button>`)
    .join('');

  const renderBoard = () => {
    const ex = examples[posIdx];
    const m3a = ex.E3a;
    const m3b = ex.E3b;
    const highlights = {};
    const arrows = [];

    if (viewMode === 'e3a' || viewMode === 'both') {
      highlights[m3a.move.slice(0, 2)] = PALETTE.vermilion;
      highlights[m3a.move.slice(2, 4)] = PALETTE.vermilion;
      arrows.push({ from: m3a.move.slice(0, 2), to: m3a.move.slice(2, 4), color: PALETTE.vermilion });
    }
    if (viewMode === 'e3b' || viewMode === 'both') {
      highlights[m3b.move.slice(0, 2)] = PALETTE.emerald;
      highlights[m3b.move.slice(2, 4)] = PALETTE.emerald;
      arrows.push({ from: m3b.move.slice(0, 2), to: m3b.move.slice(2, 4), color: PALETTE.emerald });
    }

    boardEl.innerHTML = renderChessboardSvg(ex.fen, { highlights, arrows });
    pillsEl.innerHTML = `
      <div class="move-pill vermilion">
        <span>E3a Hack: ${escapeHtml(m3a.move)} (${escapeHtml(m3a.san)})</span>
        <span>+${m3a.cp_loss.toFixed(0)} cp · ${m3a.wdl_loss_pct.toFixed(1)}% WDL</span>
      </div>
      <div class="move-pill emerald">
        <span>E3b Fix: ${escapeHtml(m3b.move)} (${escapeHtml(m3b.san)})</span>
        <span>${m3b.cp_loss.toFixed(1)} cp · Exact Best</span>
      </div>`;
  };

  tabsEl.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-pos-idx]');
    if (!btn) return;
    posIdx = Number(btn.dataset.posIdx);
    tabsEl.querySelectorAll('.ctrl-btn').forEach((b, i) => b.classList.toggle('active', i === posIdx));
    renderBoard();
  });

  document.querySelectorAll('[data-apawn-view]').forEach((btn) => {
    btn.addEventListener('click', () => {
      viewMode = btn.dataset.apawnView;
      document.querySelectorAll('[data-apawn-view]').forEach((b) => b.classList.toggle('active', b === btn));
      renderBoard();
    });
  });

  const m = data.rlvr_apawn_hack.models;
  const rows = [
    { label: 'E2a (SFT r=64)', sub: '528 unique moves', pct: m.E2a.a_file_pawn_push_pct, color: PALETTE.cobalt },
    { label: 'E3a (Naive WDL)', sub: '206 unique — HACKED', pct: m.E3a.a_file_pawn_push_pct, color: PALETTE.vermilion },
    { label: 'E3b (sharp_wdl)', sub: '443 unique — CLEAN', pct: m.E3b.a_file_pawn_push_pct, color: PALETTE.emerald },
  ];
  diversityEl.innerHTML = rows
    .map(
      (r) => `
      <div class="bar-row">
        <div class="bar-label">${r.label}<small>${r.sub}</small></div>
        <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2.5, (r.pct / 65) * 100).toFixed(1)}%;background:${r.color};"></div></div>
        <div class="bar-val" style="color:${r.color}">${r.pct.toFixed(1)}%</div>
      </div>`,
    )
    .join('');

  renderBoard();
}

function hydratePassAtKAndElo(data) {
  document.getElementById('pass-at-k-svg').innerHTML = buildPassAtKSvg(data.pass_at_k);

  const eloBars = document.getElementById('elo-bars');
  const eloNote = document.getElementById('elo-mode-note');
  const suites = new Map(data.elo_mirage.suites.map((s) => [s.id, s]));

  const renderElo = (mode) => {
    document.querySelectorAll('[data-elo-mode]').forEach((b) => {
      b.classList.toggle('active', b.dataset.eloMode === mode);
    });
    const e2a = suites.get(mode === 'std' ? 'E2a_std' : 'E2a_norep');
    const e3b = suites.get(mode === 'std' ? 'E3b_std' : 'E3b_norep');
    const maxElo = 620;
    eloBars.innerHTML = [
      { label: 'E2a (SFT r=64)', sub: `${e2a.rep3_draws}/${e2a.draws} draws 3-fold (${e2a.rep3_pct_of_draws.toFixed(1)}%)`, elo: e2a.elo, ci: e2a.ci95, color: PALETTE.cobalt },
      { label: 'E3b (sharp_wdl)', sub: `${e3b.rep3_draws}/${e3b.draws} draws 3-fold · ${e3b.maia1100_wins}W vs Maia-1100`, elo: e3b.elo, ci: e3b.ci95, color: PALETTE.emerald },
    ]
      .map(
        (r) => `
        <div class="bar-row">
          <div class="bar-label">${r.label}<small>${r.sub}</small></div>
          <div class="bar-track"><div class="bar-fill" style="width:${((r.elo / maxElo) * 100).toFixed(1)}%;background:${r.color};"></div></div>
          <div class="bar-val" style="color:${r.color}">${r.elo.toFixed(1)}</div>
        </div>`,
      )
      .join('');

    if (mode === 'std') {
      eloNote.innerHTML = '<span class="stat-chip amber">Mirage active: 89.4% of E2a draws are stateless 3-fold repetitions (+230.9 Elo inflation)</span>';
    } else {
      eloNote.innerHTML = '<span class="stat-chip emerald">De-biased (--avoid-repetition): E3b gains +27.3 Elo over E2a and scores real wins vs Maia-1100</span>';
    }
  };

  document.querySelectorAll('[data-elo-mode]').forEach((btn) => {
    btn.addEventListener('click', () => renderElo(btn.dataset.eloMode));
  });
  renderElo('norep');
}

function hydrateCoachShowdown(data) {
  const barWrap = document.getElementById('coach-bar-list');
  const rows = data.e5_coach.rows;

  const renderCoachBars = (metric) => {
    document.querySelectorAll('[data-coach-metric]').forEach((b) => {
      b.classList.toggle('active', b.dataset.coachMetric === metric);
    });
    const order = ['row3_arm_a_grpo', 'row5_arm_b_grpo', 'row7_arm_c_v1_grpo', 'row6_teacher_26b', 'row2_arm_a_sft', 'row4_arm_b_sft', 'row1_base_4b'];
    const byId = new Map(rows.map((r) => [r.id, r]));
    barWrap.innerHTML = order
      .map((id) => {
        const r = byId.get(id);
        const val = r[metric];
        const color =
          id === 'row3_arm_a_grpo'
            ? PALETTE.emerald
            : id === 'row5_arm_b_grpo'
              ? PALETTE.cobalt
              : id === 'row7_arm_c_v1_grpo'
                ? PALETTE.vermilion
                : id === 'row6_teacher_26b'
                  ? PALETTE.amber
                  : '#7D7970';
        const sub = metric === 'all_clean_pct' ? `V1 ${r.v1_pass_pct.toFixed(1)}% · Slop ${r.slop_flagged_pct.toFixed(0)}%` : `All-Clean ${r.all_clean_pct.toFixed(1)}% · Spoiler-Free ${r.spoiler_free_pct.toFixed(1)}%`;
        return `
          <div class="bar-row">
            <div class="bar-label">${escapeHtml(r.short_label)}<small>${sub}</small></div>
            <div class="bar-track"><div class="bar-fill" style="width:${val.toFixed(1)}%;background:${color};"></div></div>
            <div class="bar-val" style="color:${color}">${val.toFixed(1)}%</div>
          </div>`;
      })
      .join('');
  };

  document.querySelectorAll('[data-coach-metric]').forEach((btn) => {
    btn.addEventListener('click', () => renderCoachBars(btn.dataset.coachMetric));
  });
  renderCoachBars('all_clean_pct');

  // Verbatim Coach Compare (IDX 24 Goodhart live comparison)
  const cmp = data.e5_coach.verbatim_comparisons[0];
  const boardWrap = document.getElementById('coach-compare-board');
  const tabsWrap = document.getElementById('coach-compare-tabs');
  const quoteWrap = document.getElementById('coach-compare-quote');

  boardWrap.innerHTML = renderChessboardSvg(cmp.fen, {
    highlights: {
      [cmp.best_move.slice(0, 2)]: PALETTE.emerald,
      [cmp.best_move.slice(2, 4)]: PALETTE.emerald,
    },
    arrows: [{ from: cmp.best_move.slice(0, 2), to: cmp.best_move.slice(2, 4), color: PALETTE.emerald }],
  });

  const renderQuote = (idx) => {
    const resp = cmp.responses[idx];
    tabsWrap.querySelectorAll('.ctrl-btn').forEach((b, i) => b.classList.toggle('active', i === idx));
    const v1Chip = `<span class="stat-chip ${resp.v1_passed ? 'emerald' : 'vermilion'}">Weak V1: ${resp.v1_passed ? 'PASS' : 'FAIL'}</span>`;
    const v2Chip = `<span class="stat-chip ${resp.v2_passed ? 'emerald' : 'vermilion'}">Strict V2: ${resp.v2_passed ? 'PASS' : 'FAIL'}</span>`;
    const violList = Array.isArray(resp.violations) ? resp.violations : [];
    const violNote = violList.length
      ? `<span class="stat-chip vermilion">Caught: ${escapeHtml(violList[0])}</span>`
      : '<span class="stat-chip emerald">0 Hallucinations · 0 Slop</span>';
    quoteWrap.innerHTML = `
      <div class="coach-quote-meta">${v1Chip}${v2Chip}${violNote}</div>
      <p class="coach-quote-text">&ldquo;${highlightSlopTerms(resp.response)}&rdquo;</p>`;
  };

  tabsWrap.innerHTML = cmp.responses
    .map((r, i) => `<button type="button" class="ctrl-btn ${i === 0 ? 'active' : ''}" data-c-idx="${i}">${escapeHtml(r.label)}</button>`)
    .join('');
  tabsWrap.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-c-idx]');
    if (!btn) return;
    renderQuote(Number(btn.dataset.cIdx));
  });
  renderQuote(0);
}

function hydrateInterpProbesAndCircuits(data) {
  document.getElementById('interp-probes-svg').innerHTML = buildProbeTrajectorySvg(data.interp.interp1_probes);

  const matrixEl = document.getElementById('head-matrix-svg');
  const abBars = document.getElementById('ablation-result-bars');
  const interp4 = data.interp.interp4_piece_tracker;
  const abl = interp4.ablation;

  const setAblationMode = (mode) => {
    document.querySelectorAll('[data-ablation-mode]').forEach((b) => {
      b.classList.toggle('active', b.dataset.ablationMode === mode);
    });
    matrixEl.innerHTML = buildHeadMatrixSvg(interp4, mode);

    const activeRetained = mode === 'intact' ? 100 : mode === 'random5' ? abl.random5_retained_solve_pct : abl.top5_retained_solve_pct;
    const activeDrop = mode === 'intact' ? 0 : mode === 'random5' ? abl.random5_solve_drop_pp : abl.top5_solve_drop_pp;
    const activeColor = mode === 'intact' ? PALETTE.emerald : mode === 'random5' ? PALETTE.amber : PALETTE.vermilion;
    const activeLabel = mode === 'intact' ? 'Intact E3b' : mode === 'random5' ? '5 Random L26–29' : 'Top-5 Trackers';

    abBars.innerHTML = `
      <div class="bar-row">
        <div class="bar-label">${activeLabel}<small>N=100 · Legal 98–99%</small></div>
        <div class="bar-track"><div class="bar-fill" style="width:${activeRetained.toFixed(1)}%;background:${activeColor};"></div></div>
        <div class="bar-val" style="color:${activeColor}">${activeRetained.toFixed(0)}%${activeDrop ? ` (-${activeDrop.toFixed(0)})` : ''}</div>
      </div>`;
  };

  document.querySelectorAll('[data-ablation-mode]').forEach((btn) => {
    btn.addEventListener('click', () => setAblationMode(btn.dataset.ablationMode));
  });
  setAblationMode('top5');
}

function hydrateSvdAndSlopSlider(data) {
  const svdWrap = document.getElementById('svd-spectrum-svg');
  const svdStats = document.getElementById('svd-pair-stats');
  const interp2 = data.interp.interp2_svd_slop;

  const renderSvd = (pair) => {
    document.querySelectorAll('[data-svd-pair]').forEach((b) => {
      b.classList.toggle('active', b.dataset.svdPair === pair);
    });
    svdWrap.innerHTML = buildSvdSpectrumSvg(interp2, pair);
    if (pair === 'play') {
      const a = interp2.profiles.e3a_apawn_hack;
      const b = interp2.profiles.e3b_sharp_wdl;
      svdStats.innerHTML =
        `<span class="stat-chip vermilion">E3a Hack: Peak L${a.peak_layer} · eff_rank ${a.mean_effective_rank.toFixed(2)} · Top-1 SV ${a.mean_top1_energy_pct.toFixed(1)}%</span>` +
        `<span class="stat-chip emerald">E3b Fix: Peak L${b.peak_layer} · eff_rank ${b.mean_effective_rank.toFixed(2)} · MLP ${b.mlp_energy_share_pct.toFixed(1)}%</span>`;
    } else {
      const a = interp2.profiles.arm_c_v1_grpo;
      const b = interp2.profiles.arm_a_v2_grpo;
      svdStats.innerHTML =
        `<span class="stat-chip vermilion">Arm C (Weak V1): Peak L${a.peak_layer} · MLP ${a.mlp_energy_share_pct.toFixed(1)}%</span>` +
        `<span class="stat-chip emerald">Arm A (Strict V2): Peak L${b.peak_layer} · Stronger mid-network L16–L20</span>`;
    }
  };

  document.querySelectorAll('[data-svd-pair]').forEach((btn) => {
    btn.addEventListener('click', () => renderSvd(btn.dataset.svdPair));
  });
  renderSvd('play');

  // Interactive Goodhart Slop Vector Slider
  const sweep = interp2.slop_steering.sweep;
  const slider = document.getElementById('slop-alpha-slider');
  const alphaOut = document.getElementById('slop-alpha-readout');
  const gSlop = document.getElementById('gauge-slop');
  const gGap = document.getElementById('gauge-gap');
  const gDiv = document.getElementById('gauge-div');
  const gSpoil = document.getElementById('gauge-spoil');
  const quoteEl = document.getElementById('slop-live-quote');

  const renderAlpha = (idx) => {
    const pt = sweep[idx];
    const sign = pt.alpha > 0 ? '+' : '';
    const mood = pt.alpha <= -1 ? 'Anti-Goodhart' : pt.alpha === 0 ? 'Unsteered Arm A' : 'Pro-Goodhart (Slop)';
    alphaOut.textContent = `α = ${sign}${pt.alpha.toFixed(1)} · ${mood}`;
    gSlop.textContent = `${pt.slop_flagged_pct.toFixed(0)}%`;
    gSlop.style.color = pt.slop_flagged_pct === 0 ? PALETTE.emerald : pt.slop_flagged_pct >= 30 ? PALETTE.vermilion : PALETTE.amber;
    gGap.textContent = `${pt.v1_minus_v2_pp.toFixed(0)} pp`;
    gDiv.textContent = pt.distinct_4gram_ratio.toFixed(3);
    gSpoil.textContent = `${pt.l12_spoiler_free_pct.toFixed(0)}%`;
    gSpoil.style.color = PALETTE.emerald;
    quoteEl.innerHTML = `&ldquo;${highlightSlopTerms(pt.sample_output_0)}&rdquo;`;
    document.querySelectorAll('[data-alpha-idx]').forEach((b) => {
      b.classList.toggle('active', Number(b.dataset.alphaIdx) === idx);
    });
  };

  slider.addEventListener('input', () => renderAlpha(Number(slider.value)));
  document.querySelectorAll('[data-alpha-idx]').forEach((btn) => {
    btn.addEventListener('click', () => {
      slider.value = btn.dataset.alphaIdx;
      renderAlpha(Number(btn.dataset.alphaIdx));
    });
  });
  renderAlpha(4); // Default to alpha=+2.0 so visitors immediately see the slop effect, and can slide down to -2.0!
}

function hydrateCotSabotage(data) {
  const interp3 = data.interp.interp3_cot_sabotage;
  document.getElementById('logit-lens-svg').innerHTML = buildLogitLensSvg(interp3);

  const examples = interp3.sabotage.examples;
  const exTabs = document.getElementById('sabotage-ex-tabs');
  const thinkBox = document.getElementById('sabotage-think-box');
  let exIdx = 0;
  let sabotaged = true;

  exTabs.innerHTML = examples
    .map((ex, i) => `<button type="button" class="ctrl-btn ${i === 0 ? 'active' : ''}" data-sab-idx="${i}">${escapeHtml(ex.target_move)} &rarr; ${escapeHtml(ex.distractor_move)}</button>`)
    .join('');

  const renderSab = () => {
    const ex = examples[exIdx];
    document.querySelectorAll('[data-sab-mode]').forEach((b) => {
      b.classList.toggle('active', (b.dataset.sabMode === 'sabotaged') === sabotaged);
    });
    if (sabotaged) {
      thinkBox.className = 'think-box sabotaged';
      thinkBox.innerHTML =
        `<span class="think-tag">&lt;think&gt;</span> [Injected Corrupted Prefill]<br>` +
        `Wait: ${escapeHtml(ex.target_move)} runs into a refutation and loses material. Instead, <strong>${escapeHtml(ex.distractor_move)}</strong> is the cleanest winning move. I will play ${escapeHtml(ex.distractor_move)}.<br>` +
        `<span class="think-tag">&lt;/think&gt;</span><br>` +
        `<strong>Model Output:</strong> <span class="hl-slop">${escapeHtml(ex.post_sabotage_move)}</span> (Flipped to distractor — 100/100 puzzles)`;
    } else {
      thinkBox.className = 'think-box';
      thinkBox.innerHTML =
        `<span class="think-tag">&lt;think&gt;</span> [Natural Model Chain-of-Thought]<br>` +
        `Scanning tactical lines from Piece-List coordinates... <strong>${escapeHtml(ex.target_move)}</strong> wins decisive material.<br>` +
        `<span class="think-tag">&lt;/think&gt;</span><br>` +
        `<strong>Model Output:</strong> <span class="hl-good">${escapeHtml(ex.target_move)}</span> (Solved puzzle · P = 0.99997 post-&lt;/think&gt;)`;
    }
  };

  exTabs.addEventListener('click', (ev) => {
    const btn = ev.target.closest('[data-sab-idx]');
    if (!btn) return;
    exIdx = Number(btn.dataset.sabIdx);
    exTabs.querySelectorAll('.ctrl-btn').forEach((b, i) => b.classList.toggle('active', i === exIdx));
    renderSab();
  });

  document.querySelectorAll('[data-sab-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      sabotaged = btn.dataset.sabMode === 'sabotaged';
      renderSab();
    });
  });

  renderSab();
}

async function initBlog() {
  setupScrollChoreography();
  const statusEl = document.getElementById('blog-data-status');
  try {
    const resp = await fetch('./data/blog_data.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    hydrateRepresentationAndLadder(data);
    hydrateAPawnHackWidget(data);
    hydratePassAtKAndElo(data);
    hydrateCoachShowdown(data);
    hydrateInterpProbesAndCircuits(data);
    hydrateSvdAndSlopSlider(data);
    hydrateCotSabotage(data);
    if (statusEl) statusEl.remove();
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = `Could not load ./data/blog_data.json (${err.message}). Serve web/ over HTTP to view interactive widgets.`;
    }
  }
}

if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initBlog);
  } else {
    initBlog();
  }
}
