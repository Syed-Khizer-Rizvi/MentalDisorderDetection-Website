// Memory Match: flip two cards; a match stays face up with a green glow, a miss turns back after 1 s.
(function () {
  const ICONS = [['🌸', 'Flower'], ['🍃', 'Leaf'], ['☀️', 'Sun'], ['🌙', 'Moon'], ['☁️', 'Cloud'],
                 ['⭐', 'Star'], ['🌊', 'Wave'], ['⛰️', 'Mountain'], ['🌳', 'Tree'], ['🌈', 'Rainbow']];
  const LEVELS = { easy: { cols: 4, pairs: 6 }, medium: { cols: 4, pairs: 8 }, hard: { cols: 5, pairs: 10 } };
  const $ = (id) => document.getElementById(id);
  const grid = $('mmGrid'), movesEl = $('mmMoves'), timeEl = $('mmTime'), pairsEl = $('mmPairs');
  let level, cards, first, locked, moves, found, clock, started;

  function announce(text) { $('mmAnnounce').textContent = text; }

  function build() {
    level = document.querySelector('input[name="level"]:checked').value;
    const { cols, pairs } = LEVELS[level];
    clearInterval(clock);
    first = null; locked = false; moves = 0; found = 0; started = false;
    movesEl.textContent = '0'; timeEl.textContent = '0:00'; pairsEl.textContent = `0/${pairs}`;
    $('mmDone').hidden = true;
    grid.hidden = false;
    grid.style.setProperty('--cols', cols);
    const deck = Game.shuffle([...ICONS.slice(0, pairs), ...ICONS.slice(0, pairs)]);
    grid.innerHTML = '';
    cards = deck.map(([icon, name], i) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'mm-card';
      btn.dataset.name = name;
      btn.setAttribute('aria-label', `Card ${i + 1}, face down`);
      btn.innerHTML = '<span class="mm-inner"><span class="mm-face mm-back"></span><span class="mm-face mm-front" aria-hidden="true"></span></span>';
      btn.querySelector('.mm-front').textContent = icon;
      btn.addEventListener('click', () => flip(btn, i));
      grid.appendChild(btn);
      return btn;
    });
  }

  function flip(card, index) {
    if (locked || card === first || card.classList.contains('is-matched')) return;
    if (!started) {
      started = true;
      Game.start();
      clock = setInterval(() => { timeEl.textContent = Game.formatTime(Game.elapsed()); }, 1000);
    }
    card.classList.add('is-flipped');
    card.setAttribute('aria-label', `Card ${index + 1}, ${card.dataset.name}`);
    if (!first) { first = card; announce(card.dataset.name); return; }

    moves += 1;
    movesEl.textContent = moves;
    const a = first, b = card;
    first = null;
    if (a.dataset.name === b.dataset.name) {
      [a, b].forEach((c) => {
        c.classList.add('is-matched');
        c.setAttribute('aria-label', `${c.dataset.name}, matched`);
        c.disabled = true;
      });
      found += 1;
      pairsEl.textContent = `${found}/${LEVELS[level].pairs}`;
      announce(`${b.dataset.name}: it's a match. ${found} of ${LEVELS[level].pairs} pairs.`);
      if (found === LEVELS[level].pairs) finish();
    } else {
      locked = true;
      announce(`${b.dataset.name}. No match.`);
      setTimeout(() => {
        [a, b].forEach((c) => {
          c.classList.remove('is-flipped');
          c.setAttribute('aria-label', `Card ${cards.indexOf(c) + 1}, face down`);
        });
        locked = false;
      }, 1000);
    }
  }

  async function finish() {
    clearInterval(clock);
    const secs = Math.round(Game.elapsed());
    $('mmDoneText').textContent = `Completed in ${moves} moves, ${secs} seconds.`;
    setTimeout(() => { $('mmDone').hidden = false; $('mmAgain').focus(); }, 600);
    announce(`All pairs found in ${moves} moves and ${secs} seconds.`);
    const result = await Game.complete({ score: moves, level });
    if (result && result.new_best) $('mmBest').hidden = false;
  }

  $('mmRestart').addEventListener('click', build);
  $('mmAgain').addEventListener('click', () => { $('mmBest').hidden = true; build(); });
  document.querySelectorAll('input[name="level"]').forEach((r) => r.addEventListener('change', build));
  build();
})();
