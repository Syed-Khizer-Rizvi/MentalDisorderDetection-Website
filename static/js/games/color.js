// Color Focus (Stroop-inspired): a colour word in a different ink colour; pick what the WORD says.
(function () {
  const COLORS = [
    { name: 'RED', hex: '#DC2626' }, { name: 'BLUE', hex: '#2563EB' },
    { name: 'GREEN', hex: '#16A34A' }, { name: 'YELLOW', hex: '#CA8A04' },
  ];
  const ROUNDS = 10;
  const $ = (id) => document.getElementById(id);
  const stage = $('cfStage'), word = $('cfWord'), ink = $('cfInk'), choices = $('cfChoices');
  let round = 0, score = 0, current = null, lastWord = null, playing = false, accepting = false, clock;

  COLORS.forEach((c, i) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'cf-choice';
    btn.dataset.name = c.name;
    btn.setAttribute('aria-label', `${c.name.toLowerCase()} (key ${i + 1})`);
    btn.innerHTML = `<span class="cf-dot" style="background:${c.hex}"></span><span>${c.name[0] + c.name.slice(1).toLowerCase()}</span><kbd>${i + 1}</kbd>`;
    btn.disabled = true;
    btn.addEventListener('click', () => answer(c.name));
    choices.appendChild(btn);
  });

  function nextWord() {
    round += 1;
    $('cfRound').textContent = `${round}/${ROUNDS}`;
    let w;
    do { w = COLORS[Math.floor(Math.random() * COLORS.length)]; } while (w.name === lastWord);
    const inks = COLORS.filter((c) => c.name !== w.name);   // ink is always different from the word
    const inkColor = inks[Math.floor(Math.random() * inks.length)];
    lastWord = w.name;
    current = w.name;
    word.textContent = w.name;
    word.style.color = inkColor.hex;
    ink.textContent = `written in ${inkColor.name.toLowerCase()} ink`;
    stage.classList.remove('is-right', 'is-wrong');
    accepting = true;
  }

  function answer(name) {
    if (!playing || !accepting) return;
    accepting = false;
    const right = name === current;
    if (right) score += 1;
    $('cfScore').textContent = score;
    void stage.offsetWidth;   // restart the animation if the same class is applied twice
    stage.classList.add(right ? 'is-right' : 'is-wrong');
    setTimeout(() => (round >= ROUNDS ? finish() : nextWord()), right ? 350 : 550);
  }

  function setChoices(enabled) { choices.querySelectorAll('button').forEach((b) => { b.disabled = !enabled; }); }

  function start() {
    round = 0; score = 0; lastWord = null; playing = true;
    $('cfScore').textContent = '0';
    $('cfDone').hidden = true; $('cfBest').hidden = true; $('cfStart').hidden = true;
    stage.hidden = false; choices.hidden = false;
    setChoices(true);
    Game.start();
    clearInterval(clock);
    clock = setInterval(() => { $('cfTime').textContent = Game.formatTime(Game.elapsed()); }, 500);
    nextWord();
    choices.querySelector('button').focus();
  }

  async function finish() {
    playing = false;
    clearInterval(clock);
    setChoices(false);
    const secs = Math.round(Game.elapsed());
    $('cfTime').textContent = Game.formatTime(secs);
    stage.hidden = true; choices.hidden = true;
    $('cfDoneText').textContent = `${score} of ${ROUNDS} correct in ${secs} seconds.` +
      (score === ROUNDS ? ' Perfect focus!' : score >= 7 ? ' Great attention.' : ' Every round trains your focus.');
    $('cfDone').hidden = false;
    $('cfAgain').focus();
    const result = await Game.complete({ score });
    if (result && result.new_best) $('cfBest').hidden = false;
  }

  $('cfStart').addEventListener('click', start);
  $('cfAgain').addEventListener('click', start);
  document.addEventListener('keydown', (e) => {
    const n = Number(e.key);
    if (n >= 1 && n <= 4) answer(COLORS[n - 1].name);
  });
})();
