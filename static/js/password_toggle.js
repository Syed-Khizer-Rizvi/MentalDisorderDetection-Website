/* Show / hide toggle for password fields.
   Every <input type="password" class="pw-field"> gets an eye button inside its wrapper (the input's parent, which
   must be position: relative). Clicking it switches the field between type="password" and type="text". The button's
   label and aria-pressed state change with it, and a polite live region says "Password shown" / "Password hidden"
   because screen readers don't reliably announce a change of input type. The field goes back to hidden before its
   form submits, so browsers and password managers still treat it as a password. */
(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var ICONS = {
    eye: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    'eye-off': '<path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c6.5 0 10 8 10 8a18.5 18.5 0 0 1-2.16 3.19"/>' +
               '<path d="M6.61 6.61A13.53 13.53 0 0 0 2 12s3.5 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/>' +
               '<path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"/><line x1="2" y1="2" x2="22" y2="22"/>'
  };
  var announcer = null;

  function announce(text) {
    if (!announcer) {
      announcer = document.createElement('span');
      announcer.className = 'pw-sr-only';
      announcer.setAttribute('aria-live', 'polite');
      document.body.appendChild(announcer);
    }
    announcer.textContent = '';
    // A new text node after a tick makes repeated messages get read again
    setTimeout(function () { announcer.textContent = text; }, 50);
  }

  function setIcon(svg, name) { svg.innerHTML = ICONS[name]; }

  function render(input, button, svg) {
    var shown = input.type === 'text';
    setIcon(svg, shown ? 'eye-off' : 'eye');
    button.setAttribute('aria-label', shown ? 'Hide password' : 'Show password');
    button.setAttribute('aria-pressed', shown ? 'true' : 'false');
    button.title = shown ? 'Hide password' : 'Show password';
  }

  function enhance(input) {
    if (input.dataset.pwToggle === 'ready' || !input.parentElement) return;
    input.dataset.pwToggle = 'ready';

    var button = document.createElement('button');
    button.type = 'button';                 // never submits the form
    button.className = 'pw-toggle';
    if (input.id) button.setAttribute('aria-controls', input.id);
    var svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', '18');
    svg.setAttribute('height', '18');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');
    svg.setAttribute('stroke-linecap', 'round');
    svg.setAttribute('stroke-linejoin', 'round');
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('focusable', 'false');
    button.appendChild(svg);
    render(input, button, svg);

    button.addEventListener('click', function () {
      var show = input.type === 'password';
      // Keep the caret where it was, so toggling mid-typing doesn't jump to the start
      var start = input.selectionStart, end = input.selectionEnd, focused = document.activeElement === input;
      input.type = show ? 'text' : 'password';
      render(input, button, svg);
      announce(show ? 'Password shown' : 'Password hidden');
      if (focused && start !== null) { input.focus(); try { input.setSelectionRange(start, end); } catch (e) { /* noop */ } }
    });

    if (input.form) {
      input.form.addEventListener('submit', function () {
        if (input.type !== 'password') { input.type = 'password'; render(input, button, svg); }
      }, true);
    }

    input.classList.add('pw-has-toggle');
    input.insertAdjacentElement('afterend', button);
  }

  function init(root) {
    var fields = (root || document).querySelectorAll('input.pw-field');
    for (var i = 0; i < fields.length; i++) {
      if (fields[i].type === 'password' || fields[i].dataset.pwToggle === 'ready') enhance(fields[i]);
    }
  }

  window.PasswordToggle = { init: init };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { init(); });
  else init();
})();
