// Public pages (templates/base.html): scroll reveals on the home page and a short fade between pages.
// No libraries. Nothing here runs for visitors who prefer reduced motion, and without this script
// every element is simply visible (the hidden state only exists once .reveal-ready is set).
(function () {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (reduceMotion) return;

  // ---- Scroll reveal (home page sections; the hero has its own entrance animation) ----
  const REVEAL = ['.section-head', '.about-left', '.about-art', '.highlights-card', '.feat-card', '.step-item',
    '.team-art', '.team-member-card', '.supervisor-row', '.contact-form-card', '.info-card'];
  if (document.querySelector('.hero') && 'IntersectionObserver' in window) {
    const counts = new Map();
    const targets = [];
    document.querySelectorAll(REVEAL.join(',')).forEach((el) => {
      if (el.closest('.hero')) return;
      const n = counts.get(el.parentElement) || 0;   // cards in the same grid come in one after another
      counts.set(el.parentElement, n + 1);
      el.style.setProperty('--reveal-delay', `${Math.min(n, 5) * 80}ms`);
      el.classList.add('reveal');
      targets.push(el);
    });
    document.documentElement.classList.add('reveal-ready');

    const finish = (el) => {   // hand the element back to its normal styles (hover lifts etc.)
      el.classList.remove('reveal', 'is-visible');
      el.style.removeProperty('--reveal-delay');
    };
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const el = entry.target;
        observer.unobserve(el);
        el.classList.add('is-visible');
        const onEnd = (event) => {
          if (event.target !== el || event.propertyName !== 'opacity') return;
          el.removeEventListener('transitionend', onEnd);
          finish(el);
        };
        el.addEventListener('transitionend', onEnd);
        setTimeout(() => finish(el), 1500);   // in case transitionend never fires (e.g. a background tab)
      });
    }, { rootMargin: '0px 0px -8% 0px', threshold: 0.12 });
    targets.forEach((el) => observer.observe(el));
  }

  // ---- Page transition: fade out for 300 ms, then follow the link ----
  const FADE_MS = 300;
  document.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest('a[href]');
    if (!link || (link.target && link.target !== '_self') || link.hasAttribute('download')) return;
    const url = new URL(link.href, window.location.href);
    if (url.origin !== window.location.origin) return;   // external, mailto:, tel:
    if (url.pathname === window.location.pathname && url.search === window.location.search) return;   // same page: #anchors just scroll
    event.preventDefault();
    document.body.classList.add('page-leaving');
    setTimeout(() => { window.location.href = url.href; }, FADE_MS);
  });
  // Coming back with the browser's Back button can restore the faded-out page from cache
  window.addEventListener('pageshow', (event) => {
    if (event.persisted) document.body.classList.remove('page-leaving');
  });
})();
