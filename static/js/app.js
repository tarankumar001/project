/**
 * app.js
 * Motor Fault Detection – SaaS Dashboard Animations & UI Logic
 */

/* ================================================================
   1. SIDEBAR TOGGLE (mobile)
   ================================================================ */
document.addEventListener('DOMContentLoaded', function () {

  const sidebar      = document.getElementById('sidebar');
  const toggleBtn    = document.getElementById('sidebarToggle');
  const overlay      = document.getElementById('sidebarOverlay');

  if (toggleBtn && sidebar) {
    toggleBtn.addEventListener('click', function () {
      sidebar.classList.toggle('open');
      if (overlay) overlay.style.display = sidebar.classList.contains('open') ? 'block' : 'none';
    });
  }

  if (overlay) {
    overlay.addEventListener('click', function () {
      if (sidebar) sidebar.classList.remove('open');
      overlay.style.display = 'none';
    });
  }

  /* ================================================================
     2. ACTIVE NAV LINK HIGHLIGHT
     ================================================================ */
  const currentPath = window.location.pathname;
  document.querySelectorAll('.nav-link').forEach(function (link) {
    const href = link.getAttribute('href');
    if (href && currentPath === href) {
      link.classList.add('active');
    }
  });

  /* ================================================================
     3. ANIMATED METRIC COUNTER
     Targets elements with data-counter="<target_value>"
     ================================================================ */
  function animateCounter(el) {
    const target   = parseFloat(el.dataset.counter);
    const suffix   = el.dataset.suffix  || '';
    const decimals = parseInt(el.dataset.decimals || '0', 10);
    const duration = parseInt(el.dataset.duration || '1200', 10);
    const start    = performance.now();

    function step(ts) {
      const elapsed  = ts - start;
      const progress = Math.min(elapsed / duration, 1);
      // Ease out cubic
      const eased    = 1 - Math.pow(1 - progress, 3);
      const current  = target * eased;
      el.textContent = current.toFixed(decimals) + suffix;
      if (progress < 1) requestAnimationFrame(step);
    }

    requestAnimationFrame(step);
  }

  // Use IntersectionObserver to trigger counters when visible
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });

    document.querySelectorAll('[data-counter]').forEach(function (el) {
      observer.observe(el);
    });
  } else {
    // Fallback: animate immediately
    document.querySelectorAll('[data-counter]').forEach(animateCounter);
  }

  /* ================================================================
     4. TRAINING FORM – FULL-SCREEN OVERLAY
     ================================================================ */
  const trainingForm    = document.getElementById('trainingForm');
  const trainingOverlay = document.getElementById('training-overlay');

  if (trainingForm && trainingOverlay) {
    trainingForm.addEventListener('submit', function (e) {
      const fileInput = document.getElementById('csvFile');
      if (!fileInput || !fileInput.files || fileInput.files.length === 0) return; // let browser validate

      // Show overlay
      trainingOverlay.classList.add('visible');

      // Animate progress bar
      startTrainingProgress();
    });
  }

  function startTrainingProgress() {
    const bar   = document.getElementById('trainingBar');
    const pct   = document.getElementById('trainingPct');
    const steps = document.getElementById('trainingStep');
    if (!bar) return;

    const messages = [
      'Loading dataset…',
      'Splitting train/test sets…',
      'Fitting RandomForestClassifier…',
      'Evaluating model metrics…',
      'Generating confusion matrix…',
      'Plotting ROC curve…',
      'Saving model to disk…',
      'Finalizing results…',
    ];

    let progress = 0;
    let msgIndex = 0;

    const interval = setInterval(function () {
      // Slow down as we approach 90%
      const increment = progress < 60 ? (Math.random() * 12 + 3) :
                        progress < 85 ? (Math.random() * 5  + 1) :
                                        (Math.random() * 1  + 0.2);
      progress = Math.min(progress + increment, 93); // never reach 100 until server responds

      bar.style.width = progress.toFixed(1) + '%';
      if (pct) pct.textContent = Math.round(progress) + '%';

      if (steps && msgIndex < messages.length) {
        const threshold = (msgIndex + 1) * (90 / messages.length);
        if (progress >= threshold) {
          steps.textContent = messages[msgIndex];
          msgIndex++;
        }
      }

      if (progress >= 93) clearInterval(interval);
    }, 350);
  }

  /* ================================================================
     5. DRAG-AND-DROP UPLOAD ZONE
     ================================================================ */
  const dropZone  = document.getElementById('dropZone');
  const csvInput  = document.getElementById('csvFile');
  const fileLabel = document.getElementById('fileLabel');

  if (dropZone && csvInput) {
    dropZone.addEventListener('click', function () { csvInput.click(); });

    dropZone.addEventListener('dragover', function (e) {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', function () {
      dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      const files = e.dataTransfer.files;
      if (files.length) {
        csvInput.files = files;
        if (fileLabel) fileLabel.textContent = files[0].name;
      }
    });

    csvInput.addEventListener('change', function () {
      if (this.files.length && fileLabel) {
        fileLabel.textContent = this.files[0].name;
      }
    });
  }

  /* ================================================================
     6. STAGGERED TABLE ROW ANIMATION
     ================================================================ */
  document.querySelectorAll('.data-table tbody tr').forEach(function (row, index) {
    row.style.animationDelay = (index * 0.045) + 's';
  });

  /* ================================================================
     7. SCROLL-TRIGGERED SLIDE-UP for .anim-slide-up elements
     ================================================================ */
  if ('IntersectionObserver' in window) {
    const slideObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.style.animationPlayState = 'running';
          slideObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.anim-slide-up').forEach(function (el) {
      el.style.animationPlayState = 'paused';
      slideObserver.observe(el);
    });
  }

  /* ================================================================
     8. COPY CODE BUTTON
     ================================================================ */
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      const pre = btn.closest('.code-container').querySelector('pre');
      if (!pre) return;
      navigator.clipboard.writeText(pre.innerText).then(function () {
        btn.textContent = '✓ Copied!';
        setTimeout(function () { btn.textContent = 'Copy'; }, 2000);
      });
    });
  });

  /* ================================================================
     9. AUTO-REFRESH on Monitor Page
     ================================================================ */
  const monitorPage = document.getElementById('monitorPage');
  if (monitorPage) {
    let countdown = 2;
    const cdEl = document.getElementById('refreshCountdown');
    setInterval(function () {
      countdown--;
      if (cdEl) cdEl.textContent = countdown;
      if (countdown <= 0) window.location.reload();
    }, 1000);
  }

});
