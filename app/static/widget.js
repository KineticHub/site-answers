/* Site Answers widget.
 *
 * One line on the owner's site adds it:
 *   <script src="https://YOUR-SERVER/widget.js" defer></script>
 *
 * It adds an "Ask a question" button. With each question it sends a reading of the forms
 * on the page: which fields are failing and why. It never sends what the visitor typed.
 */
(() => {
  if (window.__siteAnswersLoaded) return;
  window.__siteAnswersLoaded = true;
  // Our own crawler loads pages in a browser; the widget has nothing to do there.
  if (navigator.userAgent.includes("SiteAnswersBot")) return;

  const server = new URL(document.currentScript.src).origin;
  const PROBLEMS = [
    "valueMissing", "typeMismatch", "patternMismatch", "tooLong", "tooShort",
    "rangeUnderflow", "rangeOverflow", "stepMismatch", "badInput", "customError",
  ];
  const ERROR_SELECTOR = '[role="alert"], [aria-live="assertive"], .error, .errors, .invalid-feedback, .form-error';
  let lastSubmit = null; // { at: ms timestamp, blocked: boolean }

  // ---- reading the page ----------------------------------------------------

  function labelFor(el) {
    if (el.labels && el.labels.length) return el.labels[0].textContent.trim();
    return (el.getAttribute("aria-label") || el.placeholder || el.name || el.id || "").trim();
  }

  function isEmpty(el) {
    if (el.type === "checkbox" || el.type === "radio") return !el.checked;
    if (el.type === "file") return el.files.length === 0;
    return el.value === "";
  }

  /** Every value the visitor has typed, so it can be removed from any text we send. */
  function typedValues() {
    const values = [];
    document.querySelectorAll("input, textarea, select").forEach((el) => {
      if (el.type === "checkbox" || el.type === "radio" || el.type === "file") return;
      if (el.value && el.value.length >= 3) values.push(el.value);
    });
    return values.sort((a, b) => b.length - a.length);
  }

  function withoutTypedValues(text, values) {
    let out = text;
    for (const value of values) out = out.split(value).join("[what you typed]");
    return out.slice(0, 300);
  }

  function readPage() {
    const typed = typedValues();
    const forms = [...document.forms].slice(0, 10).map((form) => {
      const submit = form.querySelector('button[type="submit"], input[type="submit"], button:not([type])');
      const fields = [...form.elements]
        .filter((el) => el.willValidate !== undefined && !["submit", "button", "reset", "hidden", "fieldset"].includes(el.type) && el.tagName !== "BUTTON")
        .slice(0, 80)
        .map((el) => {
          const problems = PROBLEMS.filter((p) => el.validity && el.validity[p]);
          return {
            name: (el.name || el.id || "").slice(0, 200),
            label: labelFor(el).slice(0, 300),
            type: el.type || el.tagName.toLowerCase(),
            required: !!el.required,
            empty: isEmpty(el),
            valid: el.validity ? el.validity.valid : true,
            problems,
            hint: (el.title || "").slice(0, 300),
            // Only the site's own message, set by its scripts; browser messages can echo typed text.
            customMessage: problems.includes("customError") ? withoutTypedValues(el.validationMessage, typed) : "",
          };
        });
      return {
        name: (form.getAttribute("name") || form.id || "form").slice(0, 200),
        submitLabel: submit ? (submit.textContent || submit.value || "").trim().slice(0, 200) : "",
        submitDisabled: submit ? submit.disabled : false,
        fields,
      };
    });

    const errors = [...document.querySelectorAll(ERROR_SELECTOR)]
      .filter((el) => el.offsetParent !== null && el.textContent.trim() && !el.closest("#site-answers"))
      .slice(0, 10)
      .map((el) => withoutTypedValues(el.textContent.trim(), typed));

    return {
      forms,
      errors,
      lastSubmit: lastSubmit
        ? { secondsAgo: Math.round((Date.now() - lastSubmit.at) / 1000), blocked: lastSubmit.blocked }
        : null,
    };
  }

  // ---- noticing a stuck visitor -------------------------------------------
  //
  // Free checks in the browser spot the candidates. The server decides whether to
  // offer help, so a visitor selecting text or paging through photos is left alone.

  const INTERACTIVE = 'a[href], button, input, select, textarea, label, summary, [role="button"], [role="link"], [onclick]';
  const RAGE = { clicks: 3, withinMs: 1000, radius: 30 };   // fast clicks on one spot
  const DEAD = { clicks: 2, withinMs: 4000 };               // clicks on a switched-off control
  const FAILED = { submits: 2, withinMs: 60000 };           // submits the page refused
  const SAME_ATTEMPT_MS = 500;
  const MAX_CHECKS = 3;                                     // per page load
  const DISMISSED_KEY = "site-answers-dismissed";
  let recentClicks = [];
  let deadClicks = [];
  let failedSubmits = [];
  let lastAttemptAt = 0;
  let checksSent = 0;
  let checking = false;
  let noteShown = false;

  function wasDismissed() {
    try { return sessionStorage.getItem(DISMISSED_KEY) === "1"; } catch (error) { return false; }
  }

  function hasInvalidField(form) {
    return [...form.elements].some((el) => el.willValidate && el.validity && !el.validity.valid);
  }

  function describeTarget(el) {
    const control = el.closest(INTERACTIVE) || el;
    const form = control.closest("form");
    const isField = ["INPUT", "TEXTAREA", "SELECT"].includes(control.tagName) && !["submit", "button"].includes(control.type);
    const text = isField ? "" : (control.innerText || control.value || control.getAttribute("aria-label") || control.alt || "");
    return {
      tag: control.tagName.toLowerCase(),
      role: control.getAttribute("role") || "",
      text: withoutTypedValues(text.trim(), typedValues()).slice(0, 120),
      label: isField ? labelFor(control).slice(0, 200) : "",
      disabled: !!control.disabled || control.getAttribute("aria-disabled") === "true",
      isSubmit: !!form && (control.type === "submit" || (control.tagName === "BUTTON" && !control.getAttribute("type"))),
      interactive: control.matches(INTERACTIVE),
      formName: form ? (form.getAttribute("name") || form.id || "form") : "",
      cursor: getComputedStyle(control).cursor.slice(0, 40),
    };
  }

  async function reportSignal(kind, count, withinMs, el) {
    if (noteShown || checking || checksSent >= MAX_CHECKS || wasDismissed() || !panel.hidden) return;
    checksSent += 1;
    checking = true;
    try {
      const response = await fetch(`${server}/api/nudge`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host: location.host,
          page_url: location.href,
          signal: { kind, count, withinMs, target: describeTarget(el) },
          page_state: readPage(),
        }),
      });
      if (!response.ok) return;
      const data = await response.json();
      if (data.show && panel.hidden) showNote(data);
    } catch (error) {
      console.error("Site Answers:", error); // help is optional; the page carries on without it
    } finally {
      checking = false;
    }
  }

  function noteFailedAttempt(form, el) {
    const now = Date.now();
    if (now - lastAttemptAt < SAME_ATTEMPT_MS) return; // one press can raise several events
    lastAttemptAt = now;
    lastSubmit = { at: now, blocked: true };
    failedSubmits = failedSubmits.filter((t) => now - t < FAILED.withinMs);
    failedSubmits.push(now);
    if (failedSubmits.length >= FAILED.submits) {
      const count = failedSubmits.length;
      failedSubmits = [];
      reportSignal("failed_submit", count, FAILED.withinMs, el || form);
    }
  }

  // A submit the page's own script handles (it still fires when fields are invalid).
  document.addEventListener("submit", (event) => {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) return; // not one of the page's own forms
    if (hasInvalidField(form)) noteFailedAttempt(form, event.submitter);
    else lastSubmit = { at: Date.now(), blocked: false };
  }, true);

  // A submit the browser itself blocked: no submit event fires, only "invalid" on each bad field.
  document.addEventListener("invalid", (event) => {
    const field = event.target;
    if (field instanceof Element && field.form) noteFailedAttempt(field.form, null);
  }, true);

  // Disabled buttons swallow click events, but pointerdown still reaches the page.
  document.addEventListener("pointerdown", (event) => {
    const el = event.target;
    if (!(el instanceof Element) || el === host) return;
    const now = Date.now();
    const control = el.closest(INTERACTIVE) || el;

    if (control.disabled || control.getAttribute("aria-disabled") === "true") {
      deadClicks = deadClicks.filter((t) => now - t < DEAD.withinMs);
      deadClicks.push(now);
      if (deadClicks.length >= DEAD.clicks) {
        const count = deadClicks.length;
        deadClicks = [];
        reportSignal("dead_click", count, DEAD.withinMs, el);
      }
      return;
    }

    recentClicks = recentClicks.filter(
      (c) => now - c.t < RAGE.withinMs && Math.hypot(c.x - event.clientX, c.y - event.clientY) <= RAGE.radius
    );
    recentClicks.push({ t: now, x: event.clientX, y: event.clientY });
    if (recentClicks.length >= RAGE.clicks) {
      const count = recentClicks.length;
      recentClicks = [];
      // Triple-clicking text selects it, and clicking around inside a field is ordinary editing.
      const isField = ["INPUT", "TEXTAREA", "SELECT"].includes(control.tagName) && !["submit", "button", "checkbox", "radio"].includes(control.type);
      // A hand cursor means the element presents itself as clickable, so selection there is a side effect.
      const looksClickable = control.matches(INTERACTIVE) || getComputedStyle(control).cursor === "pointer";
      const selectingText = !looksClickable && String(window.getSelection()).trim() !== "";
      if (!isField && !selectingText) reportSignal("rage_click", count, RAGE.withinMs, el);
    }
  }, true);

  // ---- the panel -------------------------------------------------------------

  const host = document.createElement("div");
  host.id = "site-answers";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>
      :host { all: initial; }
      * { box-sizing: border-box; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
      .launcher {
        position: fixed; right: 16px; bottom: 16px; z-index: 2147483000;
        background: #14323f; color: #fff; border: 0; border-radius: 999px;
        padding: 12px 18px; font-size: 15px; cursor: pointer; box-shadow: 0 4px 14px rgba(0,0,0,.25);
      }
      .panel {
        position: fixed; right: 16px; bottom: 16px; z-index: 2147483001;
        width: min(380px, calc(100vw - 32px)); height: min(520px, calc(100vh - 32px));
        background: #fff; color: #1d2b36; border: 1px solid #cfd8de; border-radius: 12px;
        display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,.25);
      }
      .panel[hidden], .launcher[hidden] { display: none; }
      header { display: flex; align-items: center; justify-content: space-between; padding: 12px 14px; background: #14323f; color: #fff; }
      header h2 { margin: 0; font-size: 15px; font-weight: 600; }
      header button { background: none; border: 0; color: #fff; font-size: 14px; cursor: pointer; padding: 4px 6px; }
      .log { flex: 1; overflow-y: auto; padding: 14px; display: flex; flex-direction: column; gap: 10px; font-size: 14px; line-height: 1.45; }
      .intro { color: #5b6b78; }
      .msg { max-width: 92%; padding: 9px 11px; border-radius: 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
      .msg.you { align-self: flex-end; background: #14323f; color: #fff; }
      .msg.site { align-self: flex-start; background: #eef2f5; }
      .msg.problem { align-self: flex-start; background: #fdecea; color: #8c1d18; }
      .notice { font-size: 12px; color: #5b6b78; margin-top: 6px; }
      .sources { margin-top: 8px; font-size: 13px; }
      .sources a { color: #0b6e6e; display: block; }
      form { display: flex; gap: 8px; padding: 10px; border-top: 1px solid #e2e8ec; }
      input { flex: 1; min-width: 0; padding: 9px 10px; border: 1px solid #b8c4ce; border-radius: 8px; font-size: 14px; }
      form button { background: #0b6e6e; color: #fff; border: 0; border-radius: 8px; padding: 0 14px; font-size: 14px; cursor: pointer; }
      form button:disabled { background: #9fb3b3; cursor: default; }
      .note {
        position: fixed; right: 16px; bottom: 72px; z-index: 2147483001;
        width: min(330px, calc(100vw - 32px)); background: #fff; color: #1d2b36;
        border: 1px solid #cfd8de; border-left: 4px solid #0b6e6e; border-radius: 10px;
        padding: 12px 34px 12px 14px; font-size: 14px; line-height: 1.45; box-shadow: 0 8px 24px rgba(0,0,0,.2);
      }
      .note[hidden] { display: none; }
      .note p { margin: 0; }
      .note ul { margin: 6px 0 0; padding-left: 18px; }
      .note-close {
        position: absolute; top: 6px; right: 6px; background: none; border: 0; color: #5b6b78;
        font-size: 18px; line-height: 1; cursor: pointer; padding: 4px 8px;
      }
      .note-ask { margin-top: 10px; background: none; border: 0; padding: 0; color: #0b6e6e; font-size: 14px; cursor: pointer; text-decoration: underline; }
    </style>
    <div class="note" hidden role="status">
      <button class="note-close" type="button" aria-label="Dismiss this note">&times;</button>
      <p class="note-text"></p>
      <ul class="note-list"></ul>
      <button class="note-ask" type="button">Ask a follow-up</button>
    </div>
    <button class="launcher" type="button">Ask a question</button>
    <section class="panel" hidden aria-label="Ask about this site">
      <header><h2>Ask about this site</h2><button type="button" class="close">Close</button></header>
      <div class="log" aria-live="polite">
        <div class="intro">Ask anything about this site. If a form or button isn't working, ask why and I'll check it.</div>
      </div>
      <form>
        <input type="text" maxlength="1000" placeholder="Type your question" aria-label="Your question" autocomplete="off">
        <button type="submit">Ask</button>
      </form>
    </section>`;

  const launcher = root.querySelector(".launcher");
  const panel = root.querySelector(".panel");
  const log = root.querySelector(".log");
  const askForm = root.querySelector("form");
  const input = root.querySelector("input");
  const send = askForm.querySelector("button");

  function setOpen(open) {
    panel.hidden = !open;
    launcher.hidden = open;
    if (open) input.focus();
  }
  const note = root.querySelector(".note");
  let noteContent = "";

  function showNote(data) {
    noteShown = true;
    root.querySelector(".note-text").textContent = data.message;
    const list = root.querySelector(".note-list");
    list.replaceChildren(...data.details.map((detail) => {
      const item = document.createElement("li");
      item.textContent = detail;
      return item;
    }));
    list.hidden = data.details.length === 0;
    noteContent = [data.message, ...data.details.map((detail) => `- ${detail}`)].join("\n");
    note.hidden = false;
  }

  root.querySelector(".note-close").addEventListener("click", () => {
    note.hidden = true;
    try { sessionStorage.setItem(DISMISSED_KEY, "1"); } catch (error) { /* private mode: the note simply may reappear */ }
  });
  root.querySelector(".note-ask").addEventListener("click", () => {
    note.hidden = true;
    addMessage("site", noteContent);
    setOpen(true);
  });

  launcher.addEventListener("click", () => { note.hidden = true; setOpen(true); });
  root.querySelector(".close").addEventListener("click", () => setOpen(false));

  function addMessage(kind, text) {
    const el = document.createElement("div");
    el.className = `msg ${kind}`;
    el.textContent = text;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  askForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    const question = input.value.trim();
    if (!question) return;
    input.value = "";
    send.disabled = true;
    addMessage("you", question);
    const waiting = addMessage("site", "Checking the site…");

    try {
      const response = await fetch(`${server}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          host: location.host,
          page_url: location.href,
          question,
          page_state: readPage(),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        waiting.className = "msg problem";
        waiting.textContent = typeof data.detail === "string" ? data.detail : "That question could not be sent. Please try again.";
        return;
      }
      waiting.textContent = data.answer;
      if (data.sources.length) {
        const sources = document.createElement("div");
        sources.className = "sources";
        sources.append("From: ");
        for (const s of data.sources) {
          const a = document.createElement("a");
          a.href = s.url;
          a.textContent = s.title;
          sources.appendChild(a);
        }
        waiting.appendChild(sources);
      }
      if (data.notice) {
        const notice = document.createElement("div");
        notice.className = "notice";
        notice.textContent = data.notice;
        waiting.appendChild(notice);
      }
    } catch (error) {
      waiting.className = "msg problem";
      waiting.textContent = "The answer service can't be reached right now. Please try again in a moment.";
      console.error("Site Answers:", error);
    } finally {
      send.disabled = false;
      log.scrollTop = log.scrollHeight;
    }
  });

  document.body.appendChild(host);
})();
