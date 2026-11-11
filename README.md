# Site Answers

Site Answers is a local prototype that reads a website, reports what makes its pages hard for search engines and AI assistants to use, and adds a visitor widget that answers questions from the site and explains form errors.

## Run locally

Install Python 3.12 with SQLite FTS5 support and Google Chrome. The crawler uses Chrome to read pages that need JavaScript.

```sh
git clone https://github.com/KineticHub/site-answers.git
cd site-answers
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8100
```

In a second terminal, from the same directory:

```sh
.venv/bin/python -m http.server 8101 --bind 127.0.0.1 --directory demo-site
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.

## Try the demo

1. Open the [owner page](http://localhost:8100/), enter `http://localhost:8101/`, and click **Read this site**.
2. Review the SEO checks and click **Build the report**. A configured AI key also enables the question coverage checks that power the GEO score.
3. Open the fictional [Harbour Cycle Repair booking page](http://localhost:8101/book.html). Use **Fill demo form** to populate blank fields with sample data. Try the widget by asking where to book or why the form will not submit. The demo booking stays in the browser.

The app creates its SQLite database under `data/` when it starts. Keep both servers bound to localhost. The prototype's APIs do not require authentication.

## Optional AI setup

The crawler, rule-based report, passage search, and basic form diagnosis work without a key. To enable generated answers and the AI checks in the gap report, copy `.env.sample` to `.env`, set `ANTHROPIC_API_KEY` to your own key, and restart the app. Set `SITE_ANSWERS_MODEL` if your account uses a different model. Provider calls can incur charges.

Set `TYPESAFE_API_KEY` only if you want Jev to evaluate unclear signs that a visitor is stuck. Keep `.env` private.

## Run tests

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```
