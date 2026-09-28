# Frontend

`templates/index.html` is the operator dashboard. `templates/docs.html` is the
technical API guide. `static/app.js` implements interactions and charts;
`static/styles.css` defines the layout; `static/openapi.json` documents the API.

There are no frontend package dependencies or build tools. Flask serves these
files at `/` and `/static/`; all browser API calls use the same local server.
Start everything with `python main.py` from the project root. Python dependencies
are maintained once in the root `requirements.txt`.
