# sample_projects

Code-mode project names entered in the dashboard resolve here.

Examples:

- `hello` -> `<repo>/sample_projects/hello`
- `sample_projects/hello` -> `<repo>/sample_projects/hello`
- `~.\sample_projects\hello` -> `<repo>/sample_projects/hello`

If a folder does not exist, code mode creates it and writes the latest project files there. This folder is the source of truth and is intentionally separate from `.workspace` / `.workspaces` runtime folders.
