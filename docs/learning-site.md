# Reading the course with Jekyll + Just the Docs

Use Vim to edit the existing READMEs and a browser to read the course. The site offers ordered chapter navigation, a page table of contents, search, previous/next links, and code-copy buttons. Ruby implementation links open highlighted source pages; original files remain downloadable. Images and result JSON stay accessible.

The documentation toolchain uses Ruby: Jekyll, Just the Docs, Kramdown, Rouge, and WEBrick. Browser search uses the theme's bundled JavaScript. No Python, Node installation, or JavaScript build process is required.

## First-time installation

From the repository root, install the separate documentation bundle:

```bash
bin/learning-docs setup
```

This uses `docs/course_site/Gemfile` and its lockfile, placing documentation gems under the ignored `tmp/learning-docs-gems/`. It leaves the root training Gemfile and lockfile unchanged. Ruby and Bundler are required; native gem dependencies may need the usual compiler/development packages. No Torch, CUDA, training run, corpus, or model download is needed to read the site.

## Daily reading

```bash
bin/learning-docs serve
```

Open **http://127.0.0.1:8001/**. Start at the overview, then follow chapters 00–16. Use the left navigation to switch chapters, the table of contents to jump within a page, and the footer to move to the previous/next page. Search across lessons with the search box; `Ctrl+K` (macOS: `Command+K`) focuses it. On small screens, open navigation with the menu button.

Keep the terminal running, edit with Vim in another terminal, and save to rebuild and refresh the page. Stop with `Ctrl+C`.

If port 8001 is occupied:

```bash
bin/learning-docs serve --port 8002
```

Open **http://127.0.0.1:8002/**. The server listens on localhost by default.

## Static build

```bash
bin/learning-docs build
```

HTML and assets are written to the ignored `tmp/learning-site/`. You can serve that build using Ruby/WEBrick without rebuilding:

```bash
bin/learning-docs static
```

Open **http://127.0.0.1:8001/**. Use HTTP rather than opening HTML with `file://`, so browser search can load its index. Rebuild after editing the source documents. No remote hosting is necessary.

## Files and configuration

- `bin/learning-docs`: setup, build, preview, and static-server commands.
- `docs/course_site/Gemfile` and `Gemfile.lock`: separate Ruby documentation dependencies.
- `docs/course_site/_config.yml`: theme, search, and reading settings.
- `docs/course_site/_plugins/course.rb`: reads the existing lessons, converts links, and adds source/print pages.
- `learning/README.md` and the chapter READMEs: the source of course content; edit these rather than generated HTML.

The build reads original Markdown and Ruby files directly, without maintaining another copy of the lessons. Only course documents, source, result snapshots, and images are included; local runs, weights, and training datasets are excluded. Unresolved course links fail the build. The [course overview](../learning/README.md) and [printable course](learning-course-print.md) remain ordinary repository Markdown.

Official references: [Jekyll](https://jekyllrb.com/docs/), [Just the Docs navigation](https://just-the-docs.com/docs/navigation/main/), and [Just the Docs search](https://just-the-docs.com/docs/search/).
