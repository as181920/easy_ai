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

## Deploy through GitLab CI

The root `.gitlab-ci.yml` follows the existing static-project workflow: a **manual** production job on **main**, using the shell runner's SSH credentials and rsync. It installs only the separate documentation bundle with the committed lockfile, builds with `JEKYLL_ENV=production`, validates links/search/URLs, and publishes the generated static files. It never runs training or starts a Ruby service on the server.

The production origin defaults to **https://easy-ai-learning.code-li.com** and the deployment directory to **/opt/www/easy_ai_learning**. In GitLab **Settings → CI/CD → Variables**, you can override these defaults:

| Variable | Value / purpose |
| --- | --- |
| `LEARNING_SITE_URL` | Defaults to `https://easy-ai-learning.code-li.com`; public origin without a path, matching your Caddy domain. |
| `LEARNING_SITE_BASEURL` | Empty for a dedicated domain/subdomain (default); `/course` if Caddy serves it under that path. |
| `DEPLOY_USER` | Defaults to `deployer`, matching the reference project. |
| `DEPLOY_HOST` | Defaults to `www.code-li.com`, matching the reference project. |
| `DEPLOY_PORT` | Defaults to `22`. |
| `DEPLOY_PATH` | `/opt/www/easy_ai_learning`, the confirmed server deployment directory. |

The runner needs Ruby 3.4.x, Bundler compatible with the documentation lockfile (currently 4.0.15), native-gem build tools, `ssh`, and `rsync`. Its SSH identity and trusted `known_hosts` must allow the existing `deployer@www.code-li.com` connection without interactive prompts. The server needs rsync and a writable `/opt/www/easy_ai_learning` for that deployer; create the directory with suitable ownership first if the deployer cannot create it. Caddy needs read access to the files. The job sets directories to 755 and files to 644, and you manage the Caddy configuration separately.

After pushing to `main`, open the pipeline and run `deploy_production`. The job synchronizes the **contents** of `tmp/learning-site/` to `/opt/www/easy_ai_learning/`. `--delete` removes obsolete files inside this dedicated course directory; do not share that directory with another application. A `resource_group` serializes course deployments. The static artifact remains downloadable from the job for one week.

To preview a production build locally before publishing:

```bash
LEARNING_SITE_URL=https://easy-ai-learning.code-li.com JEKYLL_ENV=production bin/learning-docs build
LEARNING_SITE_URL=https://easy-ai-learning.code-li.com bin/learning-docs check
```

The commands use the default production origin. Production builds require `LEARNING_SITE_URL` to prevent localhost canonical URLs. For a subpath, supply the same `LEARNING_SITE_BASEURL` for both commands. All generated course, source, image, search, and navigation links include that prefix; Caddy should map the prefix to the deployed directory. Local development retains its original localhost configuration.

### Caddy example

Add this site block to your server's Caddy configuration. The same example is stored at `docs/deployment/Caddyfile.learning.example`:

```caddyfile
easy-ai-learning.code-li.com {
    root * /opt/www/easy_ai_learning
    encode zstd gzip
    file_server
}
```

Serve the deployed directory directly; a Ruby reverse proxy is unnecessary. The course uses real HTML paths, so it does not need a single-page-application fallback. With the standard systemd Caddy installation, validate the complete configuration and reload it on the server:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile
sudo systemctl reload caddy
```

You manage this server configuration yourself; the CI job only publishes static files. Reference: [Caddy static file server](https://caddyserver.com/docs/caddyfile/directives/file_server).

## Files and configuration

- `bin/learning-docs`: setup, build, validation, preview, and static-server commands.
- `docs/course_site/Gemfile` and `Gemfile.lock`: separate Ruby documentation dependencies.
- `docs/course_site/_config.yml`: theme, search, and reading settings.
- `docs/course_site/_plugins/course.rb`: reads the existing lessons, converts links, and adds source/print pages.
- `learning/README.md` and the chapter READMEs: the source of course content; edit these rather than generated HTML.

The build reads original Markdown and Ruby files directly, without maintaining another copy of the lessons. Only course documents, source, result snapshots, and images are included; local runs, weights, and training datasets are excluded. Unresolved course links fail the build. The [course overview](../learning/README.md) and [printable course](learning-course-print.md) remain ordinary repository Markdown.

Official references: [Jekyll](https://jekyllrb.com/docs/), [Just the Docs navigation](https://just-the-docs.com/docs/navigation/main/), and [Just the Docs search](https://just-the-docs.com/docs/search/).
