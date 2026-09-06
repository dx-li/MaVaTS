# Publishing a release

The workflow `.github/workflows/publish.yml` follows
[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/).
No permanent PyPI token is stored in GitHub or passed through local tools.

The PyPI project `mavats` must trust owner `dx-li`, repository `MaVaTS`, workflow
filename `publish.yml`, environment `pypi`. The matching GitHub environment
allows release tags matching `v*`. The workflow additionally rejects tagged
commits outside `main` history and requires the tag to match package metadata.

1. Update `pyproject.toml`, `mavats.__version__`, README installation/versioned
   guide links, and `CHANGELOG.md`. Do not rewrite historical benchmark hashes
   or version fields; they identify the sources that actually produced results.
   Regenerate the current gallery after source/version changes with
   `python -m examples.air_quality --output docs/gallery`; its fingerprints
   must describe the release source that actually generated the figures.
2. Open and merge a PR only after the supported-platform tests and the release
   `build-and-check` job pass. PRs exercise packaging but cannot upload to PyPI.
3. Create a new annotated `v<version>` tag on the approved main commit and push
   that tag. This is the explicit publication trigger. Never move a released tag.
4. Monitor the Publish to PyPI workflow. It checks source/tag/metadata versions,
   runs tests, builds an sdist and wheel, checks rendered metadata with Twine,
   compares archive contents with the checkout, installs the wheel in isolation,
   and executes the synthetic examples and all 49 observational walkthroughs
   against that installed wheel outside the checkout. Nested guides, figures,
   dataset attribution and data files are checked byte-for-byte in the sdist.
   Only then does a separate
   job receive OIDC permission and upload the validated artifact, with attestations.
5. Verify the version and both distribution hashes through PyPI's JSON endpoint;
   install from PyPI in a fresh environment and verify import/version and a
   numerical example. Create the matching GitHub release, marking alpha/beta/RC
   versions as prereleases and attaching the same validated distributions.

The workflow is also manually dispatchable on an existing release tag for
recovery. Dispatching on a branch validates artifacts without publishing.
Do not enable `skip-existing` to conceal mismatched or partially uploaded
artifacts. If an upload fails, inspect PyPI and the logs before any retry;
published distribution filenames cannot simply be replaced. Fix defects with
a new version rather than silently changing an existing release.

GitHub Pages publishes the API reference from `main`, separately from PyPI.
Detailed guides and benchmark reports live in the source repository; README
links use the release tag so the PyPI project page points to matching source
documentation. A unified, versioned documentation portal remains future work.
