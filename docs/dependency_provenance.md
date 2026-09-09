# Dependency provenance

The VLM package imports shared cache classes and codecs from `mlx_lm`. Its
single packaging owner is `requirements.txt`, which `pyproject.toml` exposes as
the project's dynamic dependency source through setuptools. The required LM
source is pinned immutably:

```text
mlx-lm @ git+https://github.com/LibraxisAI/mlx-lm-mirror.git@b5b2865007f864ef5a6378f88159c2bbb733ea43
```

That commit's full Git tree matches the previously tested LM source commit
`586343868bf7ed37456ec89daec97131d68480b1`. The pin selects the shared cache
and codec owner; it does not copy or reimplement that ownership in VLM.

Publication, dependency resolution, wheel `Requires-Dist`, installation,
imports, tests, model execution, and runtime behavior remain separate gates.
In particular, this source declaration does not prove that the pinned commit
is reachable from the public repository. The complete graph must later retain
`transformers>=5.14.0` and VLM's existing media dependencies when it is resolved
and built after structural closure.
