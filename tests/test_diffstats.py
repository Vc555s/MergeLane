import math

from mergelane import diffstats
from mergelane.inference import BASE_FEATURES, FEATURES

PATCH = """@@ -1,4 +1,6 @@
 def load(path):
-    return open(path).read()
+    if path is None:
+        raise ValueError("no path")
+    # read the file
+    return open(path).read()
"""


def test_count_file():
    r = diffstats.count_file("src/io.py", "modified", PATCH)
    assert r["hunks"] == 1 and r["add"] == 4 and r["del"] == 1
    assert r["add_cond"] == 1 and r["add_null"] == 1 and r["add_raise"] == 1 and r["add_comment"] == 1
    assert diffstats.count_file("x.py", "removed", None)["removed_file"] == 1


def test_diff_features_transform():
    f = diffstats.diff_features([{"filename": "src/io.py", "status": "modified", "patch": PATCH}] * 2, n_commits=2)
    assert f["add"] == math.log1p(8) and f["n_commits_seen"] == math.log1p(2)
    assert f["r_add_cond"] == 2 / (8 + 2 + 1)
    assert list(f) == diffstats.feature_names() and len(f) == 72


def test_diff_text_keeps_source_lines_only():
    t = diffstats.diff_text([{"filename": "README.md", "patch": "+docs"},
                             {"filename": "src/io.py", "patch": PATCH}])
    assert "README" not in t and t.startswith("\n### src/io.py\n") and "+    if path is None:" in t
    long = diffstats.diff_text([{"filename": f"src/f{i}.py", "patch": PATCH * 40} for i in range(10)])
    assert len(long) <= diffstats.MAX_CHARS


def test_token_matrix_shape():
    m = diffstats.token_matrix([diffstats.diff_text([{"filename": "a.py", "patch": PATCH}])])
    assert m.shape == (1, 2 * diffstats.N_HASH) and m.nnz > 0


def test_feature_list():
    assert FEATURES == BASE_FEATURES + diffstats.feature_names() + ["code_score"]
