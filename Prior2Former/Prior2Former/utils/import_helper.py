"""
From U3HS
"""

cache = {}


def get_cuml():
    if "cuml" not in cache:
        print("Importing cuml for first time")
        import cuml

        cache["cuml"] = cuml
    return cache["cuml"]


def get_cudf():
    if "cudf" not in cache:
        print("Importing cudf for first time")
        import cudf

        cache["cudf"] = cudf
    return cache["cudf"]


def get_cugraph():
    if "cugraph" not in cache:
        print("Importing cugraph for first time")
        import cugraph

        cache["cugraph"] = cugraph
    return cache["cugraph"]
