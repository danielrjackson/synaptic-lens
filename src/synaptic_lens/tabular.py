"""Small deterministic table export, independent of study paths and loaders."""

import csv


def write_csv(path, rows):
    """Write nonempty mapping rows in first-row column order, with LF endings.

    Like the historical helper, the caller owns directory creation, overwrite
    decisions and empty-table handling. This function is not an artifact store.
    """
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
