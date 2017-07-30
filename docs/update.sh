#!/usr/bin/env bash

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATE=$(date '+%Y-%m-%d %T')

sphinx-build "$DIR" "$DIR/_build"
ghp-import -m "Last update at $DATE" -b gh-pages "$DIR/_build"
git push -f origin gh-pages
