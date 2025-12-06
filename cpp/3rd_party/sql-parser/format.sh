#!/bin/bash

unamestr=$(uname)
if [[ "$unamestr" == 'Darwin' ]]; then
    clang_format="/usr/local/opt/llvm/bin/clang-format"
  format_cmd="$clang_format -i -style=file '{}'"
elif [[ "$unamestr" == 'Linux' ]]; then
    format_cmd="clang-format -i -style=file '{}'"
fi

source_regex="^(src|test).*\.(cpp|h|y)"

if [ "${1}" = "all" ]; then
    find src test | grep -E "$source_regex" | xargs -I{} sh -c "${format_cmd}"
elif [ "$1" = "modified" ]; then
    # Run on all changed as well as untracked cpp/hpp files, as compared to the current HEAD. Skip deleted files.
    { git diff --diff-filter=d --name-only & git ls-files --others --exclude-standard; } | grep -E "$source_regex" | xargs -I{} sh -c "${format_cmd}"
elif [ "$1" = "staged" ]; then
    # Run on all files that are staged to be committed.
    git diff --diff-filter=d --cached --name-only | grep -E "$source_regex" | xargs -I{} sh -c "${format_cmd}"
else
    # Run on all changed as well as untracked cpp/hpp files, as compared to the current master. Skip deleted files.
    { git diff --diff-filter=d --name-only master & git ls-files --others --exclude-standard; } | grep -E "$source_regex" | xargs -I{} sh -c "${format_cmd}"
fi
