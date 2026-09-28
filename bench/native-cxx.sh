#!/bin/sh
if [ "$(uname -s)" = Darwin ]; then
  exec "${CXX:-c++}" "$@"
fi
exec "${CXX:-c++}" -march=native "$@"
