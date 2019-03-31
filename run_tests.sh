#!/usr/bin/env bash
py.test -v --cov prfpy --cov-report term-missing prfpy -m "not confound"