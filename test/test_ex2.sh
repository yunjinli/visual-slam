#!/usr/bin/env bash

set -x
set -e

./calibration --dataset-path ../data/euroc_calib/ --cam-model ds --show-gui 0
if !  ../test/compare_json.py ../test/ex2_test_data/gt_calib_ds.json opt_calib.json; then
    diff -u ../test/ex2_test_data/gt_calib_ds.json opt_calib.json
    exit 1
fi
rm opt_calib.json

./calibration --dataset-path ../data/euroc_calib/ --cam-model kb4 --show-gui 0
if ! ../test/compare_json.py ../test/ex2_test_data/gt_calib_kb4.json opt_calib.json; then
    diff -u ../test/ex2_test_data/gt_calib_kb4.json opt_calib.json
    exit 1
fi
rm opt_calib.json

./calibration --dataset-path ../data/euroc_calib/ --cam-model pinhole --show-gui 0
if ! ../test/compare_json.py ../test/ex2_test_data/gt_calib_pinhole.json opt_calib.json; then
    diff -u ../test/ex2_test_data/gt_calib_pinhole.json opt_calib.json
    exit 1
fi
rm opt_calib.json

./calibration --dataset-path ../data/euroc_calib/ --cam-model eucm --show-gui 0
if ! ../test/compare_json.py ../test/ex2_test_data/gt_calib_eucm.json opt_calib.json; then
    diff -u ../test/ex2_test_data/gt_calib_eucm.json opt_calib.json
    exit 1
fi
rm opt_calib.json

