<div align="center">
  <img src="resources/mmtrack-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmtrack)](https://pypi.org/project/mmtrack/)
[![PyPI](https://img.shields.io/pypi/v/mmtrack)](https://pypi.org/project/mmtrack)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmtracking.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmtracking/workflows/build/badge.svg)](https://github.com/open-mmlab/mmtracking/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmtracking/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmtracking)
[![license](https://img.shields.io/github/license/open-mmlab/mmtracking.svg)](https://github.com/open-mmlab/mmtracking/blob/master/LICENSE)

[📘Documentation](https://mmtracking.readthedocs.io/) |
[🛠️Installation](https://mmtracking.readthedocs.io/en/latest/install.html) |
[👀Model Zoo](https://mmtracking.readthedocs.io/en/latest/model_zoo.html) |
[🆕Update News](https://mmtracking.readthedocs.io/en/latest/changelog.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmtracking/issues/new/choose)

</div>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>

## Introduction :wave:
This repository contains the configuration files to evaluate SORT and DeepSORT using Spatial Hold on MOT17.

## Findings :mag:
From this study, we found that there are no significant differences between using the Hungarian algorithm or Spatial Hold when making bounding box detections.

![min_metrics](https://github.com/eplatero97/SpatialHold/blob/master/assets/sort_min_metrics.png)

## Installation :floppy_disk:
Once you install mmtracking, run below to install Spatial Hold:
```bash
python custom_setup.py install_lib
pip install loguru
```

## Run Experiments :running:
To run validation, run below:
```bash
CONFIG_FILE=configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-public-half_spatialhold.py
WORK_DIR=results/
python tools/test.py $CONFIG_FILE --eval track bbox --work-dir=$WORK_DIR
```
