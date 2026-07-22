<h1 align="center">ReferTrack: Referring Then Tracking for Embodied Visual Tracking</h1>

<p align="center">
    <a href="https://medlartea.github.io/">Hanjing Ye</a><sup>1,2</sup>
    &nbsp;
    <a>Tianle Zeng</a><sup>1</sup>
    &nbsp;
    <a href="https://jzhzhang.github.io/">Jiazhao Zhang</a><sup>3</sup>
    &nbsp;
    <a href="https://wsakobe.github.io/">Shaoan Wang</a><sup>3</sup>
    &nbsp;
    <a>Zibo Zhang</a><sup>4</sup>
    <br>
    <a href="https://situ-weixi.github.io/">Weisi Situ</a><sup>1</sup>
    &nbsp;
    <a href="https://yuchen2199.github.io/">Yuchen Zhou</a><sup>2</sup>
    &nbsp;
    <a href="https://ygling2008.github.io/">Yonggen Ling</a><sup>2,4*</sup>
    &nbsp;
    <a href="https://scholar.google.com/citations?user=J7UkpAIAAAAJ&hl=en">Hong Zhang</a><sup>1*</sup>
</p>

<p align="center">
    <sup>1</sup>RCV Laboratory, SUSTech
    &nbsp;
    <sup>2</sup>Tencent Robotics X
    <br>
    <sup>3</sup>Peking University
    &nbsp;
    <sup>4</sup>Futian Laboratory
</p>

<p align="center">
    <a href="">arXiv</a>
    &nbsp;|&nbsp;
    <a href="">Video</a>
</p>

## Overview

**_ReferTrack_** is a *referring-then-tracking* paradigm for embodied visual tracking that first grounds a language-described target to an image-space bounding box and then decodes tracking waypoints from this decision, using temporal-viewpoint-bbox indicator (TVBI) tokens to inject previously selected bounding boxes into the visual history and preserve target motion cues over time, achieving state-of-the-art single-view performance on EVT-Bench with robust sim-to-real transfer to legged and humanoid robots.

<p align="center">
    <img src="assets/method.png" alt="ReferTrack method overview" width="95%">
</p>

## TODO List

* [ ] Release model checkpoints and evaluation code.
* [ ] Release the dataset.
* [ ] Release the training code.
* [ ] Release the data engine.
