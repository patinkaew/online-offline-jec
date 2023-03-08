# online and offline jet energy correction (JEC) studies

## Introduction

In the begining, the code was designed to compare online (High Level Trigger or HLT) jets and offline jets. In the current version, online and offline are now on the equal footing so the software definition is blurry, i.e. it is possible to swap online and offline and get similar result. That means, for example, it is possible set online = PUPPI and offline = Gen to get offline vs gen jets, usual for JEC studies 

## Installation

Detailed installation will be updated later but this study is based on [coffea](https://coffeateam.github.io/coffea/).

## Running on CERNCluster
1. Start grid with proxy: ```voms-proxy-init --rfc --voms cms -valid 192:00 --out ~/private/gridproxy.pem``` (in config file, set ```proxy_path```)
2. Start LCG_103 (LCG_102 ships with numba version incompatible with python 3.9): ```. /cvmfs/sft.cern.ch/lcg/views/LCG_103/x86_64-centos7-gcc11-opt/setup.sh```
3. Run code like usual (in config file, set ```executor``` to dask)
### Known problems
- HTCondor v9_0 recommendation authentication failed: check ```which condor_submit```. this should be ```/usr/bin/condor_submit```. Nightlies ships with different ```condor_submit``` and this might be a problem.
- code is running, but ```condor_q``` says it is done: unsure, remove conda environment or check logfile

## Structure

Will be updated later
  * code is divided into two parts (1) processing and (2) analysis
    * processing: from JMENano to histograms, including high dimentional)
    * analysis: reduce from histograms to plots
  * processing:
    * is based on coffea processor
    * is designed to be similar to pytorch
    * selector is the most basic unit which apply selection (x -> smaller x). This is very similar to nn.Module
    * processor is stacked of selectors then followed by histograms filling
    * histograms filling code still need some clean up
  * analysis:
    * compute statistics from histograms
    * visulization: reduce to lower dimensional histograms for human-interpretability
