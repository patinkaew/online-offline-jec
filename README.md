# online and offline jet energy correction (JEC) studies

## Introduction

In the begining, the code was designed to compare online (High Level Trigger or HLT) jets and offline jets. In the most recent version, online and offline are now on the equal footing so the software definition is blurry, i.e. it is possible to swap online and offline and get similar result. That means, for example, it is possible set online = PUPPI and offline = Gen to get offline vs gen jets, usual for JEC studies 

## Installation

Detailed installation will be updated later but this study is based on [coffea](https://coffeateam.github.io/coffea/).

## Structure

Will be updated later
  * code is divided into two parts (1) processing and (2) analysis
    * processing: from JMENano to histograms, including high dimentional)
    * analysis: reduce from histograms to plots
  * processing:
    * is based on coffea processor
    * (in the most recent version) is designed to be similar to pytorch
    * selector is the most basic unit which apply selection (x -> smaller x). This is very similar to nn.Module
    * processor is stacked of selectors then followed by histograms filling
    * histograms filling code still need some clean up
  * analysis:
    * compute statistics from histograms
    * visulization: reduce to lower dimensional histograms for human-interpretability
