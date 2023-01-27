# online and offline jet energy correction (JEC) studies

Detailed installation will be updated later but this study is based on [coffea](https://coffeateam.github.io/coffea/).

Some terminlogy:
  * code is divided into two parts (1) processing and (2) analysis
    * processing: from JMENano to histograms, including high dimentional)
    * analysis: reduce from histograms to plots
  * processing:
    * is based on coffea processor
    * (in the most recent version) is designed to be similar to pytorch
    * selector is the most basic unit which apply selection (x -> smaller x). This is very similar to nn.Module
    * processor is stacked of selectors then followed by filling histograms
  * analysis:
    * compute statistics from histograms
    * visulization: reduce to lower dimensional histograms for human-interpretability
