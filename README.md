# 朝倉書店「不動産テック」

朝倉書店「不動産テック」西分担執筆部分のサポートサイトです．
本文中で記載のあるプログラムを配布しています．
自分でシミュレーションを行い，理解を深めるためにご利用ください．

## ご注意

本サイトで示しているプログラムは，実務で利用するには向かないと思います．
線形回帰モデルや分位点回帰に関しては，
[statsmodels](https://www.statsmodels.org/stable/index.html)などを利用すると，
推定が安定的であるのみならず，パラメータの検定なども行ってくれます．
ニューラルネットワークについては，[Chainer](https://chainer.org)，
[PyTorch](https://pytorch.org)，[TensorFlow](https://www.tensorflow.org)
などを使うと効率的にプログラミングできると思います．
またこれらはGPUサポートをしているので，GPUを積んだマシンであればさらに高速化できます．

しかし，本サイトのモデル群はこれらのパッケージに依存せずに実装してありますので，
どのようにすれば推定できるのか気になる場合は参考になると思います．

## 参考

- [ゼロから作るDeep Learning―Pythonで学ぶディープラーニングの理論と実装](https://www.oreilly.co.jp/books/9784873117584/)
  - 斎藤 康毅, オライリー・ジャパン
- [Chainer](https://github.com/chainer/chainer)
  - 開発を修了してPyTorchに移行するとのことですので，これから使う場合はPyTorchかTensorFlowを推奨します．
