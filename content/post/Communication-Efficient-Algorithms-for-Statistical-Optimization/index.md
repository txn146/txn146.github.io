---
title: "Communication Efficient Algorithms for Statistical Optimization"
description: 【他山之石】"Communication Efficient Algorithms for Statistical Optimization"
date: 2025-02-24T13:49:23Z
image: 
math: 
license: 
hidden: false
comments: true
draft: true
tags: 
    - stochastic optimization
    - subsampling
categories:
    - distributed learning
---

本篇为“Communication Efficient Algorithms for Statistical Optimization”一文的读书笔记，原文章2013年发表于Journal of Machine Learning Research，作者为Yuchen Zhang，John C. Duchi & Martin J. Wainwright

## 背景说明
许多统计估计方法都基于经验风险最小化的形式，即通过最小化由数据上的损失函数的平均值定义的目标函数路爱估计一个感兴趣的参数。考虑到目前数据量的爆炸式增长，设计高效的算法来解决大规模问题是一个重要的课题，在集中式设置中，已经有许多解决经验风险最小化问题的程序，包括标准的凸优化方法和随机逼近和优化算法。然而， 当数据集的规模变得极其庞大时，将所有数据存贮在单个计算机中是不可行的，这篇论文的重点是研究一些分布式且通信高效的经验风险最小化程序。

在分布式框架下，最简单的统计估计方法为“平均混合(AVGM)”算法。其给定$m$台机器和大小为$N$的数据集，首先将数据集平均分配给每台机器，使得每台机器处理的自己大小为$n=N/m$, 然后让每台机器计算该自己的经验最小化器$\theta_i$,最后将所有机器的参数估计值$\theta_i$进行平均。然而，许多实际使用的估计器是有偏的，问题在于，在更一般的情况下，这种分布式方法是否仍然有效？是否能证明AVGM方法在计算效率上通常由于单台机器使用$n=N/m$ 样本的简单方法。

## 问题设定

## 理论结果

## 数值模拟

## 实例分析

## 参考文献

- Zhang, Y., Duchi, J. C., & Wainwright, M. J. (2013). Communication-Efficient Algorithms for Statistical Optimization. *Journal of Machine Learning Research*, 14, 3321-3363.