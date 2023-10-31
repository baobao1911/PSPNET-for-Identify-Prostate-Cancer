# Multi-Scale Parallel Branch For MICCAI Automatic Prostate Gleason Grading Challenge 2019

This challenge is part of the MICCAI 2019 Conference to be held from October 13 to 17 in Shenzhen, China. This challenge will be one of the three challenges under the MICCAI 2019 Grand Challenge for Pathology.

## Objectives
This challenge will provide a unique dataset and evaluation framework for the important and challenging task of prostate cancer Gleason grading. It will help establish a benchmark for assessing and comparing the state of the art image analysis and machine learning-based algorithms for this challenging task. It will also help evaluate the accuracy and robustness of these computerized methods against the opinion of multiple human experts. Given the critical importance of prostate cancer and extreme utility of Gleason grade system for detection and diagnosis of prostate cancer, the results of this challenge can be of utmost utility for medical community.

*The challenge involves two separate tasks:*
Task 1: Pixel-level Gleason grade prediction
Task 2: Core-level Gleason score prediction

In this project, i will process to solve task 1.
## Preprocessing
Data used in this challenge consists of a set of tissue micro-array (TMA) images. Each TMA image is annotated in detail by several expert pathologists. So i use Majority-Voting to get GroundTruth.
[*Download here*](https://gleason2019.grand-challenge.org/Register/)

![image](https://github.com/baobao1911/Semantic_Segmentation_for_Prostate_Cancer_Detection/assets/79504672/bee8587b-204e-44b6-b36f-d6031523142b)

## Proposed Model
