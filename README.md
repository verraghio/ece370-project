# MIMO Detection Performance Comparison

This repository contains a MATLAB simulation that evaluates and compares the performance of various decoding algorithms in a Multiple-Input Multiple-Output (MIMO) wireless communication system. The simulation calculates the Symbol Error Rate (SER) across a range of Signal-to-Noise Ratio (SNR) values.

## Overview

The main script (`code.m`) performs a Monte Carlo simulation of a MIMO system operating over a Rayleigh fading channel. It specifically focuses on comparing standard linear detectors against the computationally expensive optimal Maximum Likelihood (ML) detector, as well as proposed Hybrid ML detectors designed to balance complexity and performance.

## System Parameters

The simulation models a wireless communication system with the following configuration:
* **Transmitting Antennas (Tx):** 2
* **Receiving Antennas (Rx):** 6
* **Modulation Scheme:** 16-QAM (Quadrature Amplitude Modulation)
* **Channel Model:** Rayleigh fading channel with Additive White Gaussian Noise (AWGN)
* **Number of Subcarriers:** 12
* **OFDM Symbols:** 7
* **SNR Range:** 5 dB to 15 dB (in increments of 2.5 dB)
* **Independent Trials:** 750 per SNR value

## Requirements

* MATLAB (The script uses standard built-in functions, but the Communications Toolbox is required for the `qammod` and `qamdemod` functions).


