# Main Tables

## Table 1 | Retained materials by genre and level

| Genre | Source sentences | Target items | Triples |
|---|---:|---:|---:|
| Academic prose | 241 | 264 | 385 |
| Fiction | 82 | 86 | 134 |
| News | 230 | 245 | 361 |
| Total | 553 | 595 | 880 |

Note: source sentences are identified by source_sid; target items by sentence_id, with a designated target word. Each triple contains the M, A and I conditions. The three levels are counted separately and are not additive. Analyses give equal weight to 595 target items and cluster uncertainty by 553 source sentences.

## Table 2 | Regression results for the two RQ2 models

| Model / predictor | Estimate | CR1 SE | 95% CI | t | Two-sided p |
|---|---:|---:|---|---:|---:|
| M0: Intercept | 0.007336 | 0.001900 | [0.003604, 0.011068] | 3.861 | 1.26e-04 |
| M0: Contextual preference | 0.003249 | 0.000620 | [0.002032, 0.004467] | 5.242 | 2.27e-07 |
| M1: Intercept | 0.008479 | 0.002014 | [0.004523, 0.012435] | 4.210 | 2.98e-05 |
| M1: Context increment | 0.002611 | 0.000676 | [0.001283, 0.003939] | 3.863 | 1.25e-04 |
| M1: Word-only preference | 0.003730 | 0.000680 | [0.002396, 0.005065] | 5.489 | 6.16e-08 |

Note: both models predict continuation alignment, Δᵢ, from 595 target items nested within 553 source sentences. Coefficients use ordinary least squares (OLS). CR1 cluster-robust standard errors allow within-source dependence. Intervals are 95% confidence intervals based on a Student t reference distribution with 552 degrees of freedom. All p values are two-sided and unadjusted for multiple comparisons. M0 includes contextual preference; M1 includes the context increment and word-only preference. All preference scores are natural-log response-probability contrasts in nats. R-squared is 0.0402 for M0 and 0.0462 for M1. Displayed values are rounded; Source Data retain full precision and covariance matrices.
