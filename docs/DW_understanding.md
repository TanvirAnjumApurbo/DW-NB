# DW-NB understanding from Amer et al. (2025) Eq. 9-15

DW-NB keeps Naive Bayes training unchanged and ports only the WPRkNN test-time neighborhood weighting logic into posterior correction. From the paper's WPRkNN section (Eq. 9-15) and flow in Figure 6, the local signal is built from three different geometric views of the k-neighbor set around a query point x*.

The first factor, w1, is inverse-distance accumulation per class: w1^c = sum_{i in N_{k,c}(x*)} 1/d_i. Geometrically, this is proximity-sensitive mass: very close neighbors contribute much more than far neighbors. It is continuous and strongly distance-decaying, so classes represented by close points receive large weight.

The second factor, w2, is class frequency in the k-neighborhood: w2^c = |N_{k,c}(x*)|/k. Geometrically, this is local voting share, ignoring metric magnitude and retaining only label composition. It captures "how many" neighbors support class c.

The third factor, w3, is a bounded relative-distance weighting anchored by nearest and kth-nearest distances:
for each neighbor i,

w3_i = ((d_k - d_i)/(d_k - d_1)) * ((d_k + d_i)/(d_k + d_1)) if d_k != d_1, else 1.

Then w3^c = sum_{i in N_{k,c}(x*)} w3_i. Geometrically, this gives graduated proximity relative to the observed local distance range; the closest neighbor gets 1 and the furthest gets 0 when d_k != d_1.

Because w1, w2, and w3 are on different raw scales, each component is normalized over classes before combination: w~_q^c = w_q^c / sum_{c'} w_q^{c'}. This makes each strategy contribute one unit of total mass before summation, preventing one strategy from dominating merely due to scale (for example, potentially large inverse-distance sums in w1).

DW-NB then fuses global and local evidence by geometric interpolation:

P_DW(c|x*) proportional to P_NB(c|x*)^(1-lambda) * W_c(x*)^lambda.

This is not an ensemble vote between two classifiers. It is a single posterior construction where log-space contributions are blended and then normalized once. So the output remains one coherent posterior distribution, not majority voting between model outputs.

Degenerate limits are clear: lambda = 0 recovers standard GaussianNB exactly; lambda = 1 yields a pure WPRkNN-style local classifier driven only by W_c. Also, if all k neighbors are equidistant (d_k = d_1), the paper sets w3_i = 1 for all neighbors. Then w3^c = |N_{k,c}| and, after normalization, w~_3^c = |N_{k,c}|/k, identical to frequency voting. In that degenerate geometry, w3 stops expressing distance gradation and collapses to count-based support.
