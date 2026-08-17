# Yield-impact assumptions and verified literature

> Web audit updated 2026-08-16. This document distinguishes reported evidence
> from project assumptions. It must not be presented as a field-calibrated
> causal yield model.

## 1. Supported scope

The project supports Early Blight (*Alternaria solani*), Late Blight
(*Phytophthora infestans*) and Bacterial Spot (*Xanthomonas* spp.). Disease
severity is estimated from the semantic lesion mask:

```text
severity (%) = 100 × disease pixels inside leaf / heuristic leaf pixels
```

Grad-CAM, Integrated Gradients and similar classifier explanations are not
segmentation masks and must not be used to count diseased pixels. They are
allowed only as qualitative explanations.

## 2. What the literature actually supports

### 2.1 Early Blight

Saha and Das studied tomato early blight over two seasons and fitted a linear
relationship between disease severity and crop loss. Their abstract reports a
loss of 0.75–0.77 t/ha for every one-percentage-point increase in severity,
with a pooled value of 0.76 t/ha. This is an absolute yield slope in t/ha, not
a universal percentage-loss coefficient.

Reference:

> Saha, P., & Das, S. (2012). Assessment of Yield Loss Due to Early Blight
> (*Alternaria solani*) in Tomato. *Indian Journal of Plant Protection*,
> 40(3), 195–198.

Metadata/abstract: https://indianjournals.com/article/ijpp1-40-3-007

### 2.2 Late Blight

Fontem evaluated protected and unprotected tomato varieties in Cameroon using
disease progress and marketable yield. Loss depended strongly on variety and
epidemic conditions; several unprotected plots reached 100% marketable yield
loss. The study does not establish a universal slope of 1.0–1.2 percentage
yield loss per one percent leaf severity.

References:

> Fontem, D. A. (2003). Quantitative Effects of Early and Late Blights on
> Tomato Yields in Cameroon. *Tropicultura*, 21(1), 36–41.

Full-text record: https://www.researchgate.net/publication/45266478_Quantitative_Effects_of_Early_and_Late_Blights_on_Tomato_Yields_in_Cameroon

> Nowicki, M., Foolad, M. R., Nowakowska, M., & Kozik, E. U. (2012). Potato
> and Tomato Late Blight Caused by *Phytophthora infestans*: An Overview of
> Pathology and Resistance Breeding. *Plant Disease*, 96(1), 4–17.
> https://doi.org/10.1094/PDIS-05-11-0458

### 2.3 Bacterial Spot

Pohronezny and Volin found significant reductions in marketable yield after
artificial bacterial-spot epidemics, with greater damage after early
inoculation. Fruit lesions and sunscald also contributed to marketable loss.
The paper does not provide a universal β=0.4 mapping from one leaf image to
percentage farm yield loss.

Reference:

> Pohronezny, K., & Volin, R. B. (1983). The Effect of Bacterial Spot on Yield
> and Quality of Fresh Market Tomatoes. *HortScience*, 18(1), 69–70.
> https://doi.org/10.21273/HORTSCI.18.1.69

Official PDF: https://journals.ashs.org/downloadpdf/view/journals/hortsci/18/1/article-p69.pdf

## 3. Project sensitivity scenarios

Because this repository contains no field-level yield observations paired
with its leaf images, it cannot estimate β statistically. The following values
are retained from the original project proposal only as central scenarios and
sensitivity intervals:

| Disease | β low | β central | β high |
|---|---:|---:|---:|
| Early Blight | 0.50 | 0.60 | 0.70 |
| Late Blight | 1.00 | 1.10 | 1.20 |
| Bacterial Spot | 0.30 | 0.40 | 0.45 |

For scenario analysis:

```text
potential_yield_impact = clip(beta × severity_proxy, 0, 100)
```

Every prediction must report the low, central and high values. Recommended
wording is “potential yield-impact sensitivity scenario”, not “predicted farm
yield loss”. A healthy classification returns zero.

## 4. Corrections to the previous version

- The Bashi, Rotem and Palti citation could not be verified with the stated
  title, volume and pages; it was replaced by Saha and Das (2012).
- The stated Olanya et al. (2001) tomato/potato title and pages could not be
  matched confidently; it was replaced by Fontem (2003) plus the APS review by
  Nowicki et al. (2012).
- The Pohronezny and Volin citation was not in *Plant Disease* 67(9), 979–982;
  the verified publication is *HortScience* 18(1), 69–70.
- None of the verified papers directly validates β=0.6, 1.1 or 0.4 for this
  dataset. Those values are therefore explicitly treated as assumptions.

## 5. Report limitation statement

The deep-learning results directly evaluate disease classification and lesion
segmentation. Severity remains a proxy because the leaf denominator is
heuristic. Yield impact is a sensitivity calculation informed by disease
literature but not calibrated against longitudinal plant-level disease or
harvest observations. Field validation requires repeated severity/AUDPC and
measured marketable yield for cultivar, growth stage and environment.
