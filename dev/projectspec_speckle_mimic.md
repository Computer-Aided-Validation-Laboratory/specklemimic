# Project Specification: Speckle Mimic

## Background & Motivation

Experiments used to validate the performance of fusion engineering components are often large, time consuming, technically challenging and expensive. It is therefore desirable to maximise the information obtained from limited experimental campaigns through the use of full-field, image-based measurement techniques such as digital image correlation (DIC).

To maximise the value of DIC measurements, experimental configurations should be designed and assessed before significant experimental campaigns are undertaken. This requires simulation tools capable of predicting the expected performance of the measurement system, identifying important sources of uncertainty and quantifying systematic measurement errors arising from the experimental configuration.

Synthetic imaging simulations are a key tool for characterising systematic measurement uncertainties in DIC. In a synthetic imaging workflow, a known deformation field is applied to a virtual specimen and synthetic camera images are generated. The images can then be processed using the same DIC algorithms that will be used experimentally. Because the underlying deformation field is known exactly, errors introduced by the imaging and measurement process can be isolated and quantified.

A key source of uncertainty in DIC is the speckle pattern applied to the specimen surface. Speckle size, shape, density, contrast, spatial distribution and spatial frequency content can all influence the accuracy and robustness of DIC measurements. Consequently, realistic representation of experimental speckle patterns is important when using synthetic imaging to predict experimental measurement uncertainty.

It is desirable to use images of real experimental speckle patterns as the basis for these simulations because they directly encode characteristics of the actual experimental process, including irregular speckle shapes, clustering, variations in density, edge characteristics and other features that may be difficult to reproduce using simple procedural models.

However, experimentally acquired images are inherently discrete and spatially sampled by the camera sensor. Directly mapping these images into a synthetic imaging simulation therefore introduces the sampling and interpolation characteristics of the original image into the virtual experiment. These discretisation effects can themselves introduce parasitic numerical bias and make it difficult to separate errors originating from the experimental speckle pattern from errors introduced by its digital representation.

A tool is therefore required that can analyse one or more experimental images of a speckle pattern, infer the characteristics of the underlying continuous stochastic pattern and generate new representations that reproduce those characteristics without directly reproducing the discretisation of the input image.

The resulting speckle representation should either:

* generate a new high resolution image that is statistically and metrologically equivalent to the experimental pattern; or
* provide parameters for a procedural or analytic speckle model capable of generating a continuous representation of a statistically or metrologically equivalent pattern.

The generated pattern should not be interpreted as recovering the unique underlying high resolution experimental pattern. Information lost through optical blur and sensor sampling cannot generally be recovered uniquely. Instead, the objective is to generate new realisations that are statistically and metrologically consistent with the observed experimental pattern and reproduce its relevant behaviour when used for DIC simulation.

---

## Aims & Objectives

The aim of this project is to develop a suite of software tools for characterising experimental DIC speckle patterns and generating new high resolution and/or continuous speckle fields that reproduce the important characteristics of a pattern supplied by the user.

The fundamental problem to be addressed is:

> **Given an undersampled experimental image of a stochastic speckle pattern, infer the statistical characteristics of the underlying speckle field and generate new, higher resolution realisations that are statistically and metrologically equivalent to that field.**

An important secondary question is whether generative machine learning models provide a measurable advantage over conventional statistical characterisation and procedural reconstruction techniques.

The objectives of the project are to:

* Identify the image and spatial statistics that are important for characterising experimental DIC speckle patterns and their resulting measurement uncertainties.
* Develop software for automatically extracting these characteristics from experimental images.
* Investigate conventional statistical and optimisation based approaches for fitting procedural speckle generators to experimental patterns.
* Investigate machine learning approaches for learning compact representations of experimental speckle characteristics.
* Investigate generative models capable of producing new high resolution speckle patterns conditioned on one or more experimental reference images.
* Determine whether generated patterns reproduce the statistical properties of the reference experimental patterns - *Statisical Equivalence*.
* Determine whether generated patterns reproduce the DIC measurement behaviour of the reference experimental patterns - *Metrological Equivalence*.
* Quantify the effect of input image resolution, optical blur, noise and other imaging parameters on the inferred speckle characteristics.
* Provide uncertainty estimates or multiple plausible pattern realisations where the available experimental image does not uniquely constrain the underlying high resolution pattern.
* Integrate the resulting methods into `PyVale` through a verified, performant, documented, tested and user friendly software interface.

All work for this project should be undertaken in this github repo as a staging post and standalone research python package code before final integration into `PyVale`: https://github.com/Computer-Aided-Validation-Laboratory/specklemimic.

---

## Speckle Generation Methods

The project should consider two complementary approaches to reproducing experimental speckle patterns.

### Procedural Reconstruction

In this approach, the experimental image is analysed and a set of parameters is estimated for an existing or newly developed procedural speckle generator.

Parameters may describe characteristics such as:

* speckle density;
* speckle size distribution;
* speckle intensity distribution;
* background intensity;
* contrast;
* speckle aspect ratio;
* orientation distribution;
* speckle edge sharpness;
* minimum speckle spacing;
* clustering;
* overlap behaviour;
* spatial anisotropy;
* multi-scale pattern characteristics.

The resulting parameters can then be used by `PyVale` to generate arbitrary resolution or continuous speckle fields. This approach has the advantage that the resulting representation is interpretable, compact and potentially independent of any machine learning model once the parameters have been inferred.

### Direct Generative Reconstruction

In this approach, a machine learning model generates a new high resolution speckle pattern directly from one or more reference images. The generated output should not simply interpolate or super resolve the original pixels. Instead, it should generate a new realisation from a statistical distribution conditioned on the supplied experimental pattern. Different random seeds should therefore be capable of producing multiple distinct patterns that retain the relevant statistical characteristics of the input. The model should ideally support arbitrary or configurable output resolution rather than being restricted to a single fixed image size.

---

## Speckle Pattern Characterisation

An initial part of the project should determine which statistical descriptors are required to distinguish experimentally relevant DIC speckle patterns. Candidate descriptors should include the following but a literature search of the relevant DIC literature will be required to make this list comprehensive. A sensitivity study will be required to down sample thes to the most relevant parameters for *Statistical Equivalence* and *Metrological Equivalence* of speckle patterns.

### Intensity Statistics

* mean intensity;
* intensity variance;
* intensity histogram;
* dynamic range;
* foreground/background intensity;
* contrast;
* saturation and clipping characteristics.

### Speckle Morphology

Where individual speckles can be identified:

* speckle area;
* equivalent diameter;
* major and minor dimensions;
* aspect ratio;
* eccentricity;
* orientation;
* perimeter;
* circularity;
* edge sharpness;
* overlap between neighbouring speckles.

### Spatial Statistics

The project should investigate descriptors capable of characterising the spatial arrangement of speckles, including:

* autocorrelation;
* spatial correlation length;
* radial distribution statistics;
* nearest neighbour distance;
* clustering;
* minimum spacing or exclusion distance;
* spatial anisotropy;
* local speckle density variations.

### Spatial Frequency Statistics

The frequency content of a speckle pattern is likely to be particularly important for DIC and may be characterised using quantities such as:

* two dimensional power spectra;
* radially averaged power spectra;
* directional frequency content;
* dominant spatial scales;
* multi-scale frequency characteristics.

### DIC Specific Characteristics

The project should also investigate whether generic image statistics are sufficient to predict DIC behaviour. Additional descriptors may include:

* image gradient magnitude;
* image gradient orientation;
* local intensity variation;
* local pattern entropy;
* spatial frequency content relative to pixel spacing;
* frequency content relative to DIC subset size;
* characteristics of the DIC correlation surface;
* sensitivity of displacement estimates to pattern variations.

---

## Work Packages & Deliverables

### WP1 — Literature Review and Requirements Definition

Review existing approaches relevant to:

* statistical texture analysis;
* stochastic field reconstruction;
* image texture synthesis;
* image super resolution;
* generative modelling;
* generative adversarial networks;
* variational autoencoders;
* diffusion models;
* procedural texture fitting;
* statistical characterisation of DIC speckle patterns.

Particular attention should be given to methods that generate new statistically equivalent realisations rather than simply reproducing or interpolating an existing image.

**Deliverables**

* Literature review.
* Definition of relevant speckle statistics.
* Selection of baseline reconstruction approaches.
* Definition of quantitative validation metrics.

---

### WP2 — Speckle Characterisation Toolkit

Develop a  software tool capable of analysing experimental speckle images and reporting their statistical characteristics. The implementation may initially consist of exploratory scripts or notebooks. The toolkit should support visualisation of relevant quantities and comparison between multiple images.

**Deliverables**

* Speckle analysis software as a python package.
* Statistical descriptors for experimental patterns.
* Visualisation tools.
* Example analysis of representative experimental speckle images and synthetic speckle images.

---

### WP3 — Synthetic Training and Validation Dataset

Develop a controlled dataset of synthetic speckle patterns for which the underlying high resolution pattern and generation parameters are known. `PyVale` and its procedural speckle generation capabilities should be used where possible. Synthetic high resolution patterns should be passed through configurable imaging models to produce low resolution experimental like input images.

Relevant effects should include:

* spatial sampling;
* optical blur;
* sensor resolution;
* image noise;
* bit depth;
* intensity clipping;
* illumination variation;
* contrast variation.

The resulting dataset should provide known ground truth for evaluating the ability of different methods to infer underlying pattern characteristics.

**Deliverables**

* Automated synthetic dataset generator.
* Training, validation and test datasets.
* Dataset metadata containing known generation and imaging parameters.
* Baseline tests demonstrating the effect of image degradation on measured speckle statistics.

---

### WP4 — Procedural Parameter Inference

Investigate methods for fitting procedural speckle generators to experimental images. Initial approaches should use conventional optimisation methods and the statistical descriptors identified in WP2.

Potential methods include:

* Direct gradient-free optimisation such as genetic algorithms and particle swarms;
* Bayesian optimisation;

The objective is to determine whether an experimental pattern can be represented adequately using a compact set of interpretable procedural parameters.

**Deliverables**

* Prototype parameter fitting implementation.
* Quantitative comparison between known and inferred parameters for synthetic test cases.
* Assessment of parameter identifiability.
* Assessment of the limitations of the procedural representation.

---

### WP5 — Generative Model Development

Investigate machine learning approaches capable of learning representations of experimental speckle patterns and generating statistically equivalent high resolution realisations.

Candidate architectures may include:

* convolutional autoencoders;
* variational autoencoders;
* generative adversarial networks;
* conditional diffusion models;
* learned neural texture representations.

The project should favour relatively small, domain specific models over general purpose image generation models unless there is a demonstrated advantage to using a large pretrained model.

The model should ideally accept a reference image and requested output scale or resolution and generate a new speckle pattern conditioned on the input.

**Deliverables**

* Prototype generative model.
* Training pipeline.
* Configurable high resolution pattern generation.
* Support for multiple random realisations from the same reference input.
* Comparison between candidate model architectures.

---

### WP6 — Statistical Validation

Generated patterns should be compared quantitatively with their reference patterns. Validation should consider:

* intensity statistics;
* morphology;
* spatial statistics;
* spatial frequency statistics;
* multi-scale statistics.

The evaluation should consider distributions generated from multiple realisations rather than relying on a single generated image.

**Deliverables**

* Automated statistical validation framework.
* Comparison between procedural and generative methods.
* Quantification of statistical agreement for representative test cases.
* Identification of failure modes.

---

### WP7 — DIC Validation

Assess whether generated patterns are equivalent to experimental reference patterns from the perspective of DIC measurement performance. Reference and generated patterns should be subjected to equivalent synthetic deformation and imaging conditions. DIC should then be performed on the resulting images and the measurement errors compared.

Metrics should include, where appropriate:

* displacement bias;
* displacement variance;
* strain bias;
* strain variance;
* convergence rate;
* correlation quality;
* robustness to subset size;
* robustness to image noise.

The objective is to establish whether statistical similarity translates into **Metrological Equivalence**.

**Deliverables**

* Automated DIC validation workflow.
* Comparison of DIC errors for reference and generated patterns.
* Definition of DIC specific acceptance criteria.
* Assessment of which speckle statistics are most important for predicting DIC performance.

---

### WP8 — PyVale Integration

Convert the successful research prototypes into maintainable PyVale functionality. The final implementation should include:

* a clear public API;
* input validation;
* configuration objects;
* deterministic generation through random seeds;
* unit tests;
* regression tests;
* example scripts;
* API documentation;
* user documentation.

**Deliverables**

* PyVale software implementation.
* Automated test suite.
* Documentation.
* Examples and tutorials.
* Representative example datasets.
