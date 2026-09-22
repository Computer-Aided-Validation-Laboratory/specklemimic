# 📝 Summaries

This markdown file gives summaries of various papers that I have read. I have tried to keep these summaries/takeaways short and sweet. 

It may be required in the future to split this file into a bunch of smaller markdown files based on the specific topic (depending on how good this literature review goes!)

---

## 🔬 Speckle Pattern Statistics

**1. Dong, Y. L., and B. Pan. “A Review of Speckle Pattern Fabrication and Assessment for Digital Image Correlation.” Experimental Mechanics 57, no. 8 (2017): 1161–81. https://doi.org/10.1007/s11340-017-0283-1.**

> This paper is honestly really amazing! It is a good source of literature for various speckle pattern fabrication techniques and quality assessment metrics. 
> Obviously, since it is a review paper it doesn't explain stuff as well as the individual papers but it is a great hub/collection of information.

**2. Yaofeng, Sun, and John H. L. Pang. “Study of Optimal Subset Size in Digital Image Correlation of Speckle Pattern Images.” Optics and Lasers in Engineering 45, no. 9 (2007): 967–74. https://doi.org/10.1016/j.optlaseng.2007.01.012.**

> This paper introduces speckle entropy as a local quality assessment parameter. 
> I think its a bit distracted in its writing so the biggest takeaway for me is just how the speckle entropy is calculated by looking at nearby pixels.

**3. Pan, Bing, Huimin Xie, Zhaoyang Wang, Kemao Qian, and Zhiyong Wang. “Study on Subset Size Selection in Digital Image Correlation for Speckle Patterns.” Optics Express 16 (May 2008): 7037–48. https://doi.org/10.1364/OE.16.007037.**

> This paper is very thorough. The mathematically motivated derivation of SSSIG is really good compared to something like speckle entropy which is very empirically derived. 
> It basically tries to mathematically minimize a cost function b/w deformed subsets (assuming a first order response) and adding some noise terms. 
> They carry out the mathematical evaluation here and then find that the overall sensitivity to noise (variance) has SSIG as a denominator, so they propose maximizing SSSIG for subset size.

**4. Pan, Bing, Zixing Lu, and Huimin Xie. “Mean Intensity Gradient: An Effective Global Parameter for Quality Assessment of the Speckle Patterns Used in Digital Image Correlation.” Optics and Lasers in Engineering 48, no. 4 (2010): 469–77. https://doi.org/10.1016/j.optlaseng.2009.08.010.**

> This paper is a generalization of SSSIG over the whole domain to give something known as a mean intensity gradient.
> It looks like alongside SSSIG, MIG is also used as a dominant metric to assess speckle pattern quality. This makes sense because both of them are very well motivated mathematically.

**5. Lecompte, D., A. Smits, Sven Bossuyt, et al. “Quality Assessment of Speckle Patterns for Digital Image Correlation.” Optics and Lasers in Engineering 44, no. 11 (2006): 1132–45. https://doi.org/10.1016/j.optlaseng.2005.10.004.**

> This paper has good numerical results on how the speckle size and subset size are directly correlated that could have a decent impact on our project.
> I would assume that if the newly generated speckle pattern did not conserve speckle size, the DIC performance would be significantly different and this paper just shows how that can be true.

**6. Liu, Xiao-Yong, Rong-Li Li, Hong-Wei Zhao, et al. “Quality Assessment of Speckle Patterns for Digital Image Correlation by Shannon Entropy.” Optik 126, no. 23 (2015): 4206–11. https://doi.org/10.1016/j.ijleo.2015.08.034.**

> This paper talks about using Shannon entropy for speckle quality assessment. Shannon entropy basically encodes the information content of the speckle pattern.
> The larger the Shannon entropy, the better the speckle pattern performance in the numerical experiments that were conducted.

**7. Bossuyt, Sven. “Optimized Patterns for Digital Image Correlation.” Conference Proceedings of the Society for Experimental Mechanics Series 3 (January 2013): 239–48. https://doi.org/10.1007/978-1-4614-4235-6_34.**

> This paper was honestly very important in that it introduces a very nice way to look at correlation and speckle patterns.
> It basically shows how we can write the sensitivity and robustness of the model in terms of the correlation between patterns displaced by a known quantity.
> In practice, autocorrelation is used to find the "correlation landscape" and we can define the autocorrelation peak radius and watershed radius using the resulting data.

**8. Stoilov, Georgi, Vasil Kavardzhikov, and Dessislava Pashkouleva. “A Comparative Study of Random Patterns for Digital Image Correlation.” Journal of Theoretical and Applied Mechanics 42 (June 2012): 55–66. https://doi.org/10.2478/v10254-012-0008-x.**

> This paper discusses the autocorrelation peak comparison metric which compares the primary and secondary correlation peaks.
> This metric is interesting but has some drawbacks such as being biased towards secondary peak length (instead of quantity) and potentially taking up a lot of compute.

**9. Bomarito, G. F., J. D. Hochhalter, T. J. Ruggles, and A. H. Cannon. “Increasing Accuracy and Precision of Digital Image Correlation through Pattern Optimization.” Optics and Lasers in Engineering 91 (April 2017): 73–85. https://doi.org/10.1016/j.optlaseng.2016.11.005.**

> This paper is very well written and researched. It discusses how to use an optimization technique to generate a speckle pattern into a fixed subset size.
> The methods developed here are pretty cool and also it talks about how multiple quality assessment parameters can be compared to give good results.
> I think one thing they haven't checked is which parameter here is super important in its translation to actual DIC accuracy.
