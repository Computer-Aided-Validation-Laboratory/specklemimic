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
