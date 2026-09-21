# 💡 Ideas

## 📌 General Ideas

**1. Use quality assessment metrics as a type of speckle statistic. These could possibly be combined to give some sort of a cost function.**

> **LG:** I think that we will have to mess around quite a bit with crafting good cost functions, that will be a mini-project by itself I think.
> That would make getting some sort of a good speckle generated/training dataset even more important.

**2. Thinking about how to use local quality parameters, maybe we can have a fixed subset and step size for each speckle pattern and then run through them all comparing the local parameter. After that, we can formulate the average of this parameter and its standard deviation as a potential loss function.**
> **LG:** These are just some ideas on how we can use local parameters too for reconstruction, we may not have to only use global parameters. It could also be some sort of a test to see if the ideal subset size we get by keeping a threshold SSSIG stays the same in our reconstructed speckle pattern.

---

## 🔭 Long Shots

*_These are ideas that are stewing for future use, not necessarily to be focused on right now._*

**1. Looking into preserving some sort of topological structure while doing the super-resolution.**

> **LG:** There is some relevant stuff on something like this in topological data analysis I think, for such data-driven approaches.
> **LG:** Maybe one topological invariant/structure that can be preserved is the mean speckle size as its such a rudimentary but important parameter to keep in mind.
