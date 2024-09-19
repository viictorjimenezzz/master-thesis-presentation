# Introduction

- Good morning, thanks a lot for being here (for your time).

- I am Victor Jimenez, and I will present my master thesis which is titled "Improved robustness of deep learning models through posterior agreement based model selection".



# Experimental pathway

This thesis stems from previous work from Joao and Alessandro, in which they proposed the use of posterior agreement for the assessment of robustness in image classification tasks; in particular under covariate shift, so basically adversarial attacks and distribution shifts.

So the main goal of this project was to provide further evidence in favour of this hypothesis (so with more moreld, more experiments) and evaluate whether this intuition extends not only to the assessment of robustness but also to the selection of robust models via early stopping, in the same setting.

And this is the work that I will present to you. 

The experimental pathway, as also is more or less introduced in this presentation.

1. The first step was the formalization/characterization of the different sources of randomness that want to make models to be robust to. It was important to common approach to encompass all of them under the framework of probaility theory.

2. The next step is to derive an operative version of PA for finite hypothesis classes so that is suitable for image classification. Have versatile and efficient implementation, so that it can be computed every epoch, and of course dervie its properties both analytically and empirically.

3. Then the first experimental setting...The first experiments would be conducted on the adversarial setting. Adversarial perturbations are easy to characterize from a data perspective and also are designed to modify the predicted class, so the contcept of robustness in this setting aligns with performance or accuracy-based metrics. In the sense that a model will be more robust the better it performs under adversarial attacks.

4. Then the assessment was extended to out-of-distribution setting, in which the characterization of the covariate shift is not standard, BUT and in particular under a specific data generation process that the specific parameters that determine the shift. Basically so that we can control the nature of the randomness that each pair of samples entails.

5. And finally we extended the suitability of these results to the selection of models, fist under controlled conditions and finally in some benchmark robustness datasets.



# The robustness challenge

So let's start with an introduction to the core problem

## Introduction

In a broad sense, a model is considered robust if it can maintain its predictive power in observations that present some kind of variation or perturbation from the ones it has seen during training.

Here is a very swiss illustration of the sources of randomness that are relevant in our context. If a model is trained on these original samples, then sampling randomness referes to other instantiations of the same experiment. So generalization to these variations is taken for granted, given that the performance is usually reported on validation or test subsets of the same data source.

But then there are two other sources of randomness that are not accounted for, and that conform our covariate shift setting, which are adversarial perturbations and distribution shifts. 

The first one corresponds to adversarial perturbations, that consist of small inperceptible perturbations that exploit the vulnerabilities of the model in the feature space and are design to mislead models.

And the second case refers to the situation in which the assumption that the data is identically distributed over the support does not hold for numerous reasons, and some data for test does not follow the same statistical patterns that the model has learned, thus impeding generalization. In this case, this image breaks the spurious correlation between cows and rural backgrounds.

## Challenges

- At the core of the robustness challenge lies the poor understanding of how models construct their inductive bias and the nature of the transformations between the space of weights and the space of functions that they are able to represent. 

- It has ben shown that features learned by the optimal standard classifier can be completely different from those learned by a robust classifier, regardless of the amount of data provided, which results in a fundamental limitation for task performance. 

- Besides, the feature space that models navigate is different from that in which we perceptually rely on, and therefore we should not expect models to be invariant to the same features we humans are naturally invariant to.

- And on top of that there is of course the classic bias-variance trade-off. The more expressive are the models unders consideration, the greater is the overfitting risk.  Intuitively, robnust learning entails decision boundaries that are more complex than the ones derived via standard training, intuitively demanding more data and more complex architectures, at the risk of overfitting that this entails.

- Here is a very illustrative example in the adversarial setting. A very complex decision boundary has to be learned to separate linearly-separable points if we want the model to be robust to Manhattan norm perturbations.




# Learning framework

Considering the challenge, what is our learning framework

## Introduction

The task to perform is to learn a target function expressed as a transformation between an input space and an output space. In our particular case,

- The input space are images and the output are classes or labels.

- The function class are classifiers parametrized with neural networks.

- And most importantly we assume that the target function encodes the causal structure that generates the data. Which means that is invariant and independent of the availability of the data.

## Sample

From a probability theory perspective, we define samples in the following way.

- We consider a random variable X associated with a sampling experiment over the input space, that entails a measure of probability in the space, uniform a priori.

- We consider a simple random sample of X, which is a random vector of independent and identically distributed random variables as X, and we perform a sampling experiment that yields a sample.

- And we construct supervised datasets by associating the sample to its true output space value with the target function.

## Model

As I said the model will be parametrized classifiers, and I write this just to mention that we have access to this intermediate steps, which are the feature space representations, and the discriminant function that generates the log-odds for every class. The decision rule amounts to selecting the class with the highest log-odds.

## Algorithm

And our baseline learning algorithm will be empirical risk minimization, possibly with some regularization term, using the cross-entropy loss function.



# Posterior Agreement

As said before, this project explores the use of posterior agreement as a robustness metric. I will not extend myself so much in this for obvious reasons, but I have to deliver the same presentation in barcelona so I need to keep these slides.

## 1)

Only to state some definitions, we define the hypotheses class as the set of possible outcomes of a sampling experiment, and the posterior given a function models the stochastical relation between realizations of the experiment and hypothses.

## 2)

Under this framework, a generalization error arises from a concept of stability, in the sense that two different realizations of the same experiment should give rise to the same model, and therefore to the same probability over the hypothesis class. So stability in the hypothesis class is desired.

Generalization-complexity trade-off from this perspective by measuring complecity is the informativeness of the function, which represents its ability to learn the patterns in the data while filtering out the noise. The more expressive (i.e. complex) a function class is, the higher will be the estimated information content of the data. If the information content is underestimated, the approximated function will lack the capacity to learn some patterns in the data, whereas if informativeness is overestimated, it will overfit to the noise and thus not generalize to different realizations of the experiment.

The posterior agreement evaluation navigates this stability-informativeness trade-off by adjusting the posterior distributions, and a suitable measure of hypothesis quality is the description length.



## 3)

It can be shown that we can bound for this expression of generalization error that is the negative posterior agreement, and the model selection criterion stems from maximizing the posterior agreement.

## 4)

For reference, we converge to an expression of the PA kernel in which posterior distributions are gibbs functions with inverse temperature parameter beta.

## 5)

Some interesting properties of this expression is that is bounded by the uniform postarior case, it's symmetrical with respect to the data, which is a good property for a robustness measure, and also it can be proven that it has a unique maximum.

In particular, the maximum of PA which is the metric we will work with, is strictly bounded by two possible configurations of the samples, one in which samples are completely non-matching, and therefore the maximum agreement occurs with flat posteriors at beta=0, and another with all samples matching, in which case PA converges to zero.

## 6)

This is an illustration of the optimization process and the posterior agreement concept, in which posterior agreement is maximized if high-likely posteriors under x' are assigned a high probability under x''.



# Robustness against covariate shift

As we mentioned before, we assume that the target function is invariant, so we consider that we have a covariate shift whenever the generalization error is non-zero. In particular, two samples from the same experiment could encode different features and the distribution over the features would then be shifted.

In a distribution shift, x′ and x′′ are drawn from different random variables, each with
a distinct probability landscape over the support which result in implicit differences in the distribution of some features. We will usually call each random variable a domain, because of the implicit selection over the support.

In adversarial shifts, however, a single sampling experiment is performed  and the perturbation is added over the sample space to generate x'', so the randomness expected is totally different.

## Robustness metric

Under the framwork we described, the ultimate goal of robustness measurement is the
characterization of the resolution limit that can be achieved in the hypothesis space consistent with the intrinsic randomness entailed by each possible realization of the experiment.

Therefore, a robustness metric should evaluate how stable are hypothesis to different realizations of the same experiment in a model-agnostic way. As model complexity increases, so does the number of hypotheses navigated, but this also increases the risk
of overfitting to sampling randomness, leading to unstable hypotheses. 

- In that sense, we argue that an ideal robustness metric should be non-increasing with the response of the model under increasing levels of shift,

- And also discriminate models exclusively by their generalization capabilities, independent for instance of the task performance of the model.

This already poses a significant issue in the common 
