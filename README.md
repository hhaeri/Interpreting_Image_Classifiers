# Interpreting_Image_Classifiers
Interpreting Image Classifiers using Vision Explainer by OmniXAI

## What is Interpretable AI?

As industries in a variety of fields, from health care to robotics to commerce become more reliant on Machine Learning models, and especially deep learning models, understanding why models reach their conclusions has become incredibly important. For policymakers, researchers, and regulators who may be making decisions based on these models, understanding the factors that influence a model's conclusion is positively essential. That's where interpretability comes in!

To ensure we understand why our models make the decisions they do, we can add an interpretability step between the train and evaluate phases! Of course, since the ML development process is iterative, we won't just be considering interpretability once. Instead, we'll be revisiting it throughout the development process.

<img width="528" alt="image" src="https://github.com/hhaeri/Interpreting_Image_Classifiers/assets/91407046/01d82db3-f9b5-4bf8-a687-1b6ca491be15">

## Overcoming The Challenges of ML with the use of Interpretable AI

Interpretability makes it possible to assess our ML models for factors like bias and fairness, robustness, privacy, causality, and trustworthiness. For example, interpretability can be essential for ensuring the safety and well-being of the people impacted by the model. For example, ML is often used in high-stakes settings where poor performance can lead to harmful outcomes. In healthcare, practitioners need to understand a model's performance to ensure proper care for patients. When it comes to autonomous driving, we don't want the vehicle to hit a parked car, or worse yet, a pedestrian.

Interpretability also plays a vital role in rooting out bias in algorithms. One recent example of this was the identification of facial and gender bias in the Amazon Recognition Commercial System. Biases like these can have real-world ramifications – In the financial sector, for example, bias can impact who is approved for credit or loans.

Without interpretability, we can't trust our models – We won't know what our model will predict in extreme cases, and we'll be poorly equipped to identify specific shortcomings or biases. We'll also find it difficult to identify adversarial examples. These are examples that have been engineered to cause our model to fail. By incorporating interpretability into our model, we can see whether it makes similar mistakes as humans, and we can be better equipped to update the model when things go wrong. Interpretability can help us to understand models, improve models, build trust, identify causality, and determine fairness.

- Understand Models

  Fundamentally, interpretability is essential for grasping the rationale behind model decisions. When ML models are employed by policymakers to recommend budget allocations for schools or public services, it's imperative that they comprehend the underlying factors shaping these decisions to ensure a fair distribution of resources. Moreover, in the realm of epidemiology, where ML models predict public health outcomes, gaining insight into the model's decision-making process is invaluable for offering well-informed public health guidance and formulating effective policy proposals.
  
- Improve Models

  Interpretability also plays a crucial role in pinpointing the reasons behind a model's failures and making necessary adjustments to enhance its performance. For instance, in the context of sentiment analysis on text, we can examine whether there are common patterns shared among the instances where our model makes errors. Similarly, in the application of ML models in robotics for tasks like pick-and-place, we can identify specific components that the model frequently fails to recognize. Once we've identified these limitations, interpretability provides a valuable means to rectify them, as addressing issues becomes challenging without understanding their root causes. Furthermore, the insights gained from interpretability are instrumental in detecting and mitigating biases within our algorithms, not only enhancing overall model performance but also ensuring fairness across different subpopulations.
  
- Build Trust

  Comprehending the rationale behind a model's decisions fosters a higher level of trust in its results. For situations where ML is employed for cancer detection; healthcare professionals are more likely to place their trust in the model if it offers an explanation rather than remaining an enigmatic black box with an unknown outcome. However, this trust-building process must be approached with caution. It is imperative to ensure that interpretability faithfully reflects the model's reasoning, genuinely elucidating why the model predicts a certain outcome, rather than fabricating explanations. Otherwise, we risk building false trust in the system.
  
- Identify Causality

  “Causation does not equal correlation.”  In essence, this means that when two factors appear related and coincide in time, it doesn't automatically imply that one is causing the other.  Thus it is important to not jump to causal conclusions based solely on observed correlations, as there may be hidden confounding factors that explain the relationship between variables. Conversely, when it comes to the link between smoking and an increased cancer risk, there is a clear causal relationship since smoking directly elevates the risk. By gaining a deeper understanding of our model, we can more effectively discern causality. For example, if a classifier predicts "y" based on input feature "x," does this signify that "x" causes "y"? Recognizing causality empowers policymakers and researchers to evaluate risk factors, propose interventions, and uncover potentially discriminatory conclusions.
  
- Determine Fairness

  By gaining a deeper understanding of our models, we can more effectively evaluate their fairness and identify any potential discriminatory patterns. Real-world examples in finance and the legal system illustrate how algorithmic bias can lead to unjust outcomes. For instance, a 2021 study revealed disparities in loan interest rates and approvals between minority and white borrowers, even after accounting for income. In the justice system, machine learning-generated risk assessments influence decisions like bail and sentencing, but investigations have exposed biases, with black defendants being flagged as future offenders at a higher rate than white defendants. This underscores the urgency of addressing algorithmic bias for a more equitable AI future.

## Types of Explanations

### Global vs. Local


Global explanations offer an insightful way to evaluate a model's overall performance across different dimensions. They provide an explanation of how a model makes decisions overall. For instance, linear regression and decision tree models inherently provide global explanations. Conversely, local explanations serve the purpose of elucidating a specific prediction. This approach delves into the factors influencing a single decision, utilizing tools like heatmaps or rationales to provide clarity and context for that particular outcome.

### Inherent vs. Post_hoc

Inherent explanations are seamlessly integrated into the model, while post-hoc explanations necessitate the application of external techniques. Inherent explanations are the most straightforward to acquire because they are an integral part of the model itself. Models like linear regression, decision trees, and natural language explanations are inherently interpretable, as they inherently embed information about the decision boundaries and the reasoning behind a classifier's outputs.

Conversely, post-hoc explanations are more intricate and come into play with black-box models. These models, such as neural networks, don't readily reveal their decision-making process. Therefore, to comprehend how such models arrive at their decisions, we must employ external methods to extract explanations that are intelligible to humans.

## Factors of Interpretability
Here are the factors to consider when creating good interpretable system:
**Faithfulness**: Do the explanations accurately represent the true reasoning behind the model’s final decision?
**Plausibility**: Is the explanation correct or something we can believe is true, given our current knowledge of the problem?
**Understandable**: Does the explanation use terms that an end user without in- depth knowledge of the system can understand?
**Stability**: Do similar model examples have similar interpretations?

## Interpreting Image Classifier

Here I will use three common techniques for generating model explanations: LIME , SHAP , and gradients-based interpretability methods.

### LIME: Local interpretable model-agnostic explanations
