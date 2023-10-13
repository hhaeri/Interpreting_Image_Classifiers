# Interpreting an Image Classifier :blue_car: :truck: 📷

## Objective

In this project, I designed and implemented an image classifier interpretation tool that consisits of several stages:

1. Training an image classification model from scratch or utilizing a pre-trained model
2. Employing this model to generate predictions for chosen images
4. Employing model-agnostic interpretation tools to delve into the underlying rationales guiding the model's decisions

The primary emphasis of this project is to understand how Interpretable AI tools can be leveraged to gain insight about the hidden logic inside the black box AI models that influence a model's decision making. In this project I employed the Vision Explainer tool by OmniXAI (a Python library for explainable AI) to construct an image classifier model explainer.

The central focus of this project revolves around exploring the potential of interpretable AI tools to gain insights about the hidden logic inside the back box AI models that influence their decision-making processes. In this project I used the Vision Explainer tool by OmniXAI(a Python library for explainable AI) to construct an image classifier model explainer.

Further discussion on the project's background and the methodologies employed can be found in the subsequent sections.

## Table of Content

[Overcoming The Challenges of ML with the use of Interpretable AI](README.md#overcoming-the-challenges-of-ml-with-the-use-of-interpretable-ai)


* [Types of Explanations](README.md#Types-of-Explanations)
  
* [Factors of Interpretability](README.md#Factors-of-Interpretability)
  
* [Machine Learning Interpretability Techniques](README.md#Machine-Learning-Interpretability-Techniques)

[Dataset](README.md#dataset)

[Model](README.md#Image-Classifier-model)

[Image Classifier Interpretation](README.md#image-classifier-interpretation)

[Final Thoughts](README.md#final-thoughts)


## Overcoming The Challenges of ML with the use of Interpretable AI

As industries in a variety of fields, from health care to robotics to commerce become more reliant on Machine Learning models, and especially deep learning models, understanding why models reach their conclusions has become incredibly important. For policymakers, researchers, and regulators who may be making decisions based on these models, understanding the factors that influence a model's conclusion is positively essential. That's where interpretability comes in!

Interpretability is a critical aspect of assessing machine learning models, enabling evaluation for factors such as bias, fairness, robustness, privacy, causality, and trustworthiness. Incorporating interpretability into models help us to: 

* **Understand Models**

  Interpretability is essential for grasping the rationale behind model decisions, gaining insight into the model's decision-making process is invaluable for offering well-informed guidance based on AI models.
  
* **Improve Models**

  Interpretability also plays a crucial role in pinpointing the reasons behind a model's failures and making necessary adjustments to enhance its performance. 
  
* **Build Trust**

  Comprehending the rationale behind a model's decisions fosters a higher level of trust in its results. However to ensure not building false trust in the system, it is imperative to ensure that interpretability faithfully reflects the model's reasoning, genuinely elucidating why the model predicts a certain outcome, rather than fabricating explanations.
  
* **Identify Causality**

  Although “causation does not equal correlation”,  by gaining a deeper understanding of our model, we can more effectively discern causality. Recognizing causality empowers policymakers and researchers to evaluate risk factors, propose interventions, and uncover potentially discriminatory conclusions.
  
* **Determine Fairness**

  By gaining a deeper understanding of our models, we can more effectively evaluate their fairness and identify any potential discriminatory patterns by addressing algorithmic bias for a more equitable AI future.

### Types of Explanations

#### Global vs. Local


Global explanations offer an insightful way to evaluate a model's overall performance across different dimensions. They provide an explanation of how a model makes decisions overall. For instance, linear regression and decision tree models inherently provide global explanations. Conversely, local explanations serve the purpose of elucidating a specific prediction. This approach delves into the factors influencing a single decision, utilizing tools like heatmaps or rationales to provide clarity and context for that particular outcome.

#### Inherent vs. Post_hoc

Inherent explanations are seamlessly integrated into the model, while post-hoc explanations necessitate the application of external techniques. Inherent explanations are the most straightforward to acquire because they are an integral part of the model itself. Models like linear regression, decision trees, and natural language explanations are inherently interpretable, as they inherently embed information about the decision boundaries and the reasoning behind a classifier's outputs.

Conversely, post-hoc explanations are more intricate and come into play with black-box models. These models, such as neural networks, don't readily reveal their decision-making process. Therefore, to comprehend how such models arrive at their decisions, we must employ external methods to extract explanations that are intelligible to humans.

### Factors of Interpretability
Here are the factors to consider when creating good interpretable system:

* **Faithfulness**: Do the explanations accurately represent the true reasoning behind the model’s final decision?

* **Plausibility**: Is the explanation correct or something we can believe is true, given our current knowledge of the problem?

* **Understandable**: Does the explanation use terms that an end user without in- depth knowledge of the system can understand?

* **Stability**: Do similar model examples have similar interpretations?

### Machine Learning Interpretability Techniques

Here I will use three common techniques for generating model explanations: LIME , SHAP , and gradients-based interpretability methods.

#### LIME: Local interpretable model-agnostic explanations

LIME generates simplified, interpretable surrogate models, such as linear regression or decision trees, for individual predictions by perturbing the input data and observing how the model's output changes. These surrogate models approximate the behavior of the complex model for that instance. By analyzing the coefficients or structure of the surrogate model, users can gain insights into which features (variables) had the most significant impact on the model's prediction for that instance. This information can help identify the key factors driving the decision. LIME's explanations make it easier for users to trust the predictions of complex models and debug them when necessary. It provides transparency and helps ensure that model decisions align with domain knowledge and expectations, making it easier to trust and debug machine learning models, particularly in critical applications like healthcare and finance. LIME is a model agnostic tool for model interpretability, it bridges the gap between complex machine learning models and human understanding by offering local, instance-specific explanations for model predictions.

#### SHAP: SHapley Additive exPlanations

SHAP is a versatile and model-agnostic interpretability framework that goes beyond traditional feature importance methods to provide a more nuanced understanding of how individual features influence model predictions. At its core, SHAP is based on concepts from cooperative game theory, specifically the Shapley value, which is used to fairly distribute contributions among players in a cooperative game. In the context of machine learning, each feature is treated as a "player," and SHAP calculates the Shapley value for each feature to quantify its contribution to a prediction. Ulinke LIME, SHAP offers both local and global interpretability. Local explanations help users understand why a specific prediction was made by quantifying the impact of individual features for that instance. Global explanations provide an overview of feature importance across the entire dataset, allowing users to identify trends and patterns.

#### Integrated Gradients

Integrated Gradients is an interpretability method used to understand how individual features or pixels in input data contribute to a machine learning model's predictions, particularly deep neural networks. It works by integrating gradients along a path from a baseline input to the actual input, calculating feature attributions. These attributions indicate the importance of each feature and can be visualized to highlight influential regions in data, making it useful for understanding complex model decisions in tasks like image analysis and natural language processing. Integrated Gradients aids in model debugging and building trust in AI systems. While Integrated Gradients can provide valuable insights into the importance of features for individual predictions, it doesn't offer a global feature importance summary across an entire dataset or model. 

## Dataset

In this project, I utilized images of trucks and cars sourced from the extensive COCO (Common Objects in Context) dataset to showcase the interpretability capabilities of model-agnostic explainers I introduced earlier (SHAP, LIME, and IG). The COCO dataset boasts an impressive scale, comprising over 200,000 images and containing more than 1.5 million object instances. These images encompass a wide array of scenes and situations, meticulously annotated across approximately 80 distinct object categories. Notably, COCO's annotations encompass over 330,000 object instances, precisely delineating bounding boxes, and it includes textual descriptions for nearly 91,000 images. 

Here is one example image from the dataset I will use to explain the image classifier model discussed in the next section:

![image](https://github.com/hhaeri/Interpreting_Image_Classifiers/assets/91407046/7c2c8953-fdec-4c2f-840d-045d7654ffc9)

## Image Classifier Model

The model considered here is a ResNet model pretrained on ImageNet, renowned for its capabilities in image classification. ImageNet (a large-scale dataset with 14 million of labeled images spanning more than 20,000 categories) has played a pivotal role in training deep learning models for various vision tasks. By leveraging the transfer learning provided by the pretrained ResNet model, we can apply its learned features to our specific image classification problem, enhancing our model's ability to recognize and differentiate between objects in the COCO dataset. 

## Image Classifier Interpretation

For interpreting a classification task, there are multiple dimensions to choose from as explained above (Global vs Local, Model agnostic vs. specific, Inherent vs. post hoc). I will be using a Model agnostic post hoc method and deploy it at a local scale.

Specifically, I will use LIME, SHAP, and integrated-gradient in this project. For each of these algorithms, I will be documenting the compute time and visualizing their explanations. At the end of the project, I'll be comparing the three evaluation approaches and assessing which I agree with most.

To construct my image classification model explainer, I employed the Vision Explainer tool by OmniXAI. OmniXAI (Omni eXplainable AI) is a Python library for explainable AI (XAI). OmniXAI aims to be a one-stop comprehensive library that makes explainable AI easy for data scientists, ML researchers and practitioners who need explanation for various types of data, models and explanation methods at different stages of ML process. The class VisionExplainer is designed for vision tasks. VisionExplainer provides a unified easy-to-use interface for all the supported explainers such as LIME, SHAP, IG, etc. 

to initialize VisionExplainer, we need to set the following parameters:

* explainers: The names of the explainers to apply, e.g., [“lime”, “ig”, “shap”].

* model: The ML model to explain, e.g., a scikit-learn model, a tensorflow model, a pytorch model or a black-box prediction function.

* preprocess: The preprocessing function converting the raw data (a Image instance) into the inputs of model.

* postprocess (optional): The postprocessing function transforming the outputs of model to a user-specific form, e.g., the predicted probability for each class.

* mode: The task type, e.g., “classification” or “regression”.

After the explainations are created by OmniXAI, we can then plot the generated explainations in IPython:

Model Prediction:
![newplot](https://github.com/hhaeri/Interpreting_Image_Classifiers/assets/91407046/2352eda3-321a-40b2-ace9-50c3a0b8db67)
LIME:
![newplot (7)](https://github.com/hhaeri/Interpreting_Image_Classifiers/assets/91407046/a6e993a7-1552-48ba-9626-1797653bb898)
SHAP:
![newplot (8)](https://github.com/hhaeri/Interpreting_Image_Classifiers/assets/91407046/d36d36c4-ed72-4ccb-8f85-2a85f47894ea)
Integrated Gradient:
![newplot (9)](https://github.com/hhaeri/Interpreting_Image_Classifiers/assets/91407046/3f5ad744-f3fb-4b0d-b212-518cce0543e7)


### Final Thoughts

By generating and visualizing the local explanations for couple of images, we can gain knowledge about the hidden logic inside the black box AI models. LIME's heatmaps highlighs parts of each image that led to the prediction. Similarly, SHAP and integrated gradients overlay and score images depict the regions that influence the model's predictions. The integrated gradients explainer method did a better job of interpreting images because it takes a more comprehensive approach. By considering the model's behavior across all pixels, integrated gradients can identify the most relevant pixels that consistently influence predictions. This provides a robust importance score for each pixel in the entire image. In contrast, SHAP and LIME are more localized in their explanations, so they may not reveal global patterns. The universal view of integrated gradients gives it an advantage for highlighting what the model deems most important. on the other hand, For a non-technical audience, for this particular application, I think it’s clear that the Integrated Gradients (IG) saliency map would be most easily understood. A more technical audience might appreciate the nuances of the SHAP & LIME values. 

The ResNet architecture exhibits remarkable predictive capabilities when applied to the car images tested in this project. This proficiency can be attributed to its extensive training on ImageNet, a dataset encompassing a staggering 14 million labeled images, spanning over 20,000 diverse categories. Utilizing model explainer tools, I have observed the model's ability to identify important features in the images that correspond to cars, such as windshields, side mirrors, and fenders, etc. Furthermore, the model adeptly boxes and locates the actual car within large images. Nonetheless, when the same model is tasked with detecting cars in distorted images or unconventional environments (e.g., a jungle instead of a road), its accuracy diminishes and is unable to detect cars correctly. In such scenarios, the model's explainers serve as valuable resources for dissecting the reasons behind the model's failure to provide accurate predictions. 

For a more comprehensive understanding of this project, please consult the complete Colab notebook of my project, accessible [here](InterpretingML_ImageClassifier_4git.ipynb).
