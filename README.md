# DeConDFFuse


## Abstract

Drug-Drug interaction, presently, is an essential pharmacological aspect to be known and understood. Usually, the interaction means that when two or more drugs react with each other, this could result in unexpected side-effects or adverse consequences to a person's health. Predicting these interactions precisely will help reduce such adversities, and thus, it becomes significant. Currently, the techniques proposed in this direction are generally based on either Machine Learning paradigms like Random Decision Forest (RDF), Support Vector Machines(SVM), etc., or Convolutional Neural Networks in Deep Learning. However, specific works combine traditional machine learning algorithms like RDF, LR, SVM, etc., and deep learning paradigms like CNNs in a piecemeal fashion. Hence, the present work aims to propose a representation learning based framework that is inspired from our recently established work - Deep Convolutional Transform Learning (DeConFuse), and jointly optimizes the decision forest inspired from Deep Neural Decision Forest [(dndf)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Kontschieder_Deep_Neural_Decision_ICCV_2015_paper.pdf). This framework is such that (i) it is a supervised multi-channel framework, (ii) that learns the unique filters, (iii) that fuses the information of the two input drugs to learn the sparse representations from the same, and (iv) gives the final predictions from decision forest that are jointly and globally optimized. We apply this technique to 1059 drugs from the DrugBank database and find that the proposed work gives good results compared to the state-of-the-art(s).


## Acknowledgement 
The authors acknowledge support from Inria Saclay, through the International Inria Team program COMPASS.







