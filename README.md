# IDEAS_PhD-work
Evaluation is a vital task in the teaching-learning process. Automatic Short Answer
Grading (ASAG) is an ever-increasing realm in natural language understanding. It
aims to ease the challenges of teaching, particularly in classrooms with numerous
students, where assessing brief descriptive responses poses a significant challenge.
The current research work proposes a model answer-based framework for ASAG,
namely IntelliGrader, which addresses three key evaluation facets: i) automatic
scoring of short, descriptive answers written in English, ii) identifying the
inconsistency in assessment, and iii) providing explicable feedback concerning the
inconsistently evaluated answers to the evaluator.
Here, the automated scoring is achieved using a model answer-based (reference
answer) approach, which involves performing a collaborative analysis of eight crucial
features. Initially, the ASAG task was treated as a regression task, and the models
were built using various state-of-the-art regressors by considering eight independent
features and the actual mark given by the evaluator as a dependent feature. The
proposed system has undergone thorough validation and testing across four
benchmark datasets: Automatic Student Assessment Prize-Short Answer Scoring
(ASAP-SAS), SciEntsBank, STITA, Mohler (Texas), and a primary dataset, IDEAS.
The experimental results show the regression results with the finest RMSE of 0.09
concerning the STITA dataset and 0.19 RMSE for the IDEAS dataset for a particular
question. Furthermore, a comparative study of the regression model results
concerning the existing work, GradeAid, is performed. The Normalized Root Mean
Square Error (NRMSE) results of various datasets proved promising for IntelliGrader
compared to GradeAid (existing work). Even though the results of regression models
are promising, showing the importance of the extracted features, the scores given are
real numbers like 2.34, 1.567, etc., which are not similar to human cognition and
behavior (humans provide scores in the form of an integer like 2, 5, 8, etc.). Also, the
regression models need additional discriminators (techniques alike the nearest integer)
to modify the real number into an integer.
To overcome the limitations of regression models, ASAG is considered a multiclass
classification problem instead of regression, wherein every mark is treated as a label.
Various cutting-edge machine learning classification models, such as Naïve Bayes (NB), K Nearest Neighbor (KNN), Decision tree (DT), Support Vector Machine
(SVM), Random Forest (RF), and XGBoost (XGB) are built by considering the eight
extracted features as independent and the actual score given by the evaluator as a
dependent feature. These models are validated on various benchmarks (ASAP-SAS,
SciEntsBank, STITA, Mohler (Texas), and a new dataset, Intelligent Descriptive
answer E-Assessment System (IDEAS). Experimental results reveal that the
performance of Random Forest and XGBoost classifiers dominates other classifiers
concerning all classification metrics: accuracy, precision, recall, and F1 score. The
range of accuracy spans from 65% to 95% for different questions across multiple
datasets. Results proved that treating ASAG as a multiclass classification problem is a
valid choice.
Inconsistency identification is achieved using K-Means for cluster creation and tdistributed
Stochastic Neighbor Embedding (t-SNE) for cluster visualization. The
proposed methodology is validated on various datasets, such as ASAG-SAS,
SciEntsBank, STITA, and the new dataset IDEAS. The comparative analysis of
inconsistency is accomplished in three vital cases: within the actual marks given by
the evaluator, within the best model predicted marks (both regression and
classification), and between the actual marks and the predicted marks. Experimental
results on the IDEAS dataset, with 800 answers for 20 questions, show that multiclass
classification models had 85 inconsistent answers, while regression models had 107,
compared to 108 for human-evaluated scores. This indicates that multiclass
classification models demonstrate less inconsistency, highlighting their performance
in automated scoring.
Once inconsistency is identified in the evaluation, the evaluator is provided with a
comprehensive explanation of feedback regarding keywords and various similarities
concerning inconsistent answers. This feedback includes Student ID (STDID),
matched and unmatched keywords between the model and student answer, similarities
like cosine, statistical, semantic, and summary, and actual and predicted marks. This
feedback aids the evaluator in rectifying the errors in the evaluation.
