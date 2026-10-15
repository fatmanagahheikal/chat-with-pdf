GOLDEN_PAPER = [
    # Page 1
    {"question": "How many people die from burns every year?", "pages": [1], "answer": "180,000"},
    {"question": "How was the spectroscopy model validated?", "pages": [1], "answer": "five-fold stratified cross-validation"},
    {"question": "Who funded this research?", "pages": [1, 15], "answer": "Scientists for Next Generation"},
    {"question": "How many burn conditions were studied?", "pages": [1, 2], "answer": "five burn conditions"},

    # Page 2
    {"question": "How accurate are burn specialists when they judge burn depth by eye?", "pages": [2], "answer": "diagnostic accuracies of only"},
    {"question": "Why do deeper burns look cooler on thermal images?", "pages": [2], "answer": "vascular destruction"},
    {"question": "Did the study fuse the two modalities into one model?", "pages": [2], "answer": "multimodal fusion is beyond the scope"},

    # Page 3
    {"question": "What environmental factors affect thermal image quality?", "pages": [3], "answer": "humidity, and airflow"},
    {"question": "Why might transfer learning from ImageNet models not work well for thermal data?", "pages": [3], "answer": "natural images differ considerably"},
    {"question": "Which model did previous work use to segment and grade burns at the same time?", "pages": [3, 14], "answer": "DenseMask RCNN"},
    {"question": "What is a practical limitation of DRS systems?", "pages": [3], "answer": "limited spatial coverage"},

    # Page 4
    {"question": "Which optical imaging techniques did previous multimodal studies combine?", "pages": [4], "answer": "SFDI and LSI"},
    {"question": "From which part of the pig were the skin samples taken?", "pages": [4], "answer": "paravertebral"},
    {"question": "Why is pig skin a good substitute for human skin?", "pages": [4], "answer": "widely recognized as an appropriate model"},

    # Page 5
    {"question": "How long was the hot probe applied to create a level 3 burn?", "pages": [5], "answer": "Level 3 (10 s)"},
    {"question": "How much weight pressed the heating probe onto the skin?", "pages": [5], "answer": "200 g"},
    {"question": "Did you confirm the burn depths using tissue histology?", "pages": [5], "answer": "histopathological analysis was not performed"},
    {"question": "What are the weaknesses of using dead pig skin instead of living tissue?", "pages": [5], "answer": "absence of blood perfusion"},
    {"question": "What temperature was the heating element kept at?", "pages": [5], "answer": "stabilized temperature of 160"},
    {"question": "What kind of laser was used as the light source?", "pages": [5], "answer": "continuous-wave laser"},
    {"question": "Why did the DRS system use one wavelength instead of a full spectrum?", "pages": [5, 6], "answer": "time-dependent changes"},

    # Page 6
    {"question": "How many frames per second did the thermal camera record?", "pages": [6], "answer": "25 Hz"},
    {"question": "Which spectrometer was used for the reflectance measurements?", "pages": [6], "answer": "Aurora 4000"},
    {"question": "How did you avoid data leakage between training and testing?", "pages": [6], "answer": "partitioned at the specimen level"},
    {"question": "Was data augmentation also applied to the test images?", "pages": [6], "answer": "applied only to the training dataset"},

    # Page 7
    {"question": "How was each reflectance measurement fed into the model?", "pages": [7], "answer": "one-dimensional input"},
    {"question": "How was the spectroscopy data organized as a table?", "pages": [7], "answer": "represents an individual measurement"},

    # Page 8
    {"question": "Which classical machine learning methods were mentioned as alternatives?", "pages": [8], "answer": "XGBoost"},
    {"question": "How many epochs was the thermal image model trained for?", "pages": [8], "answer": "10 epochs"},
    {"question": "Which loss function was used for the thermal CNN?", "pages": [8], "answer": "sparse categorical cross-entropy"},
    {"question": "What learning rate schedule was used for the DRS model?", "pages": [8], "answer": "cosine-decay-restarts"},

    # Page 9
    {"question": "What GPU were the experiments run on?", "pages": [9], "answer": "Tesla T4"},

    # Page 10
    {"question": "Did the thermal model ever mix up mild burns with the most severe burns?", "pages": [10], "answer": "non-adjacent burn levels"},
    {"question": "Was the thermal CNN overfitting?", "pages": [10], "answer": "no evidence of significant overfitting"},

    # Page 11
    {"question": "Which class had the lowest precision in the thermal model?", "pages": [11], "answer": "precision decreased to 0.88"},

    # Page 12
    {"question": "What was the most common mistake made by the spectroscopy model?", "pages": [12], "answer": "s4_L3 was misclassified as s1_L3"},
    {"question": "What training accuracy did the 1D-CNN reach?", "pages": [12], "answer": "approximately 95%"},

    # Page 13
    {"question": "Is the thermal model statistically proven to be better than the DRS model?", "pages": [13], "answer": "statistical significance test"},
    {"question": "How many classes had an AUC below 1.00 for the DRS model?", "pages": [13], "answer": "Only three classes"},
    {"question": "Should thermal imaging replace DRS?", "pages": [13], "answer": "rather than competing"},
]
