# Mini Project - Natural Language Processing (NLP) for Hebrew Syntax

## Background & Introduction
The construct of "Smichut" (construct state) is one of the unique features of the Hebrew language, where two nouns, usually in a specific syntactic relationship, form a singular meaningful unit. In this construction, the first noun is called the "nesmach" (head) and the second noun is called the "somech" (modifier). 

- **Definiteness in Smichut Construction**: In Hebrew, the definite article appears only before the second word of the pair. For example, "מכונת הכביסה" (washing machine), where "הכביסה" (the washing) is the second noun.
  
- **Pluralization and Gender Agreement**: In Smichut constructions, the plural form applies only to the first noun (the head), and gender is also assigned to the head noun, not the modifier.

- **Semantic Roles**: Smichut constructions can express a wide range of relations, such as possession, material, purpose, or entity (e.g., "שעון זהב" - gold watch, "גני ילדים" - kindergartens).

## Research Question
The main research question addressed in this project is: **Can we classify Hebrew Smichut constructions effectively using a classifier?**

Additional sub-questions that guide the project include:
- What combination of linguistic features can be used to classify Smichut constructions optimally?
- Should the classifier focus only on the words in the Smichut construction or also consider other words in the sentence?
- How does human classification of Smichut constructions compare to AI classifiers?

The project explores whether there are any differences in classification patterns depending on the type of text (medical articles were chosen for this study).

## Project Description
The project follows a series of steps to classify Hebrew Smichut constructions:

1. **Manual Labeling**: The first step involved identifying 1,000 Smichut constructions from medical articles and labeling them manually. This process posed challenges due to differing approaches in labeling among team members and difficulties distinguishing between similar categories.
  
2. **Vector Creation**: After labeling, we created 29 distinct feature vectors based on the linguistic properties and their combinations in different contexts (windows) within the sentence.

3. **Training Classifier**: We used a k-fold classifier (with k=10) to evaluate the performance of the classifier. We iterated the training process 10 times, using 90% of the data for training and 10% for testing.

4. **Evaluation**: Each vector type was scored based on the classifier's performance during the testing phase.

## Feature Descriptions
The study focused on five linguistic features:
1. **Word**: The actual word as it appears in the sentence.
2. **Lemma**: The dictionary form (lemma) of each word.
3. **Lexical Category**: The syntactic role of each word, such as noun, verb, adjective, etc.
4. **Syntactic Features**: The word’s syntactic role in the sentence, including its position (subject, object, etc.).
5. **Context Window**: A window around the Smichut construction, considering surrounding words (before and after).

## Results & Findings
### Vector Description
The project evaluates a set of 30 feature vectors for each Smichut construction. These vectors, which combine various linguistic properties, were used to train the classifier. Each vector type was evaluated based on the classifier’s performance during the testing phase.

### Vector Scores
The vectors were evaluated based on the success of the classifier during the testing phase. The results of these evaluations were saved and analyzed to determine the optimal vector configuration.

## Statistics
Statistics related to the vector performance, training accuracy, and classifier evaluation can be found in the results section. We used standard evaluation metrics such as accuracy, precision, recall, and F1-score.

## Analysis
In the final analysis, we compared the classifier’s performance with human classification and evaluated how different features (such as context and lexical category) affected the classifier's ability to identify and classify Smichut constructions effectively.
