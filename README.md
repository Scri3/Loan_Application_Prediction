## Decision Tree Algorithm

Implementation of **Decision Tree classifier** which
predicts whether a loan application (from a given dataset) is **Approved** or **Denied** based
on two numerical features:
- **age**
- **credit_score**

<img width="4643" height="3506" alt="decision_tree" src="https://github.com/user-attachments/assets/843c18d1-3e4b-4a31-a7ee-f0385d136d96" />


The script reads loan applications data from `'data.csv'`, converts
categorical labels into numerical values, trains a decision tree using
**scikit-learn**, and exports the resulting tree visualization as
`'decision_tree.png'`.

------------------------------------------------------------------------

## How the Decision Tree Algorithm Works

A **Decision Tree classifier** works by splitting data into branches
based on feature values to make predictions. At each step, the algorithm
chooses the most informative feature and threshold to divide the
dataset.

For this dataset:

-   The features are **age** and **credit_score**.
-   The target variable is **loan_status**, mapped to:
    -   `Approved` = **1**
    -   `Denied` = **0**

The decision tree attempts to separate approved vs. denied applications
by creating threshold rules such as:

-   `credit_score >= 665.0`
-   `age <= 40.5`

These splits continue until the tree reaches a point where it can
confidently assign a prediction. The final model can then visually show
how loans are classified.

------------------------------------------------------------------------

## CSV File Format (`data.csv`)

The sample dataset looks like this:

    age,credit_score,loan_status
    20,720,Approved
    18,680,Approved
    55,590,Denied
  
Replace **your own dataset** in `'DataFrame = pandas.read_csv("data.csv")'` to generate a Decision Tree classifier based on your data.


  -----------------------------------------------------------------------

## Output File Generated

-   The output is a **decision_tree.png** which is a visualization of the trained decision
    tree.
