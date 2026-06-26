# Smart Lender – Loan Eligibility Prediction System
## Entity Relationship Diagram (ERD) Documentation

This document outlines the database design for the Smart Lender – Loan Eligibility Prediction System. It includes definitions for all core entities, their attributes, keys, and the relationships that bind them together.

---

### ER Diagram

Below is the visual Entity-Relationship Diagram representing the schema. It has been programmatically rendered to match all required fields, primary keys, and foreign keys.

![Entity Relationship Diagram](file:///C:/Users/ADMIN/.gemini/antigravity/brain/5e7d0c00-a396-4555-9d64-acbbf2d6075a/ER_Diagram.png)

---

### Main Entities

#### 1. [USER](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/smart-lender-erd/README.md#1-user)
Stores credentials and role details of the platform's users (applicants and credit officers).
*   `user_id` (PK, INT): Unique identifier.
*   `name` (VARCHAR): User's full name.
*   `email` (VARCHAR): Contact/login email address.
*   `role` (VARCHAR): Account role (e.g., Applicant, Credit Officer).
*   `created_at` (TIMESTAMP): Account registration timestamp.

#### 2. [APPLICANT_PROFILE](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/smart-lender-erd/README.md#2-applicant_profile)
Stores the demographic and socioeconomic data points of the applicant, which are used as primary features in the Machine Learning model.
*   `applicant_id` (PK, INT): Unique identifier.
*   `user_id` (FK, INT): Links profile to a registered user.
*   `gender` (VARCHAR): Applicant's gender.
*   `married` (VARCHAR): Marital status.
*   `education` (VARCHAR): Educational background (Graduate / Under Graduate).
*   `self_employed` (VARCHAR): Professional status.
*   `dependents` (VARCHAR): Number of dependents.
*   `property_area` (VARCHAR): Location type (Urban / Semiurban / Rural).

#### 3. [CREDIT_HISTORY](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/smart-lender-erd/README.md#3-credit_history)
Stores the historical credit behavior of the applicant.
*   `credit_id` (PK, INT): Unique identifier.
*   `applicant_id` (FK, INT): Links to the corresponding applicant profile.
*   `credit_score` (INT): Numerical score representing creditworthiness.
*   `credit_history_status` (VARCHAR): Code representing credit guidelines met (e.g., 0 or 1).

#### 4. [LOAN_APPLICATION](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/smart-lender-erd/README.md#4-loan_application)
Captures individual loan requests submitted by applicants.
*   `loan_id` (PK, INT): Unique identifier.
*   `applicant_id` (FK, INT): Links the request to the applicant profile.
*   `income` (DECIMAL): Applicant monthly income.
*   `coapplicant_income` (DECIMAL): Co-applicant monthly income.
*   `loan_amount` (DECIMAL): Amount requested.
*   `loan_term` (INT): Term length (in months or days).
*   `application_date` (TIMESTAMP): Submission timestamp.

#### 5. [MODEL](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/smart-lender-erd/README.md#5-model)
Details of the trained Machine Learning models used for evaluations.
*   `model_id` (PK, INT): Unique identifier.
*   `model_name` (VARCHAR): Model title.
*   `algorithm` (VARCHAR): Classifier algorithm name (e.g., Random Forest, XGBoost).
*   `training_accuracy` (FLOAT): Score on training data.
*   `testing_accuracy` (FLOAT): Score on test partition.
*   `file_path` (VARCHAR): File location of the serialized model binary.

#### 6. [PREDICTION_RESULT](file:///C:/Users/ADMIN/.gemini/antigravity/scratch/smart-lender-erd/README.md#6-prediction_result)
Stores model inferences generated for each loan application.
*   `prediction_id` (PK, INT): Unique identifier.
*   `loan_id` (FK, INT): Links to the loan application evaluated.
*   `model_id` (FK, INT): Links to the specific model version used.
*   `prediction_status` (VARCHAR): Outcome prediction (Approved / Rejected).
*   `probability_score` (FLOAT): Confidence probability.
*   `prediction_time` (TIMESTAMP): Date and time of inference.

---

### Relationship Explanations

1.  **User ───< Applicant Profile (1-to-Many)**: A user account can setup/manage multiple applicant profiles over time (e.g., for co-signers or profile updates).
2.  **Applicant Profile ─── Credit History (1-to-1)**: Each applicant profile is associated with a single consolidated credit history check.
3.  **Applicant Profile ───< Loan Application (1-to-Many)**: An applicant can file multiple loan applications over the platform's lifecycle.
4.  **Loan Application ─── Prediction Result (1-to-1)**: A loan application is evaluated by the ML system, yielding a single final prediction record.
5.  **Model ───< Prediction Result (1-to-Many)**: A single trained model version can evaluate and create predictions for many different loan applications.

---

### Advantages of this Schema

*   **Normalization**: Separation of profiles, credit checks, and transactions reduces redundancy.
*   **Predictive Traceability**: Retaining the model version in `PREDICTION_RESULT` allows full auditing of which algorithm made which prediction.
*   **Feature Completeness**: All attributes map perfectly to standard machine learning datasets like the Kaggle Loan Prediction Dataset.
