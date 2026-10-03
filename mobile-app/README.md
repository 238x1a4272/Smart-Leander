# Smart Leander Mobile App

This is the React Native/Expo mobile frontend for the existing Flask loan-prediction application in this repository.

## Run

cd mobile-app
npm install
npx expo start

Expo supports Android, iOS and web from one project.

## Connect to the existing Flask model

The mobile app sends the same 11 fields used by the existing /submit endpoint:

Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area.

Before running the app, edit API_BASE_URL in App.js:

const API_BASE_URL = "http://YOUR-SERVER-IP:5000";

The Flask server must be reachable from the phone. Do not use localhost when testing from a physical phone; use the computer's LAN IP or a deployed HTTPS API.

The current Flask backend loads decision_tree_model.pkl and returns either Loan Approved or Loan Not Approved. The mobile app displays that result as a native mobile UI.

## Production next steps

- Deploy Flask API over HTTPS.
- Add authentication if needed.
- Add server-side input validation.
- Return JSON from a dedicated /api/predict endpoint rather than parsing HTML.
- Build Android/iOS binaries with EAS after testing.
