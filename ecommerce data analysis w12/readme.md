## Ecommerce Data Analysis 

We analyzed the e-commerce dataset by first identifying and filling missing values. Using this cleaned data, we built two predictive models: a classification model to determine if a customer will return for future purchases and a regression model to estimate their expected spending. 

**Missimg valuses**

In the sample, missing values appear in:

- time_on_site_sec
- hour
- discount_rate
- payment_risk_score
- returned, refund_amount, delivery_days, delivery_delay (mostly empty)

Handiling

time_on_site_sec: Fill with median per device/traffic_source

hour: Fill with mode or -1 if truly unknown

discount_rate: Fill with 0 if discount_used=0, else median for category

returned / refund_amount: Fill with 0 if purchase_amount = 0, else treat as 0 if missing

delivery_days / delay: Possibly irrelevent for non-purchases; fill with median for country/product

**Goals** 

a) Will customer buy again? -> Binary classification (returning_user)

b) How much will the client spend? -> Regression (purchase_amount)

**Feature Engineering**

For predicting repurchase: returning_user, session_quality_score, page_views, time_on_site_sec, customer_support_contact, out_of_stock_view, payment_risk_score.

For predicting spend: product_category, discount_rate, discount_used, user_segment, traffic_source, country, device.

**Modeling Approach**

Models:
 - Classifier (Logistic Regression / XGBoost) for will_buy_again
 - Regressor (Linear Regression / XGBoost Regressor) for spend_amount

Metrics:
  - Classification: AUC-ROC, Precision-Recall
  - Regression: RMSE, MAE

