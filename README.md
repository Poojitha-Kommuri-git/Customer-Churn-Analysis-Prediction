# Customer Churn Analysis & Prediction 📊

Hey there! 👋 

This project helps businesses understand why customers are leaving and predict who's likely to churn next. I built it to make churn analysis accessible and actionable for anyone working with customer data.

## What's This About?

Customer churn is expensive. Really expensive. This tool analyzes your customer data to:

- Figure out **why** customers are leaving
- **Predict** which customers might churn soon
- Give you **actionable recommendations** to keep them around

Think of it as your early warning system for customer retention.

## What Can It Do?

### 1. Upload & Explore
Drop in your customer CSV file and instantly see:
- How many customers you've lost
- Your overall churn rate
- Key patterns in the data

### 2. Deep Dive Analysis
See exactly where the problems are:
- Which customer groups churn the most?
- Does contract type matter? (Spoiler: it really does)
- Are newer customers leaving faster?

### 3. Machine Learning Magic
The app trains a model that can:
- Predict churn probability for any customer
- Show you which factors matter most
- Give you ~80% accuracy on predictions

### 4. Get Real Insights
Not just numbers - actual business recommendations:
- What actions to take right now
- Expected ROI from reducing churn
- Priority customers to focus on

## How to Use It

### Quick Start (5 minutes)
```bash
# Install what you need
pip install streamlit plotly pandas scikit-learn

# Run the app
streamlit run app.py

# Open your browser to http://localhost:8501
```

That's it! You're ready to go.

### Step by Step

1. **Get the Data**: Download the Telco Customer Churn dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

2. **Upload It**: Use the app's upload page to load your CSV

3. **Explore**: Click through the tabs to see different analyses

4. **Train Model**: Hit the "Train Model" button on the ML page

5. **Predict**: Enter customer details to see their churn risk

## What You'll Learn

From my analysis of the Telco dataset, here's what really matters:

- **First year is critical**: New customers churn 3-4x more than established ones
- **Contracts work**: Month-to-month customers leave way more often
- **Price matters**: Customers who churn pay about $15-20 more per month
- **Payment method counts**: Electronic check users churn more (who knew?)

## Tech Stuff (If You Care)

**Built with:**
- Streamlit for the web interface (because life's too short for HTML)
- Scikit-learn for machine learning (Logistic Regression, simple but effective)
- Plotly for pretty charts
- Pandas for data wrangling

**Why these choices?**
- Wanted it fast to build and easy to modify
- No complex frontend framework needed
- Can deploy it for free on Streamlit Cloud
- Everything in Python - no context switching

## Project Structure
```
churn-app/
├── app.py              # Main application (all the magic happens here)
├── data/               # Put your CSV files here
├── requirements.txt    # Dependencies
└── README.md          # You're reading it!
```

## Making It Your Own

Want to use this with your own data? You'll need:

- A CSV with customer information
- A column indicating if they churned (Yes/No or 1/0)
- Some features like tenure, contract type, charges, etc.

You might need to tweak the feature engineering in the code, but the structure works for most subscription businesses.

## Known Issues & Limitations

Being honest here:
- Works best with datasets similar to Telco (subscription businesses)
- Logistic Regression is simple - could use more sophisticated models
- Doesn't handle real-time data updates (you need to re-upload)
- All data is processed in memory (so huge datasets might struggle)

## What's Next?

Ideas I'm thinking about:
- [ ] Add more ML models (Random Forest, XGBoost)
- [ ] Support for time-series predictions
- [ ] Email alerts for high-risk customers
- [ ] Database integration instead of CSV uploads
- [ ] A/B testing framework for retention campaigns

## Contributing

Found a bug? Have an idea? Open an issue or send a PR. I'm pretty responsive.

## License

MIT - do whatever you want with it. If it helps your business, that's awesome. If you make it better, even more awesome.

## Questions?

Hit me up if you get stuck or want to chat about customer retention strategies. I'm always down to talk about churn analysis.

---

**Built with ☕ and a genuine desire to help businesses keep their customers happy**

P.S. - If this saves you money, consider buying me a coffee or starring the repo. Both equally appreciated! ⭐