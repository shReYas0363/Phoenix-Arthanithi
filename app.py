from flask import Flask, request, jsonify

app = Flask(__name__)

# Risk profile mapping
RISK_PROFILES = {
    1: ("Conservative", 0.04),  # 4% return
    2: ("Conservative", 0.06),  # 6% return
    3: ("Moderate", 0.08),      # 8% return
    4: ("Aggressive", 0.10),    # 10% return
    5: ("Aggressive", 0.12)     # 12% return
}

@app.route('/financial_projection', methods=['POST'])
def financial_projection():
    data = request.get_json()

    # Extracting input data
    current_age = data.get("current_age")
    retirement_age = data.get("retirement_age")
    estimated_cost = data.get("estimated_cost")
    dependents = data.get("dependents")
    initial_investment = data.get("initial_investment")
    monthly_investment = data.get("monthly_investment")
    risk_level = data.get("risk_level")

    # Validations
    if current_age >= retirement_age:
        return jsonify({"error": "Retirement age must be greater than current age."}), 400
    if risk_level not in RISK_PROFILES:
        return jsonify({"error": "Risk level must be between 1 and 5."}), 400

    years_to_retire = retirement_age - current_age
    risk_profile, annual_return_rate = RISK_PROFILES[risk_level]

    total_savings = initial_investment
    yearwise_profit = []

    for year in range(1, years_to_retire + 1):
        profit = total_savings * annual_return_rate  # Calculate profit
        total_savings += profit + (monthly_investment * 12)  # Add yearly contributions

        yearwise_profit.append({
            "year": year,
            "profit": round(profit, 2),
            "total_amount": round(total_savings, 2)
        })

    response = {
        "risk_profile": risk_profile,
        "annual_return_rate": annual_return_rate,
        "yearwise_profit": yearwise_profit
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
