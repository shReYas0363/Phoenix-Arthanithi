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

@app.route('/education_projection', methods=['POST'])
def education_projection():
    data = request.get_json()

    
    years_of_study = data.get("years_of_study") 
    estimated_cost = data.get("estimated_cost")  
    monthly_savings = data.get("monthly_savings")  
    student_loan = data.get("student_loan", 0)  
    initial_investment = data.get("initial_investment") 
    risk_level = data.get("risk_level")  

    # Validations
    if years_of_study <= 0:
        return jsonify({"error": "Years of study must be greater than 0."}), 400
    if risk_level not in RISK_PROFILES:
        return jsonify({"error": "Risk level must be between 1 and 5."}), 400

    risk_profile, annual_return_rate = RISK_PROFILES[risk_level]

    total_savings = initial_investment + student_loan  
    yearwise_projection = []

    for year in range(1, years_of_study + 1):
        profit = total_savings * annual_return_rate 
        total_savings += profit + (monthly_savings * 12) 

        yearwise_projection.append({
            "year": year,
            "profit": round(profit, 2),
            "total_savings": round(total_savings, 2)
        })

    response = {
        "risk_profile": risk_profile,
        "annual_return_rate": annual_return_rate,
        "yearwise_projection": yearwise_projection
    }

    return jsonify(response)

@app.route('/wealth_growth', methods=['POST'])
def wealth_growth():
    data = request.get_json()

    
    current_net_worth = data.get("current_net_worth")
    target_wealth = data.get("target_wealth")
    years_to_goal = data.get("years_to_goal")  
    investment_type = data.get("investment_type")  # Stocks, Crypto, etc.
    monthly_savings = data.get("monthly_savings")  
    investment_approach = data.get("investment_approach")  # Active/Passive
    risk_level = data.get("risk_level")  # 1-5 scale

    # Validations
    if years_to_goal <= 0:
        return jsonify({"error": "Years to achieve wealth must be greater than 0."}), 400
    if risk_level not in RISK_PROFILES:
        return jsonify({"error": "Risk level must be between 1 and 5."}), 400

    risk_profile, annual_return_rate = RISK_PROFILES[risk_level]

    total_wealth = current_net_worth 
    yearwise_projection = []

    for year in range(1, years_to_goal + 1):
        profit = total_wealth * annual_return_rate  # Yearly profit
        total_wealth += profit + (monthly_savings * 12)  # Add yearly savings

        yearwise_projection.append({
            "year": year,
            "profit": round(profit, 2),
            "total_wealth": round(total_wealth, 2)
        })

    # Check if goal is achievable
    goal_status = "Achievable" if total_wealth >= target_wealth else "Shortfall"

    response = {
        "risk_profile": risk_profile,
        "annual_return_rate": annual_return_rate,
        "investment_type": investment_type,
        "investment_approach": investment_approach,
        "goal_status": goal_status,
        "final_wealth": round(total_wealth, 2),
        "yearwise_projection": yearwise_projection
    }

    return jsonify(response)

@app.route('/wedding_projection', methods=['POST'])
def wedding_projection():
    data = request.get_json()

    # Extracting input data
    current_age = data.get("current_age")
    marriage_age = data.get("marriage_age")
    estimated_budget = data.get("estimated_budget")
    initial_savings = data.get("initial_savings")
    loan_amount = data.get("loan_amount", 0)  # Optional
    monthly_savings = data.get("monthly_savings", 0)
    risk_level = data.get("risk_level")

    # Validations
    if current_age >= marriage_age:
        return jsonify({"error": "Marriage age must be greater than current age."}), 400
    if risk_level not in RISK_PROFILES:
        return jsonify({"error": "Risk level must be between 1 and 5."}), 400

    years_to_marriage = marriage_age - current_age
    risk_profile, annual_return_rate = RISK_PROFILES[risk_level]

    total_savings = initial_savings + loan_amount  # Include loan if applicable
    yearwise_projection = []

    for year in range(1, years_to_marriage + 1):
        profit = total_savings * annual_return_rate  # Calculate profit
        total_savings += profit + (monthly_savings * 12)  # Add yearly savings

        yearwise_projection.append({
            "year": year,
            "profit": round(profit, 2),
            "total_savings": round(total_savings, 2)
        })

    response = {
        "risk_profile": risk_profile,
        "annual_return_rate": annual_return_rate,
        "yearwise_projection": yearwise_projection,
        "total_savings_at_marriage": round(total_savings, 2),
        "estimated_budget": estimated_budget
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
