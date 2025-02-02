import random
from flask import Flask, request, jsonify
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint for service health verification"""
    return jsonify({"status": "healthy"})

@app.route("/chat", methods=["POST"])
def chat_completion():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        N_Yes = data.get("N_Yes")
        portfolio = { "technology": ["NVIDIA Corp", "Palantir Technologies", "Microstrategy Inc", "HCL Technologies", "Infosys Limited"], "healthcare": ["Fortis Healthcare", "Apollo Hospitals Enterprise", 
                  "Global Health Limited", "KIMS"], "education": ["Veranda Learning Solutions", "Shanti Educational Initiatives", "Career Point", "Global Education Ltd"], "finance": ["CRISIL Ltd", "Angel One Ltd", 
               "Aditya Birla Sun Life AMC", "Computer Age Management Services"],
        "real estate": ["DLF Limited", "Godrej Properties", "Brigade Enterprises",
                    "Oberoi Realty", "Prestige Estates Projects"],
        "manufacturing": ["Tata Steel", "Hindalco Industries", "Bajaj Auto",
                        "Sun Pharma", "Reliance Industries"],
        "entertainment": ["Warner Bros. Discovery", "Netflix", "PVR INOX",
                        "Zee Entertainment", "Live Nation"]
        }
        
        if N_Yes is None or not isinstance(N_Yes, str):
            return jsonify({"detail": "Invalid input, N_Yes must be an sector(string)"}), 400
        domain=N_Yes.lower()
        response={f"User_{i+1}": random.sample(portfolio[domain], 2) for i in range(3) } #Dictionary {'User1':list, 'User2': list ...}
       
        return jsonify({"response": response})
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({"detail": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)