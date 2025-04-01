from flask import Flask, request, jsonify
import os
import requests
import uuid
from flask_cors import CORS
import json
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# OpenRouter API for Mistral
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# In-memory storage for negotiations
# Structure: {negotiation_id: {
#    "messages": [{"speaker": "user1", "message": "..."}, ...],
#    "user1_count": 0,
#    "user2_count": 0,
#    "total_count": 0
# }}
negotiations = {}

@app.route('/negotiate', methods=['POST'])
def negotiate():
    try:
        data = request.json
        
        # Validate incoming request
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Invalid request format"}), 400
        
        # Extract required fields
        negotiation_id = data.get('negotiation_id')
        speaker = data.get('speaker')
        message = data.get('message')
        
        # Validate fields
        if not speaker or not message:
            return jsonify({"error": "Missing required fields: 'speaker' and 'message' are required"}), 400
            
        if speaker not in ['user1', 'user2']:
            return jsonify({"error": "Speaker must be either 'user1' or 'user2'"}), 400
        
        # Create new negotiation if needed
        if not negotiation_id:
            negotiation_id = str(uuid.uuid4())
            negotiations[negotiation_id] = {
                "messages": [],
                "user1_count": 0,
                "user2_count": 0,
                "total_count": 0
            }
        elif negotiation_id not in negotiations:
            return jsonify({"error": f"Negotiation with ID {negotiation_id} not found"}), 404
        
        # Check if user has exceeded their 5-message limit
        if speaker == "user1" and negotiations[negotiation_id]["user1_count"] >= 5:
            return jsonify({"error": "User1 has already sent 5 messages"}), 400
        if speaker == "user2" and negotiations[negotiation_id]["user2_count"] >= 5:
            return jsonify({"error": "User2 has already sent 5 messages"}), 400
        
        # Add message to negotiation
        negotiations[negotiation_id]["messages"].append({
            "speaker": speaker,
            "message": message
        })
        
        # Update message counts
        if speaker == "user1":
            negotiations[negotiation_id]["user1_count"] += 1
        else:
            negotiations[negotiation_id]["user2_count"] += 1
            
        negotiations[negotiation_id]["total_count"] += 1
        
        # Check if we've reached 10 messages total (5 from each user)
        if negotiations[negotiation_id]["total_count"] == 10:
            # Generate verdict using Mistral model
            verdict = generate_verdict(negotiations[negotiation_id]["messages"])
            
            # Clean up the negotiation data (optional)
            # del negotiations[negotiation_id]
            
            return jsonify({
                "negotiation_id": negotiation_id,
                "status": "completed",
                "verdict": verdict
            })
        else:
            # Return current status
            return jsonify({
                "negotiation_id": negotiation_id,
                "status": "in_progress",
                "messages_sent": negotiations[negotiation_id]["total_count"],
                "user1_messages": negotiations[negotiation_id]["user1_count"],
                "user2_messages": negotiations[negotiation_id]["user2_count"],
                "messages_remaining": 10 - negotiations[negotiation_id]["total_count"]
            })
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_verdict(messages):
    """
    Generate a verdict using Mistral via OpenRouter API based on the negotiation messages.
    
    Args:
        messages (list): List of dictionaries containing speaker and message content
    
    Returns:
        dict: Contains summary and compromise suggestion.
    """
    try:
        # Format the conversation for the AI
        conversation_text = "\n".join([f"{msg['speaker']}: {msg['message']}" for msg in messages])
        
        # Create system and user messages for the AI
        system_message = """
        You are a fair and impartial legal mediator. Analyze the following negotiation between two parties 
        and provide a balanced verdict that represents a fair compromise. Your response must be in valid JSON 
        format with two fields:
        1. 'summary': A brief summary of the negotiation and the key points of contention
        2. 'compromise': A detailed middle-ground solution that addresses the concerns of both parties, just a paragraph and nothing else, no other objects
        
        Respond only with the JSON object, no additional text or explanation.
        """
        
        user_message = f"Here is the negotiation conversation:\n\n{conversation_text}"
        
        # Prepare the request payload
        payload = {
            "model": "mistralai/mistral-7b-instruct:free",
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ], 
            "temperature": 0.3
        }
        
        # Make the API request
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            response_data = response.json()
            response_content = response_data["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                # Clean up the response to extract just the JSON part
                json_text = extract_json(response_content)
                verdict = json.loads(json_text) if isinstance(json_text, str) else json_text
                
                # Ensure response has required fields
                if "summary" not in verdict or "compromise" not in verdict:
                    verdict = {
                        "summary": "Failed to generate proper verdict format.",
                        "compromise": "Please try again with clearer negotiation points."
                    }
                
                # Adjust the verdict format to the desired structure
                return {
                    "compromise": verdict.get("compromise", "Compromise here"),
                    "summary": verdict.get("summary", "Summary here")
                }
                
            except json.JSONDecodeError:
                # Handle case where AI doesn't return valid JSON
                return {
                    "compromise": "The negotiation was processed, but a structured verdict could not be generated. Please try again.",
                    "summary": "Error parsing AI response."
                }
        else:
            return {
                "summary": "API request failed.",
                "compromise": f"Status code: {response.status_code}. Error: {response.text}"
            }
            
    except Exception as e:
        print(f"Error generating verdict: {str(e)}")
        return {
            "compromise": f"An error occurred: {str(e)}",
            "summary": "Error generating verdict."
        }

def extract_json(text):
    """Extract JSON content from the LLM response"""
    try:
        import re
        import ast
        
        # Remove markdown code block indicators if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        
        # Try to find JSON content between curly braces
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            cleaned = match.group().replace('\n', ' ').replace('\r', '')
            return json.loads(cleaned)
        
        # As a fallback, try literal evaluation
        return ast.literal_eval(text)
    except Exception as e:
        print("JSON parsing error:", e)
        return {
            "summary": "Error parsing response.",
            "compromise": "Could not extract valid JSON from the model response."
        }

@app.route('/negotiation/<negotiation_id>', methods=['GET'])
def get_negotiation(negotiation_id):
    """
    Get the current status of a negotiation by ID, including the verdict if completed.
    """
    if negotiation_id not in negotiations:
        return jsonify({"error": "Negotiation not found"}), 404

    negotiation = negotiations[negotiation_id]
    
    response_data = {
        "negotiation_id": negotiation_id,
        "status": "in_progress" if negotiation["total_count"] < 10 else "completed",
        "messages_sent": negotiation["total_count"],
        "user1_messages": negotiation["user1_count"],
        "user2_messages": negotiation["user2_count"],
        "messages_remaining": max(0, 10 - negotiation["total_count"]),
        "messages": negotiation["messages"]
    }

    # If negotiation is completed, generate and include the verdict
    if negotiation["total_count"] == 10:
        verdict = generate_verdict(negotiation["messages"])
        response_data["verdict"] = verdict

    return jsonify(response_data)

if __name__ == '__main__':
    # Check if OpenRouter API key is set
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Warning: OPENROUTER_API_KEY environment variable is not set!")
        print("Set it before running the application: export OPENROUTER_API_KEY='your-api-key'")
    
    port = int(os.environ.get("PORT", 5500))
    app.run(host='0.0.0.0', port=port, debug=True)
