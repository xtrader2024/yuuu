# webhook_server.py

from flask import Flask, request, jsonify

app = Flask(__name__)

alerts = []

@app.route('/webhook', methods=['POST'])
def receive_alert():
    alert_data = request.get_json()
    alerts.append(alert_data)
    return jsonify({'message': 'Bildirim başarıyla alındı'})

@app.route('/alerts', methods=['GET'])
def get_alerts():
    return jsonify(alerts)

if __name__ == '__main__':
    app.run(debug=True)
