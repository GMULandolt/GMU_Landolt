import datetime

from flask import Flask, render_template, jsonify
from observatories import observatories

app = Flask(__name__)



@app.route("/ws/getObservatories", methods=['POST'])
def getObservatories():
    obs = {
        "observatories": observatories
    }
    return jsonify(obs)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)

    