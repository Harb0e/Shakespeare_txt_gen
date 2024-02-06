from flask import Flask,jsonify,request
from flask_swagger_ui import get_swaggerui_blueprint
from lib import gentext

app = Flask(__name__)
SWAGGER_URL="/swagger"
API_URL="/static/swagger.json"

swagger_ui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': 'Access API'
    }
)
app.register_blueprint(swagger_ui_blueprint, url_prefix=SWAGGER_URL)

@app.route("/")
def home():
    return jsonify({
        "Message": "app up and running successfully"
    })

@app.route("/access",methods=["POST"])
def access():
    data = request.get_json()
    print(data)
    name = data.get("text", "this is a test story")
    numwords = data.get("words",1)

    message = gentext.get(name,numwords)

    return jsonify({
        "Message": message
    })


if __name__=="__main__":
    app.run(host="0.0.0.0",port=8080)