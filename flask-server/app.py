import os
import requests
from flask import Flask, session, abort, redirect, request
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
import pathlib
from pip._vendor import cachecontrol
import google.auth.transport.requests
from FinalCode import recoms
from FinalCode import genrecom


app = Flask(__name__,static_url_path='', static_folder='frontend/build')
app.secret_key="Priyal"

os.environ["OAUTHLIB_INSECURE_TRANSPORT"]="1" #by default oauth2 only works with https, but for testing purpose we can just bypass this by setting the environment variable 

GOOGLE_CLIENT_ID="45973301105-vd6fcicbfmupnbkpm7dk7ev81tf2csba.apps.googleusercontent.com"
client_secrets_file = os.path.join(pathlib.Path(__file__).parent, "client_secret.json") #first part is just the folder of the file we r in=> basically joining the 2 paths to create the complete path to the "client_secret.json" file


flow=Flow.from_client_secrets_file(
    client_secrets_file=client_secrets_file,
    scopes=["https://www.googleapis.com/auth/userinfo.profile", "https://www.googleapis.com/auth/userinfo.email", "openid"],
    redirect_uri="http://127.0.0.1:5000/callback"
    )
# this is is decorator=> we can pass functions in this decorator to decorator and then it protects them from unauthorised users
def login_is_required(function):
    def wrapper(*args, **kwargs):
        if "google_id" not in session: #we are not using session package of flask because it uses cookies n stores it ok clint side=> should not do that in produciton projects
            return abort(401) #Authorization required
        else:
            return function()

    return wrapper

@app.route("/login")
def login():
    authorization_url, state= flow.authorization_url()
    session["state"]= state
    return redirect(authorization_url)
    # session["google_id"]="Test" #filling up the google id with some random value to test and the redirecting to the protected area
    # return redirect("/protected_area")

# to receive the data from google endpoint
@app.route("/callback")
def callback():
    flow.fetch_token(authorization_response=request.url) #fetches token in exchange of the info we just recieved

    if not session["state"] == request.args["state"]: #if state we received n the state we saved in last session r same=> protects the site from cross site attacks
        abort(500) #state doesnt match

    credentials = flow.credentials
    request_session=requests.session()
    cached_session = cachecontrol.CacheControl(request_session)
    token_request = google.auth.transport.requests.Request(session=cached_session)

    id_info = id_token.verify_oauth2_token( #to verify the info we just received
        id_token = credentials._id_token,
        request=token_request,
        audience=GOOGLE_CLIENT_ID
    )
    # return id_info
    session["google_id"]=id_info.get("sub") #appends google id and name to the session
    session["name"] = id_info.get("name")
    return redirect("/protected_area") #redirect the user to the protected area

@app.route("/logout")
def logout():
    session.clear() #clears the session n redirects to the home pg
    return redirect("/")

# 2 simple pages, index (log in or sign up pg) and protected area (rest of the app)
@app.route("/")
def index():
    return "home <a href='/login'><button>Login</button></a>"

@app.route("/protected_area")
@login_is_required
def protected_area():
    return "hello world <a href='/logout'><button>Logout</button></a>"


@app.route("/members")
def members():
    return {"members":["Member1", "Member2","Member3"]}

# @app.route("/get_recom", methods=["POST"])
# def getRecom():
#     # user_id = request.form.get("user_id")
#     #user id is hardcoded to any random number. the random number should exist as a userid in the database
#     data = recoms(5)
#     return {data}


@app.route("/get_recom")
def getRecom():
    #user id is hardcoded to any random number. the random number should exist as a userid in the database
    recomLists = recoms(5)
    # recomLists = {'frndRecom': [13075, 598023, 1962468, 10411, 3398649, 1449501, 1574629, 1586130], 'topRatedRecom': [132708, 34087, 22535, 21228, 729536, 833946, 22292, 350860], 'topVisitedRecom': [22535, 184256, 9410, 11365, 906408, 164031, 34041, 96921], 'Frnds': [0, 2, 40, 41, 42, 44]}
    return recomLists

@app.route("/get_genRecom")
def getGenrecom():
    #user id is hardcoded to any random number. the random number should exist as a userid in the database
    recomLists = genrecom(5)
    # recomLists = {'genRecom': [1251458.0, 717104.0, 10411.0, 671812.0, 719023.0, 420315.0, 61648.0, 10259.0], }
    return recomLists


if __name__ == "__main__":
    app.run(debug=True)

