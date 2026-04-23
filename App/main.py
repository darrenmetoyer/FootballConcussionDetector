from flask import Flask, send_file, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from datetime import datetime
from FullPipelineDetector import runPipeline
from AIReportGenerator import buildReportParagraph, createReportDoc
import sqlite3
import json
import os

try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
    googleAuthAvailable = True
except ImportError:
    googleAuthAvailable = False

load_dotenv("APIKeys.env")

app = Flask(__name__)
CORS(app)

uploadFolder = "Uploads"
allowedExtensions = {"mp4", "avi", "mov", "mkv"}
databasePath = "app.db"
generatedReportsFolder = "GeneratedReports"

app.config["UPLOAD_FOLDER"] = uploadFolder
os.makedirs(uploadFolder, exist_ok=True)
os.makedirs(generatedReportsFolder, exist_ok=True)

googleClientId = os.getenv("GOOGLE_CLIENT_ID", "").strip()


def getDbConnection():
    connection = sqlite3.connect(databasePath)
    connection.row_factory = sqlite3.Row
    return connection


def initDb():
    print("[initDb] initializing database")
    connection = getDbConnection()
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            googleId TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            picture TEXT,
            createdAt TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            userId INTEGER NOT NULL,
            videoName TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            rawJson TEXT NOT NULL,
            reportParagraph TEXT NOT NULL,
            docId TEXT NOT NULL,
            FOREIGN KEY (userId) REFERENCES users(id)
        )
    """)

    connection.commit()
    connection.close()
    print("[initDb] database ready")


def allowedFile(fileName):
    return "." in fileName and fileName.rsplit(".", 1)[1].lower() in allowedExtensions


def getUserById(userId):
    connection = getDbConnection()
    user = connection.execute(
        "SELECT * FROM users WHERE id = ?",
        (userId,)
    ).fetchone()
    connection.close()
    return user


def getUserByGoogleId(googleId):
    connection = getDbConnection()
    user = connection.execute(
        "SELECT * FROM users WHERE googleId = ?",
        (googleId,)
    ).fetchone()
    connection.close()
    return user


def createUser(googleId, name, email, picture):
    print(f"[createUser] creating user {email}")
    connection = getDbConnection()
    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO users (googleId, name, email, picture, createdAt)
        VALUES (?, ?, ?, ?, ?)
    """, (
        googleId,
        name,
        email,
        picture,
        datetime.now().isoformat()
    ))

    connection.commit()
    userId = cursor.lastrowid
    connection.close()
    print(f"[createUser] created userId={userId}")
    return userId


def updateUser(userId, name, email, picture):
    print(f"[updateUser] updating userId={userId}")
    connection = getDbConnection()
    connection.execute("""
        UPDATE users
        SET name = ?, email = ?, picture = ?
        WHERE id = ?
    """, (
        name,
        email,
        picture,
        userId
    ))
    connection.commit()
    connection.close()


def findOrCreateUser(googleId, name, email, picture):
    existingUser = getUserByGoogleId(googleId)

    if existingUser:
        updateUser(existingUser["id"], name, email, picture)
        return existingUser["id"]

    return createUser(googleId, name, email, picture)


@app.route("/")
def index():
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Concussion Detection System</title>
  <script crossorigin src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
  <script crossorigin src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
  <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
  <script>
    window.GOOGLE_CLIENT_ID = {json.dumps(googleClientId)};
  </script>
</head>
<body>
  <div id="root"></div>
  <script type="text/babel" src="/app.jsx"></script>
</body>
</html>"""
    return Response(html, mimetype="text/html")


@app.route("/app.jsx")
def serveJsx():
    return send_file("ConcussionDetectorUI.jsx", mimetype="text/babel")


@app.route("/api/auth/google", methods=["POST"])
def googleAuth():
    print("[googleAuth] request received")

    if not request.is_json:
        print("[googleAuth] invalid request: not json")
        return jsonify({"success": False, "error": "Expected JSON body"}), 400

    requestData = request.get_json(silent=True) or {}
    token = requestData.get("token")

    if not token:
        print("[googleAuth] missing token")
        return jsonify({"success": False, "error": "Missing token"}), 400

    if not googleClientId:
        print("[googleAuth] missing GOOGLE_CLIENT_ID on server")
        return jsonify({"success": False, "error": "Server Google client ID is not configured"}), 500

    if not googleAuthAvailable:
        print("[googleAuth] google-auth not installed")
        return jsonify({"success": False, "error": "google-auth is not installed"}), 500

    try:
        print("[googleAuth] verifying google token")
        idInfo = id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            googleClientId
        )

        googleId = idInfo.get("sub")
        name = idInfo.get("name", "User")
        email = idInfo.get("email", "")
        picture = idInfo.get("picture", "")

        userId = findOrCreateUser(googleId, name, email, picture)

        print(f"[googleAuth] success userId={userId} email={email}")
        return jsonify({
            "success": True,
            "user": {
                "id": userId,
                "googleId": googleId,
                "name": name,
                "email": email,
                "picture": picture
            }
        }), 200

    except ValueError as error:
        print(f"[googleAuth] token verification failed: {error}")
        return jsonify({
            "success": False,
            "error": f"Invalid Google token: {error}"
        }), 401


@app.route("/api/analyze", methods=["POST"])
def analyzeVideo():
    print("[analyzeVideo] request received")
    userId = request.form.get("userId")

    if not userId:
        print("[analyzeVideo] missing userId")
        return jsonify({"success": False, "error": "Missing userId"}), 400

    user = getUserById(userId)
    if not user:
        print(f"[analyzeVideo] invalid userId={userId}")
        return jsonify({"success": False, "error": "Invalid user"}), 401

    if "video" not in request.files:
        print("[analyzeVideo] no video in request.files")
        return jsonify({"success": False, "error": "No video file uploaded"}), 400

    video = request.files["video"]

    if not video or video.filename == "":
        print("[analyzeVideo] empty filename")
        return jsonify({"success": False, "error": "No selected file"}), 400

    if not allowedFile(video.filename):
        print(f"[analyzeVideo] invalid file type: {video.filename}")
        return jsonify({"success": False, "error": "Invalid file type"}), 400

    fileName = secure_filename(video.filename)
    filePath = os.path.join(app.config["UPLOAD_FOLDER"], fileName)

    try:
        print(f"[analyzeVideo] saving upload to {filePath}")
        video.save(filePath)

        print("[analyzeVideo] running detector pipeline")
        detectorResult = runPipeline(filePath)
        print(f"[analyzeVideo] detector result: {json.dumps(detectorResult, indent=2)}")

        if not detectorResult or not isinstance(detectorResult, dict):
            print("[analyzeVideo] detector returned invalid result")
            return jsonify({
                "success": False,
                "error": "Detector returned no valid result."
            }), 500

        status = detectorResult.get("status", "Unknown")
        rawJson = json.dumps(detectorResult)
        reportDate = datetime.now().strftime("%Y-%m-%d")

        print("[analyzeVideo] generating ai paragraph")
        reportParagraph = buildReportParagraph(detectorResult)

        print("[analyzeVideo] creating docx report")
        docId, _ = createReportDoc(fileName, reportParagraph)

        print("[analyzeVideo] saving report to database")
        connection = getDbConnection()
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO reports (userId, videoName, date, status, rawJson, reportParagraph, docId)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            int(userId),
            fileName,
            reportDate,
            status,
            rawJson,
            reportParagraph,
            docId
        ))

        reportId = cursor.lastrowid
        connection.commit()
        connection.close()

        print(f"[analyzeVideo] done reportId={reportId} docId={docId}")
        return jsonify({
            "success": True,
            "report": {
                "id": reportId,
                "userId": int(userId),
                "videoName": fileName,
                "date": reportDate,
                "status": status,
                "downloadUrl": f"/api/reports/{docId}/download"
            }
        }), 200

    except Exception as error:
        print(f"[analyzeVideo] failed: {error}")
        return jsonify({
            "success": False,
            "error": f"Failed to save or analyze video: {error}"
        }), 500


@app.route("/api/reports", methods=["GET"])
def getReports():
    print("[getReports] request received")
    userId = request.args.get("userId")

    if not userId:
        print("[getReports] missing userId")
        return jsonify({"success": False, "error": "Missing userId"}), 400

    user = getUserById(userId)
    if not user:
        print(f"[getReports] invalid userId={userId}")
        return jsonify({"success": False, "error": "Invalid user"}), 401

    connection = getDbConnection()
    rows = connection.execute("""
        SELECT id, videoName, date, docId
        FROM reports
        WHERE userId = ?
        ORDER BY id DESC
    """, (userId,)).fetchall()
    connection.close()

    reports = []
    for row in rows:
        rowDict = dict(row)
        reports.append({
            "id": rowDict["id"],
            "videoName": rowDict["videoName"],
            "date": rowDict["date"],
            "downloadUrl": f"/api/reports/{rowDict['docId']}/download"
        })

    print(f"[getReports] returning {len(reports)} reports")
    return jsonify(reports), 200


@app.route("/api/reports/<docId>/download", methods=["GET"])
def downloadReport(docId):
    filePath = os.path.join(generatedReportsFolder, f"{docId}.docx")
    print(f"[downloadReport] docId={docId}")

    if not os.path.exists(filePath):
        print("[downloadReport] document not found")
        return jsonify({"success": False, "error": "Document not found"}), 404

    print("[downloadReport] sending document")
    return send_file(
        filePath,
        as_attachment=True,
        download_name="concussion_report.docx"
    )


if __name__ == "__main__":
    initDb()
    print("[main] starting flask app on port 5000")
    app.run(debug=True, port=5000)