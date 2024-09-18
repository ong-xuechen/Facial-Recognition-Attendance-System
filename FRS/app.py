from flask import Flask, render_template, request, Response, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.local import Local
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import timedelta, datetime
import cv2
import datetime
import sqlite3
import threading
import numpy as np
import pytz
import base64
from io import BytesIO
from PIL import Image
from flask_socketio import SocketIO, emit
from collections import defaultdict
import re
from dateutil import parser
from pytz import timezone

# Application
app = Flask(__name__)
app.secret_key = '3d6f45a5fc12445dbac2f59c3b6c7cb1'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=60)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# Login Manager
login_manager = LoginManager(app)
login_manager.session_protection = 'strong'
login_manager.login_view = 'login'
login_manager.init_app(app)
login_manager.remember_cookie_duration = timedelta(days=7)

# Create thread-local SQLite connection and cursor
local_storage = Local()


# Class for LoginManager
class User:
    def __init__(self, user_id, role='user'):
        self.id = user_id
        self.role = role
        self.username = None
        self.first_login = True
        self.last_login = None
        self.previous_login = None

    def get_id(self):
        return self.id

    def is_active(self):
        return True

    def is_authenticated(self):
        return True

    def is_admin(self):
        return self.role == 'admin'

    # last login datetime
    def set_last_login(self, last_login):
        if last_login:
            self.previous_login = self.last_login.strftime('%A, %d %B %Y, %I:%M:%S %p') if self.last_login else None
            self.last_login = last_login.strftime('%A, %d %B %Y, %I:%M:%S %p')

    def get_last_login(self):
        if self.last_login:
            return self.last_login
        return None

    def get_previous_login(self):
        if self.previous_login:
            return self.previous_login
        return None

    def update_last_login(self):
        # update last login datetime in user database
        c = get_cursor()
        c.execute("UPDATE users SET last_login=? WHERE id=?", (self.last_login, self.id))
        c.connection.commit()

@login_manager.user_loader
def load_user(user_id):
    # Retrieve the user from the database based on the user_id
    c = get_cursor()
    c.execute("SELECT username, role, last_login FROM users WHERE id=?", (user_id,))
    result = c.fetchone()
    if result:
        # Retrieve the username and role, last_login from the result tuple
        username, role, last_login = result
        # Create a User object with the retrieved user ID and role
        user = User(user_id, role)
        user.username = username  # Add the username attribute to the User object

        # check if last_login value exists in the database
        if last_login:
            datetime_last_login = parser.parse(last_login)
            formatted_last_login = datetime_last_login.strftime('%A, %d %B %Y, %I:%M:%S %p')
            user.last_login = formatted_last_login

        return user
    return None


@login_manager.unauthorized_handler
def unauthorized_callback():
    flash('Please log in first to access this page!')
    return redirect(url_for('login'))


# Page Not Found
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error_handling/404.html'), 404


# Internal Server Error
@app.errorhandler(500)
def internal_error(error):
    return render_template('error_handling/500.html'), 500


# merged user and attendance databases (merged_database.db)
def merge_databases():

        # Connect to the user database
        user_db_conn = sqlite3.connect('user_database.db')
        user_db_cursor = user_db_conn.cursor()

        # Connect to the attendance database
        attendance_db_conn = sqlite3.connect('attendance_database.db')
        attendance_db_cursor = attendance_db_conn.cursor()

        # Create the merged database and table
        merged_db_conn = sqlite3.connect('merged_database.db')
        merged_db_cursor = merged_db_conn.cursor()
        merged_db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS merged_table (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT,
                module TEXT,
                last_marked_time TEXT,
                image BLOB
            )
        ''')

        # Merge the data from both databases
        user_db_cursor.execute('SELECT id, username, role, last_login, image FROM users')
        user_data = user_db_cursor.fetchall()

        attendance_db_cursor.execute('SELECT username, timestamp, module, last_marked_time FROM attendance')
        attendance_data = attendance_db_cursor.fetchall()

        for user in user_data:
            username = user[1]
            role = user[2]
            last_login = user[3]
            image = user[4]

            # Initialize variables to None for cases where username does not match in attendance database
            timestamp = None
            module = None
            last_marked_time = None

            # Find the corresponding attendance data for the user
            for attendance in attendance_data:
                if attendance[0] == username:
                    timestamp = attendance[1]
                    module = attendance[2]
                    last_marked_time = attendance[3]
                    break

            # Check if the record with the same username already exists in the merged database
            merged_db_cursor.execute('SELECT id FROM merged_table WHERE username = ?', (username,))
            existing_record = merged_db_cursor.fetchone()

            # If a record with the same username exists, update the merged data in the merged database
            if existing_record:
                merged_db_cursor.execute('''
                    UPDATE merged_table 
                    SET role=?, timestamp=?, last_login=?, module=?, last_marked_time=?, image=?
                    WHERE username=?
                ''', (role, timestamp, last_login, module, last_marked_time, image, username))
            # If no record with the username exists, insert the merged data into the merged database
            else:
                merged_db_cursor.execute('''
                    INSERT INTO merged_table (username, role, timestamp, last_login, module, last_marked_time, image)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (username, role, timestamp, last_login, module, last_marked_time, image))

        # Commit the changes and close the connections
        merged_db_conn.commit()
        merged_db_conn.close()
        user_db_conn.close()
        attendance_db_conn.close()

# call the function to perform the merging
merge_databases()

# def start_camera():
#     global camera
#     # Initialize the camera
#     camera = cv2.VideoCapture(0)
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

#     if not camera.isOpened():
#         raise IOError("Failed to open the camera.")

#     # Start the camera loop in a separate thread
#     threading.Thread(target=camera_loop, daemon=True).start()

#     # # Start the camera loop
#     # camera_loop()


# # use this for actual camera \/
# # camera = cv2.VideoCapture("rtsp://admin:Citi123!@192.168.1.64:554/Streaming/Channels/302")
# def camera_loop():
#     global camera
#     camera = cv2.VideoCapture(0)
#     camera.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
#     camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

#     while True:
#         # Read a frame from the camera
#         success, frame = camera.read()

#         if not success:
#             break

#         # Perform face recognition on the frame
#         recognized_frame = recognize_face(frame)
#         if recognized_frame is not None:
#             cv2.imshow("Recognition Camera", recognized_frame)
#         else:
#             cv2.imshow("Recognition Camera", frame)

#         # Wait for a key press and check if 'q' is pressed to exit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     camera.release()
#     cv2.destroyAllWindows()


# Function to get the thread-local connection
def get_connection():
    if not hasattr(local_storage, 'conn'):
        # Create a new connection for this thread
        local_storage.conn = sqlite3.connect('user_database.db')
    return local_storage.conn


def get_cursor():
    if not hasattr(local_storage, 'cursor'):
        # Create a new cursor for this thread
        local_storage.cursor = get_connection().cursor()
    return local_storage.cursor


# Function to get the thread-local connection for attendance database
def get_attendance_connection():
    if not hasattr(local_storage, 'attendance_conn'):
        # Create a new connection for this thread
        local_storage.attendance_conn = sqlite3.connect('attendance_database.db')
    return local_storage.attendance_conn


def get_attendance_cursor():
    if not hasattr(local_storage, 'attendance_cursor'):
        # Create a new cursor for this thread
        local_storage.attendance_cursor = get_attendance_connection().cursor()
    return local_storage.attendance_cursor


def add_module(module_name, start_time, end_time):
    c = get_attendance_cursor()
    c.execute("INSERT INTO modules (module_name, start_time, end_time) VALUES (?, ?, ?)",
              (module_name, start_time, end_time))
    get_attendance_connection().commit()


def get_modules():
    c = get_attendance_cursor()
    c.execute("SELECT id, module_name, start_time, end_time FROM modules")
    return c.fetchall()


def update_module(module_id, module_name, start_time, end_time):
    c = get_attendance_cursor()
    c.execute("UPDATE modules SET module_name=?, start_time=?, end_time=? WHERE id=?",
              (module_name, start_time, end_time, module_id))
    get_attendance_connection().commit()


def delete_module(module_id):
    c = get_attendance_cursor()
    c.execute("DELETE FROM modules WHERE id=?", (module_id,))
    get_attendance_connection().commit()


# Function to register a new user
def register_user(username, password, role, image_data):
    # Generate the password hash
    password_hash = generate_password_hash(password)

    # Insert the user into the database
    c = get_cursor()
    c.execute("INSERT INTO users (username, password, image, role) VALUES (?, ?, ?, ?)",
              (username, password_hash, image_data, role))  # Assign the provided role during registration
    get_connection().commit()


# Function for face recognition
def recognize_face(input_image):
    # Load the user images and labels from the database
    c = get_cursor()
    c.execute("SELECT id, image FROM users")
    users = c.fetchall()
    if len(users) == 0:
        return "No users registered."

    # Prepare the face recognition model
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    gray_faces = []
    labels = []

    # Preprocess the user images and labels
    for user in users:
        face_image = cv2.imdecode(np.frombuffer(user[1], np.uint8), cv2.IMREAD_GRAYSCALE)
        face_label = user[0]
        gray_faces.append(face_image)
        labels.append(face_label)

    # Train the face recognition model with the user images and labels
    face_recognizer.train(gray_faces, np.array(labels))

    # Convert the input image to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Perform face detection using OpenCV's Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.18, minNeighbors=6, minSize=(100, 100))

    # If no face is detected or multiple faces are detected, return None
    if len(faces) == 0 or len(faces) > 1:
        return None

    # Extract the coordinates of the detected face
    (x, y, w, h) = faces[0]

    # Create a copy of the input image
    result_image = input_image.copy()

    # Crop and resize the face region
    face_roi = gray[y:y + h, x:x + w]
    face_roi = cv2.resize(face_roi, (100, 100))

    # Predict the label and confidence of the face
    label, confidence = face_recognizer.predict(face_roi)

    # Calculate the confidence percentage
    confidence_percentage = int(100 * (1 - (confidence / 400)))

    # Draw the rectangle and label on the result image
    cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    username = get_username(label)
    cv2.putText(result_image, f"User: {username}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # cv2.putText(result_image, f"Confidence: {confidence_percentage}%", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
    #             (0, 255, 0), 2)

    if confidence_percentage >= 75:
        mark_attendance(username)

    return result_image


def mark_attendance(username):
    singapore_tz = pytz.timezone('Asia/Singapore')
    current_time = datetime.datetime.now(singapore_tz)

    modules = get_modules()

    module = "Unknown"
    for module_data in modules:
        module_id, module_name, start_time_str, end_time_str = module_data
        start_time = datetime.datetime.strptime(start_time_str, '%H:%M').time()
        end_time = datetime.datetime.strptime(end_time_str, '%H:%M').time()
        current_date = current_time.date()
        start_datetime = singapore_tz.localize(datetime.datetime.combine(current_date, start_time))
        end_datetime = singapore_tz.localize(datetime.datetime.combine(current_date, end_time))
        if start_datetime <= current_time <= end_datetime:
            module = module_name
            break

    c = get_attendance_cursor()
    c.execute("SELECT last_marked_time FROM attendance WHERE username=? ORDER BY timestamp DESC LIMIT 1", (username,))
    last_marked_time = c.fetchone()
    
    if last_marked_time:
        last_marked_time = datetime.datetime.strptime(last_marked_time[0], '%d %B %Y, %I:%M %p')
        last_marked_time = singapore_tz.localize(last_marked_time)
        time_diff = current_time - last_marked_time
        
        if time_diff.total_seconds() >= 60:
            formatted_time = current_time.strftime('%d %B %Y, %I:%M %p')
            c.execute("INSERT INTO attendance (username, module, timestamp, last_marked_time) VALUES (?, ?, ?, ?)",
                      (username, module, formatted_time, formatted_time))
            get_attendance_connection().commit()
            print(f"{username} marked present for {module} at {formatted_time}.")
        else:
            remaining_time = datetime.timedelta(seconds=60) - time_diff
            print(f"{username} can only mark attendance for {module} after {remaining_time.total_seconds()} seconds.")
    else:
        formatted_time = current_time.strftime('%d %B %Y, %I:%M %p')
        c.execute("INSERT INTO attendance (username, module, timestamp, last_marked_time) VALUES (?, ?, ?, ?)",
                  (username, module, formatted_time, formatted_time))
        get_attendance_connection().commit()
        print(f"{username} marked present for {module} at {formatted_time}.")


def get_username(label):
    c = get_cursor()
    c.execute("SELECT username FROM users WHERE id=?", (label,))
    result = c.fetchone()
    if result:
        return result[0]

    return "Unknown"


# Function for camera thread
def camera_thread():
    global camera
    global current_frame
    while True:
        success, frame = camera.read()
        if not success:
            break
        # Store the current frame in the global variable
        current_frame = frame.copy()


# Generator function to generate video stream
def generate_frames():
    while True:
        try:
            # Capture video frames
            success, frame = camera.read()
            if not success:
                raise cv2.error('Failed to capture frame')

            # Convert the frame to RGB for better compatibility with OpenCV
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform face recognition on the frame
            recognized_frame = recognize_face(rgb_frame)
            if recognized_frame is not None:
                # Check if recognized_frame is a valid numpy array
                if isinstance(recognized_frame, np.ndarray):
                    # Convert the recognized frame to BGR for displaying
                    bgr_frame = cv2.cvtColor(recognized_frame, cv2.COLOR_RGB2BGR)

                    # Encode the frame as JPEG
                    ret, buffer = cv2.imencode('.jpg', bgr_frame)
                    frame_bytes = buffer.tobytes()

                    # Yield the frame bytes as a response to the client
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except cv2.error as e:
            print(f'Error occurred: {str(e)}')


@app.route('/')
def index():
    return redirect('/login')


# Route for the register page
@app.route('/register')
def register1():
    return render_template('account_registration.html')


# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register2():
    # Get the username, password, and image file from the form
    username = request.form['username']
    password = request.form['password']
    confirm_password = request.form['confirm_password']
    image_file = request.files['image']

    # Required Fields
    if not username or not password or not confirm_password:
        error = 'All fields are required!'
        return render_template('account_registration.html', error=error)

    # Minimum and maximum length for username
    if len(username) < 3:
        error = 'Username should be 3 characters or more!'
        return render_template('account_registration.html', error=error)

    # Length of password must be at least 8 characters, enforce password complexity
    if len(password) < 8 or not (re.search(r"[A-Z]", password) and
                                 re.search(r"[a-z]", password) and
                                 re.search(r"\d", password) and
                                 re.search(r"[!@#$%^&*(),.?\":{}|<>]", password)):
        error = 'Password should be at least 8 characters long and include a combination of uppercase, lowercase, digits, and special characters!'
        return render_template('account_registration.html', error=error)

    # Check if password matches with confirm password
    if password != confirm_password:
        error = 'Passwords do not match!'
        return render_template('account_registration.html', error=error)

    # Check if the username is already taken
    c = get_cursor()
    c.execute("SELECT username FROM users WHERE username=?", (username,))
    existing_user = c.fetchone()
    if existing_user:
        error = 'Username has already been taken!'
        return render_template('account_registration.html', error=error)

    # Load and process the user image
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection using OpenCV's Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)

    # Check if a single face is detected
    if len(faces) != 1:
        error = 'No face can be detected in the image sent!'
        return render_template('account_registration.html', error=error)

    # Crop and resize the face region
    (x, y, w, h) = faces[0]
    face = gray[y:y + h, x:x + w]
    face = cv2.resize(face, (100, 100))

    # Convert the face image to bytes
    _, buffer = cv2.imencode('.jpg', face)
    image_data = buffer.tobytes()

    # Register the user with the role 'user'
    register_user(username, password, 'user', image_data)
    return redirect('/login')


# Route for login
@app.route('/login')
def login():
    return render_template('account_login.html')


# Route for login
@app.route('/login', methods=['POST'])
def login_submit():
    username = request.form['username']
    password = request.form['password']

    # Check if password matches with confirm password
    if not username or not password:
        error = 'Username and password are required.'
        return render_template('account_login.html', error=error)

    c = get_cursor()
    c.execute("SELECT id, username, password, role, last_login FROM users WHERE username=?", (username,))
    user = c.fetchone()

    if user and check_password_hash(user[2], password):
        user_obj = User(user[0], user[3])  # Pass the role as the second argument

        if user[4]:
            last_login_datetime = parser.parse(user[4])
            user_obj.last_login = last_login_datetime

        if user_obj.first_login:
            # For first time logging in, display the current login datetime
            user_obj.first_login = False
            current_login = datetime.datetime.now()
            user_obj.set_last_login(current_login)
        else:
            # Display last/previous login datetime for subsequent logins
            current_login = datetime.datetime.now()
            user_obj.set_last_login(current_login)

        user_obj.update_last_login()  # Update last login in user database

        login_user(user_obj)  # Log in the user

        if current_user.is_admin():
            return redirect(url_for('dashboard'))
        else:
            return redirect(url_for('attendance'))
    else:
        # Invalid login
        error = 'The username or password is wrong!'
        return render_template('account_login.html', error=error)


# Route for logging out
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')


# Route for terms and conditions
@app.route('/terms')
@login_required
def user_terms():

    # retrieve last login and previous login datetimes from user model
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    return render_template('user/terms_and_conditions.html', user=user, last_login=last_login, previous_login=previous_login)


# Route for user guide
@app.route('/user_guide', endpoint='user_guide')
@login_required
def user_guide():

    # retrieve last login and previous login datetimes from user model
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    return render_template('user/user_guide.html', last_login=last_login, previous_login=previous_login)


# Route for displaying attendance records
@app.route('/attendance')
@login_required
def attendance():
    username_query = request.args.get('username')
    module_query = request.args.get('module')

    if current_user.is_admin():
        # Retrieve all attendance records from the database for admin
        c = get_attendance_cursor()
        if username_query:
            c.execute("SELECT * FROM attendance WHERE username LIKE ?", ('%' + username_query + '%',))
        elif module_query:
            c.execute("SELECT * FROM attendance WHERE module LIKE ?", ('%' + module_query + '%',))
        else:
            c.execute("SELECT * FROM attendance")
        attendance_records = c.fetchall()

        # pagination query
        page = int(request.args.get('page', 1))
        per_page = 10
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_users = attendance_records[start_idx:end_idx]
        total_pages = len(attendance_records) // per_page + (1 if len(users) % per_page > 0 else 0)

        # retrieve last login and previous login datetimes from user model
        last_login = current_user.get_last_login()
        previous_login = current_user.get_previous_login()

        # Render the admin attendance template with the records
        return render_template('admin/admin_attendance.html', attendance_records=attendance_records,
                               users_data=paginated_users, current_page=page, total_pages=total_pages,
                               last_login=last_login, previous_login=previous_login)

    else:
        # Retrieve the attendance records for the current user from the database for user
        c = get_attendance_cursor()
        if module_query:
            c.execute("SELECT * FROM attendance WHERE module LIKE ? AND username=?",
                      ('%' + module_query + '%', current_user.username))
        else:
            c.execute("SELECT * FROM attendance WHERE username=?", (current_user.username,))
        attendance_records = c.fetchall()

        # pagination query
        page = int(request.args.get('page', 1))
        per_page = 10
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_users = attendance_records[start_idx:end_idx]
        total_pages = len(attendance_records) // per_page + (1 if len(users) % per_page > 0 else 0)

        # retrieve last login and previous login datetimes from user model
        last_login = current_user.get_last_login()
        previous_login = current_user.get_previous_login()

        # Render the admin attendance template with the records
        return render_template('user/user_attendance.html', attendance_records=attendance_records,
                               users_data=paginated_users, current_page=page, total_pages=total_pages,
                               last_login=last_login, previous_login=previous_login)


# Route for deleting attendance records (only accessible to admins)
@app.route('/delete_attendance', methods=['POST'])
@login_required
def delete_attendance():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Get the attendance ID from the form data
    attendance_id = request.form['attendance_id']

    # Delete the attendance record from the database
    c = get_attendance_cursor()
    c.execute("DELETE FROM attendance WHERE id=?", (attendance_id,))
    get_attendance_connection().commit()
    return redirect(url_for('attendance'))


# Route for clearing all attendance records (only accessible to admins)
@app.route('/clear_attendance', methods=['POST'])
@login_required
def clear_attendance():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Clear all attendance records from the database
    c = get_attendance_cursor()
    c.execute("DELETE FROM attendance")
    get_attendance_connection().commit()
    return redirect(url_for('attendance'))


# Route for face recognition
@app.route('/recognize')
def recognize():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route for admin guide
@app.route('/admin_guide')
@login_required
def admin_guide():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    return render_template('admin/admin_guide.html', user=user, last_login=last_login, previous_login=previous_login)


# Route for admin dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if not current_user.is_admin():
        return render_template('access_denied.html')

    try:
        # Connect to the database
        conn = sqlite3.connect("merged_database.db")
        cursor = conn.cursor()

        # check if there is a search query for username
        search_username = request.args.get('search_username')
        if search_username is None:
            search_username = ''

        # handles search query for module
        search_module = request.args.get('search_module')
        if search_module is None:
            search_module = ''

        # check if there is a pagination query
        page = int(request.args.get('page', 1))

        # Retrieve all data, including the image blob and the new columns, filtered by username, and module if provided
        if search_username:
            cursor.execute("SELECT id, username, role, timestamp, last_login, module, last_marked_time, image FROM merged_table WHERE username LIKE ?",
                           ('%' + search_username + '%',))
        elif search_module:
            cursor.execute("SELECT id, username, role, timestamp, last_login, module, last_marked_time, image FROM merged_table WHERE module LIKE ?",
                           ('%' + search_module + '%',))
        else:
            cursor.execute("SELECT id, username, role, timestamp, last_login, module, last_marked_time, image FROM merged_table")

        rows = cursor.fetchall()

        data_list = []

        timestamp_counts = {}  # dictionary to store the counts for each day
        username_counts = {}  # dictionary to store the counts of daily timestamps per day for each username

        # new dictionary to store the counts of last logins per day for all users
        last_login_counts_per_day = defaultdict(int)

        for row in rows:
            data = {
                'id': row[0],
                'username': row[1],
                'role': row[2],
                'timestamp': row[3],
                'last_login': row[4],
                'module': row[5],
                'last_marked_time': row[6]
            }

            # Convert the image blob to a base64 encoded string
            image_data = BytesIO(row[7])
            image = Image.open(image_data)
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            data['image_data'] = img_str
            data_list.append(data)

            # Check if the last_login timestamp is not None before parsing
            if row[3] is not None:
                # extract date component from datetime object
                dt_object = parser.parse(row[3])
                date_str = dt_object.strftime('%d %B %Y')  # fix date format here

                # Increment the count for the date in the timestamp_counts dictionary
                timestamp_counts[date_str] = timestamp_counts.get(date_str, 0) + 1

                # increment the count of daily timestamps per day for each username
                if search_username:
                    username_date_key = f"{row[1]}_{date_str}"
                    username_counts[username_date_key] = username_counts.get(username_date_key, 0) + 1

            if row[4] is not None:  # check if last login timestamp is not none
                last_login_dt = parser.parse(row[4])
                last_login_date = last_login_dt.date()  # extract only the date part

                # Increment the count for the date in the last_login_counts_per_day dictionary
                last_login_counts_per_day[last_login_date] += 1

        conn.close()

        # pagination query
        per_page = 5 # for demo purpose only, remember to change back to `'10'
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_users = data_list[start_idx:end_idx]
        total_pages = len(data_list) // per_page + (1 if len(data_list) % per_page > 0 else 0)

        # derive total count, number of user roles and admin roles
        data_length = len(rows)
        user_roles = sum(1 for data in rows if data[2] == 'user')
        admin_roles = sum(1 for data in rows if data[2] == 'admin')

        # derive average login frequency, number of unique modules and total last marked time (attendance)
        total_last_login = 0
        unique_modules = {}  # store module names and their occurrences
        total_last_marked_time_count = 0
        marked_time_percentages = {}  # store marked time percentages for each user and module

        for data in data_list:
            if data['last_login']:
                total_last_login += 1

            module = data['module']
            if module:
                if module in unique_modules:
                    unique_modules[module] += 1
                else:
                    unique_modules[module] = 1

            if data['last_marked_time']:
                total_last_marked_time_count += 1

            # last marked time by modules
            username = data['username']
            module = data['module']
            if username not in marked_time_percentages:
                marked_time_percentages[username] = {}
            if module not in marked_time_percentages[username]:
                total_modules = 0
                marked_modules = 0

                for data_item in data_list:
                    if data_item['username'] == username and data_item['module'] == module:
                        total_modules += 1
                        if data_item['last_marked_time'] is not None and data_item['last_marked_time'] != 'None':
                            marked_modules += 1

                if total_modules == 0:
                    marked_time_percentages[username][module] = 0
                else:
                    marked_time_percentages[username][module] = (marked_modules / total_modules) * 100

        # calculate average login frequency
        average_login_frequency = total_last_login / len(data_list) if len(data_list) > 0 else 0

        # calculate the total login frequency
        total_login_frequency = total_last_login

        # calculate the number of unique modules
        num_unique_modules = len(unique_modules)
        unique_modules_data = [(module, count) for module, count in unique_modules.items()]

        total_last_marked_time = total_last_marked_time_count

        # retrieve last login and previous login datetimes from user model (admin role)
        last_login = current_user.get_last_login()
        previous_login = current_user.get_previous_login()

        return render_template('admin_dashboard.html', user=user,
                               data=paginated_users, page=page, total_pages=total_pages,
                               search_username=search_username, data_length=data_length,
                               user_roles=user_roles, admin_roles=admin_roles,
                               timestamp_counts=timestamp_counts,
                               username_counts=username_counts,
                               last_login=last_login, previous_login=previous_login,
                               average_login_frequency=average_login_frequency,
                               num_unique_modules=num_unique_modules,
                               total_last_marked_time=total_last_marked_time,
                               total_login_frequency=total_login_frequency,
                               unique_modules_data=unique_modules_data,
                               marked_time_percentages=marked_time_percentages,
                               last_login_counts_per_day=last_login_counts_per_day
                               )

    except sqlite3.Error as e:
        print("Error occurred while retrieving the data:", e)
        return "Error occurred while retrieving the data: " + str(e)


# Route for recognition
@app.route('/recognition')
@login_required
def recognition():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    return render_template('admin/recognition.html', last_login=last_login, previous_login=previous_login)


# @app.route('/start_recognition', methods=['GET'])
# @login_required
# def start_recognition():
#     if not current_user.is_admin():
#         return render_template('access_denied.html')  # Redirect to an error page
#     else:
#         # Start the camera loop in a separate thread
#         threading.Thread(target=camera_loop, daemon=True).start()
#         return jsonify({'status': 'success'})


# Route for managing users
@app.route('/manage_users')
@login_required
def manage_users():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Retrieve all users from the database
    c = get_cursor()
    c.execute("SELECT id, username, role FROM users")
    users = c.fetchall()

    # pagination query
    page = int(request.args.get('page', 1))
    per_page = 10
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_users = users[start_idx:end_idx]
    total_pages = len(users) // per_page + (1 if len(users) % per_page > 0 else 0)

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    # Render the manage users template with the users data
    return render_template('admin/manage_users.html', users=users,
                           users_data=paginated_users, current_page=page, total_pages=total_pages,
                           last_login=last_login, previous_login=previous_login)


# Route for creating user through managing user page
@app.route('/create_user', methods=['GET', 'POST'])
@login_required
def create_user():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    if request.method == 'POST':
        # Get the username, password, and image file from the form
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        image_file = request.files['image']

        # Required Fields
        if not username or not password or not confirm_password:
            error = 'All fields are required!'
            return render_template('admin/create_user.html', error=error)

        # Length of password must be at least 8
        if len(password) < 8:
            error = 'Password should be at least 8 characters long!'
            return render_template('admin/create_user.html', error=error)

        # Check if password matches with confirm password
        if password != confirm_password:
            error = 'Passwords do not match!'
            return render_template('admin/create_user.html', error=error)

        c = get_cursor()
        c.execute("SELECT username FROM users WHERE username=?", (username,))
        existing_user = c.fetchone()
        if existing_user:
            error = 'Username has already been taken!'
            return render_template('admin/create_user.html', error=error)

        # Load and process the user image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform face detection using OpenCV's Haar cascades
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=7)

        # Check if a single face is detected
        if len(faces) != 1:
            error = 'No face can be detected in the image sent!'
            return render_template('admin/create_user.html', error=error)

        # Crop and resize the face region
        (x, y, w, h) = faces[0]
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))

        # Convert the face image to bytes
        _, buffer = cv2.imencode('.jpg', face)
        image_data = buffer.tobytes()

        register_user(username, password, 'user', image_data)  # Adjust image_data accordingly
        return redirect(url_for('manage_users'))

    return render_template('admin/create_user.html', last_login=last_login, previous_login=previous_login)


# Route for editing a user
@app.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Retrieve the user from the database based on the user_id
    c = get_cursor()
    c.execute("SELECT id, username, role FROM users WHERE id=?", (user_id,))
    user = c.fetchone()

    if user:
        if request.method == 'POST':
            # Get the updated username and role from the form
            updated_username = request.form['username']
            updated_role = request.form['role']

            # Update the user in the database
            c.execute("UPDATE users SET username=?, role=? WHERE id=?", (updated_username, updated_role, user_id))
            get_connection().commit()
            return redirect(url_for('manage_users'))
        else:
            # Render the edit user template with the user data
            return render_template('admin/edit_user.html', user=user)
    else:
        return "User not found"  # Redirect to an error page or display an error message


# Route for deleting a user
@app.route('/delete_user/<int:user_id>', methods=['GET', 'POST'])
@login_required
def delete_user(user_id):
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Retrieve the user from the database based on the user_id
    c = get_cursor()
    c.execute("SELECT id, username FROM users WHERE id=?", (user_id,))
    user = c.fetchone()

    if user:
        if request.method == 'POST':
            # Delete the user from the database
            c.execute("DELETE FROM users WHERE id=?", (user_id,))
            get_connection().commit()
            return redirect(url_for('manage_users'))
        else:
            # Render the delete user template with the user data
            return render_template('admin/delete_user.html', user=user)
    else:
        return "User not found"  # Redirect to an error page or display an error message


# Route for viewing modules
@app.route('/view_modules')
@login_required
def view_modules():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Retrieve all modules from the database
    c = get_attendance_cursor()
    c.execute("SELECT id, module_name, start_time, end_time FROM modules")
    modules = c.fetchall()

    # pagination query
    page = int(request.args.get('page', 1))
    per_page = 10
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_users = modules[start_idx:end_idx]
    total_pages = len(modules) // per_page + (1 if len(modules) % per_page > 0 else 0)

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    # Render the view_modules template with the list of modules
    return render_template('admin/admin_modules.html', modules=modules, users_data=paginated_users, current_page=page, total_pages=total_pages,
                           last_login=last_login, previous_login=previous_login)


# Route for adding new modules
@app.route('/add_modules', methods=['GET', 'POST'])
@login_required
def add_modules():
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    if request.method == 'POST':
        # Get the new module details from the form
        module_name = request.form['module_name']
        start_time = request.form['start_time']
        end_time = request.form['end_time']

        # Perform validation if needed (e.g., checking if the module already exists)

        # Insert the new module into the database
        c = get_attendance_cursor()
        c.execute("INSERT INTO modules (module_name, start_time, end_time) VALUES (?, ?, ?)",
                  (module_name, start_time, end_time))
        get_attendance_connection().commit()
        return redirect(url_for('view_modules'))
    else:
        # Render the add_module template for the GET request
        return render_template('admin/add_modules.html', last_login=last_login, previous_login=previous_login)


# Route for editing modules
@app.route('/edit_modules/<int:module_id>', methods=['GET', 'POST'])
@login_required
def edit_modules(module_id):
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Retrieve the module from the database based on the module_id
    c = get_attendance_cursor()

    # Set the row_factory to sqlite3.Row
    c.row_factory = sqlite3.Row

    c.execute("SELECT id, module_name, start_time, end_time FROM modules WHERE id=?", (module_id,))
    module = c.fetchone()

    if module:
        if request.method == 'POST':
            # Get the updated module details from the form
            updated_module_name = request.form['module_name']
            updated_start_time = request.form['start_time']
            updated_end_time = request.form['end_time']

            # Update the module in the database
            update_module(module_id, updated_module_name, updated_start_time, updated_end_time)
            return redirect(url_for('view_modules'))
        else:
            # Access the id value directly from the fetched dictionary-like object
            module_id = module['id']

            # Render the edit_modules template with the module data
            return render_template('admin/edit_modules.html', module=module, module_id=module_id)
    else:
        return "Module not found"  # Redirect to an error page or display an error message


# Route for deleting a module
@app.route('/delete_modules/<int:module_id>', methods=['GET', 'POST'])
@login_required
def delete_modules(module_id):
    if not current_user.is_admin():
        return render_template('access_denied.html')  # Redirect to an error page

    # Retrieve the module from the database based on the module_id
    c = get_attendance_cursor()

    # Set the row_factory to sqlite3.Row
    c.row_factory = sqlite3.Row

    c.execute("SELECT id, module_name, start_time, end_time FROM modules WHERE id=?", (module_id,))
    module = c.fetchone()

    if module:
        if request.method == 'POST':
            # Perform the deletion of the module from the database
            delete_module(module_id)
            return redirect(url_for('view_modules'))
        else:
            # Access the id value directly from the fetched dictionary-like object
            module_id = module['id']

            # Render the delete_modules template with the module data
            return render_template('admin/delete_modules.html', module=module, module_id=module_id)
    else:
        return "Module not found"  # Redirect to an error page or display an error message


# create a notification broadcast table
conn = sqlite3.connect('broadcast_messages.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS broadcast_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        sender_name TEXT NOT NULL,
        sender_role TEXT NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()

# route for admin broadcast
@app.route('/broadcast')
@login_required
def broadcast():

    if not current_user.is_admin():
        return render_template('access_denied.html') # Redirect to an error page

    # retrieve last login and previous login datetimes from user model (admin role)
    last_login = current_user.get_last_login()
    previous_login = current_user.get_previous_login()

    return render_template('admin/broadcast.html', previous_login=previous_login, last_login=last_login)

# notification broadcast handler
@socketio.on('broadcast')
def handle_broadcast(data):
    message = data['message']
    title = message['title']
    content = message['content']
    sender_name = message['sender']['name']
    sender_role = message['sender']['role']

    # Convert to Singapore Time (SGT)
    singapore_tz = pytz.timezone('Asia/Singapore')
    local_time = datetime.datetime.now(singapore_tz)
    timestamp = local_time.strftime('%Y-%m-%d %I:%M:%S %p')

    # Save the message to the database
    conn = sqlite3.connect('broadcast_messages.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO broadcast_messages (title, content, sender_name, sender_role, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (title, content, sender_name, sender_role, timestamp))
    conn.commit()
    conn.close()

    # Emit the message along with the timestamp to all connected clients
    emit('new_broadcast', {'message': {**message, 'timestamp': timestamp}}, broadcast=True)
    print("New Broadcast Message:", message)  # Print the message in the console


@socketio.on('connect')
def send_broadcast_messages():
    conn = sqlite3.connect('broadcast_messages.db')
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM broadcast_messages ORDER BY timestamp DESC')
    messages = cursor.fetchall()

    conn.close()

    emit('initial_broadcast', {'messages': messages})


@socketio.on('request_broadcast_messages')
def request_broadcast_messages():
    conn = sqlite3.connect('broadcast_messages.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM broadcast_messages ORDER BY timestamp DESC')
    messages = cursor.fetchall()
    conn.close()

    emit('broadcast_messages', {'messages': messages})


if __name__ == '__main__':
    # Create the attendance table if it doesn't exist
    c = get_attendance_cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    get_attendance_connection().commit()

    # Train the face recognition model with the registered users' images
    c = get_cursor()
    c.execute("SELECT id, image FROM users")
    users = c.fetchall()
    if len(users) > 0:
        faces = []
        labels = []
        for user in users:
            face_image = cv2.imdecode(np.frombuffer(user[1], np.uint8), cv2.IMREAD_GRAYSCALE)
            faces.append(face_image)
            labels.append(user[0])

        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        face_recognizer.train(faces, np.array(labels))

    # Start the camera in a separate thread
    # threading.Thread(target=start_camera, daemon=True).start()

    # Run the Flask app
    socketio.run(app, debug=True)
    # socketio.run(app, host='172.27.186.17', port=5000, debug=True) -> only accessible via nyp network
