# Facial Recognition Attendance System

## Overview

The Facial Recognition Attendance System is a comprehensive project developed as part of my final year university requirements. This system leverages advanced facial recognition technology to automate the attendance tracking process for students. By utilizing computer vision and machine learning algorithms, this system aims to enhance accuracy and efficiency in attendance management.

### Objectives

- **Automate Attendance Tracking**: Replace traditional manual attendance methods with an automated system that uses facial recognition to log attendance.
- **Reduce Errors**: Minimize human errors associated with manual attendance recording by using a reliable technology-based solution.
- **Increase Efficiency**: Save time for both educators and students by providing a fast and efficient way to track attendance.
- **User-Friendly Interface**: Design an intuitive user interface that simplifies the setup and operation of the system.

### How It Works

The system operates by capturing real-time images of students as they enter the classroom. These images are processed and compared against a pre-enrolled database of student faces. The system uses machine learning models to recognize and match faces, automatically marking the attendance for each recognized student.

1. **Face Detection**: The system detects faces in the captured images using advanced computer vision techniques.
2. **Face Recognition**: Each detected face is matched against a database of known faces to identify the student.
3. **Attendance Logging**: Once a student is identified, their attendance is recorded in a database, marking them present for that session.
4. **Reporting**: The system generates attendance reports that can be used for tracking and analysis.

### Technical Details

- **Programming Languages**: Python
- **Web Framework**: Flask
- **Database**: SQLite
- **Hardware Requirements**: Webcam or camera for capturing images, a computer with sufficient processing power for running the facial recognition algorithms.

## Features

- **Facial Recognition**: Utilizes state-of-the-art facial recognition technology to accurately identify students.
- **Automated Attendance**: Automatically marks attendance based on facial recognition results.
- **Real-Time Processing**: Processes images and updates attendance in real time.
- **Error Handling**: Includes mechanisms for handling recognition errors and ambiguous cases.
- **User Interface**: Provides an easy-to-use interface for setup and monitoring.

## Getting Started

To get started with this project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ong-xuechen/Facial-Recognition-System-For-Attendance.git
2. **Navigate to the Project Directory**:
   ```bash
   cd Facial-Recognition-System-For-Attendance
3. **Install Dependencies: Ensure you have Python installed, then install the required dependencies**:
   ```bash
   pip install -r requirements.txt
4. **Run the Application: After installing the dependencies, start the application with**:
   ```bash
   python app.py

## Usage

**Access the Application:** Open your web browser and navigate to http://localhost:5000 to access the application.

**Login:** Use the login page to authenticate. Only authorized users can access the application.
**Register:** If the user does not have an existing account, they must register via the 'Register Page' and upload a photo of themselves.

**Attendance Taking: (USER)** The student will automatically have their attendance marked, as they scan their face.
**Attendance Taking: (ADMIN)** The administrator will be able to generate a video capture to scan the faces of the students.

**View Attendance: (USER)** The student will be able to view their attendance after they have logged in.
**View Attendance: (ADMIN)** The administrator will be able to view/edit/remove student's attendance after they have logged in.


