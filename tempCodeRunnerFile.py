import os
import io
import base64
import pickle
import numpy as np
from datetime import datetime, date, timedelta
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, send_file
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
import bcrypt
import cv2
from PIL import Image
from models import db, Admin, Employee, Attendance

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SESSION_SECRET', 'smartface-attendance-pro-2025')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smartface.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

WORK_START_HOUR = 9
WORK_START_MINUTE = 0

FACE_DB_PATH = 'face_db'
os.makedirs(FACE_DB_PATH, exist_ok=True)

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

@login_manager.user_loader
def load_user(user_id):
    return Admin.query.get(int(user_id))

def init_db():
    with app.app_context():
        db.create_all()
        if not Admin.query.filter_by(username='admin').first():
            password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
            admin = Admin(username='admin', password_hash=password_hash.decode('utf-8'))
            db.session.add(admin)
            db.session.commit()
            print("Default admin user created: admin/admin123")

def get_face_embedding(image_path_or_array):
    try:
        from deepface import DeepFace
        result = DeepFace.represent(
            img_path=image_path_or_array,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=True
        )
        if result and len(result) > 0:
            return np.array(result[0]['embedding'])
        return None
    except Exception as e:
        print(f"Error extracting face embedding: {e}")
        return None

def verify_face(image_array, employee):
    try:
        from deepface import DeepFace
        stored_embedding = employee.get_embedding()
        if stored_embedding is None:
            return False, 0
        
        current_embedding = get_face_embedding(image_array)
        if current_embedding is None:
            return False, 0
        
        distance = np.linalg.norm(stored_embedding - current_embedding)
        threshold = 0.68
        similarity = max(0, 1 - distance)
        
        return distance < threshold, similarity
    except Exception as e:
        print(f"Error verifying face: {e}")
        return False, 0

def find_matching_employee(image_array):
    try:
        current_embedding = get_face_embedding(image_array)
        if current_embedding is None:
            return None, 0
        
        employees = Employee.query.all()
        best_match = None
        best_similarity = 0
        threshold = 0.68
        
        for employee in employees:
            stored_embedding = employee.get_embedding()
            if stored_embedding is not None:
                distance = np.linalg.norm(stored_embedding - current_embedding)
                similarity = max(0, 1 - distance)
                
                if distance < threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = employee
        
        return best_match, best_similarity
    except Exception as e:
        print(f"Error finding matching employee: {e}")
        return None, 0

def check_anti_spoofing(image_array):
    try:
        from deepface import DeepFace
        result = DeepFace.extract_faces(
            img_path=image_array,
            detector_backend="retinaface",
            anti_spoofing=True
        )
        if result and len(result) > 0:
            return result[0].get('is_real', False), result[0].get('antispoof_score', 0)
        return False, 0
    except Exception as e:
        print(f"Anti-spoofing check failed: {e}")
        return True, 1.0

@app.route('/')
def index():
    return redirect(url_for('attend'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        admin = Admin.query.filter_by(username=username).first()
        if admin and bcrypt.checkpw(password.encode('utf-8'), admin.password_hash.encode('utf-8')):
            login_user(admin)
            flash('Welcome back! You have been logged in successfully.', 'success')
            next_page = request.args.get('next')
            return redirect(next_page if next_page else url_for('dashboard'))
        else:
            flash('Invalid username or password. Please try again.', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    today = date.today()
    total_employees = Employee.query.count()
    today_attendance = Attendance.query.filter_by(date=today).count()
    late_today = Attendance.query.filter_by(date=today, is_late=True).count()
    
    recent_attendances = Attendance.query.filter_by(date=today).order_by(Attendance.clock_in_time.desc()).limit(10).all()
    
    return render_template('dashboard.html', 
                         total_employees=total_employees,
                         today_attendance=today_attendance,
                         late_today=late_today,
                         recent_attendances=recent_attendances)

@app.route('/dashboard/export')
@login_required
def export_today_attendance():
    today = date.today()
    attendances = Attendance.query.filter_by(date=today).all()
    
    output = io.StringIO()
    output.write('Employee ID,Full Name,Department,Clock In Time,Status\n')
    
    for att in attendances:
        status = 'Late' if att.is_late else 'On Time'
        output.write(f'{att.employee.employee_id},{att.employee.full_name},{att.employee.department},{att.clock_in_time.strftime("%H:%M:%S")},{status}\n')
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=attendance_{today}.csv'}
    )

@app.route('/employees')
@login_required
def employees():
    all_employees = Employee.query.order_by(Employee.registered_at.desc()).all()
    return render_template('employees.html', employees=all_employees)

@app.route('/employees/delete/<int:id>', methods=['POST'])
@login_required
def delete_employee(id):
    employee = Employee.query.get_or_404(id)
    
    if employee.face_image_path and os.path.exists(employee.face_image_path):
        try:
            os.remove(employee.face_image_path)
        except:
            pass
    
    db.session.delete(employee)
    db.session.commit()
    flash(f'Employee {employee.full_name} has been deleted successfully.', 'success')
    return redirect(url_for('employees'))

@app.route('/enroll', methods=['GET', 'POST'])
@login_required
def enroll():
    if request.method == 'POST':
        employee_id = request.form.get('employee_id')
        full_name = request.form.get('full_name')
        department = request.form.get('department')
        image_data = request.form.get('image_data')
        
        if Employee.query.filter_by(employee_id=employee_id).first():
            flash('An employee with this ID already exists.', 'danger')
            return redirect(url_for('enroll'))
        
        if not image_data:
            flash('Please capture or upload a photo.', 'danger')
            return redirect(url_for('enroll'))
        
        try:
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            image_array = np.array(image)
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            is_real, spoof_score = check_anti_spoofing(image_array)
            if not is_real:
                flash('Anti-spoofing check failed. Please use a real face, not a photo or video.', 'danger')
                return redirect(url_for('enroll'))
            
            embedding = get_face_embedding(image_array)
            if embedding is None:
                flash('Could not detect a face in the image. Please try again with a clearer photo.', 'danger')
                return redirect(url_for('enroll'))
            
            filename = f"{employee_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            image_path = os.path.join(FACE_DB_PATH, filename)
            cv2.imwrite(image_path, image_array)
            
            employee = Employee(
                employee_id=employee_id,
                full_name=full_name,
                department=department,
                face_image_path=image_path
            )
            employee.set_embedding(embedding)
            
            db.session.add(employee)
            db.session.commit()
            
            flash(f'Employee {full_name} has been enrolled successfully!', 'success')
            return redirect(url_for('employees'))
            
        except Exception as e:
            print(f"Error enrolling employee: {e}")
            flash(f'Error processing image: {str(e)}', 'danger')
            return redirect(url_for('enroll'))
    
    departments = ['Engineering', 'Human Resources', 'Finance', 'Marketing', 'Sales', 'Operations', 'IT Support', 'Administration']
    return render_template('enroll.html', departments=departments)

@app.route('/attend')
def attend():
    today = date.today()
    recent = Attendance.query.filter_by(date=today).order_by(Attendance.clock_in_time.desc()).limit(5).all()
    return render_template('attend.html', recent_attendances=recent)

@app.route('/api/recognize', methods=['POST'])
def recognize():
    try:
        data = request.get_json()
        image_data = data.get('image')
        
        if not image_data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        is_real, spoof_score = check_anti_spoofing(image_array)
        if not is_real:
            return jsonify({
                'success': False,
                'message': 'Anti-spoofing check failed. Please show your real face.'
            })
        
        employee, similarity = find_matching_employee(image_array)
        
        if employee is None:
            return jsonify({
                'success': False,
                'message': 'Face not recognized. Please ensure you are enrolled in the system.'
            })
        
        today = date.today()
        existing = Attendance.query.filter_by(employee_id=employee.id, date=today).first()
        
        if existing:
            return jsonify({
                'success': True,
                'already_marked': True,
                'employee_name': employee.full_name,
                'employee_id': employee.employee_id,
                'department': employee.department,
                'clock_in_time': existing.clock_in_time.strftime('%H:%M:%S'),
                'message': f'Attendance already marked for {employee.full_name} today at {existing.clock_in_time.strftime("%H:%M:%S")}'
            })
        
        now = datetime.now()
        is_late = now.hour > WORK_START_HOUR or (now.hour == WORK_START_HOUR and now.minute > WORK_START_MINUTE)
        
        attendance = Attendance(
            employee_id=employee.id,
            date=today,
            clock_in_time=now,
            is_late=is_late
        )
        db.session.add(attendance)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'already_marked': False,
            'employee_name': employee.full_name,
            'employee_id': employee.employee_id,
            'department': employee.department,
            'clock_in_time': now.strftime('%H:%M:%S'),
            'is_late': is_late,
            'similarity': round(similarity * 100, 1),
            'message': f'Attendance marked for {employee.full_name}!'
        })
        
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({
            'success': False,
            'message': f'Error during recognition: {str(e)}'
        })

@app.route('/api/recent-attendances')
def get_recent_attendances():
    today = date.today()
    recent = Attendance.query.filter_by(date=today).order_by(Attendance.clock_in_time.desc()).limit(5).all()
    
    attendances = []
    for att in recent:
        attendances.append({
            'employee_name': att.employee.full_name,
            'employee_id': att.employee.employee_id,
            'department': att.employee.department,
            'clock_in_time': att.clock_in_time.strftime('%H:%M:%S'),
            'is_late': att.is_late
        })
    
    return jsonify({'attendances': attendances})

@app.route('/records')
@login_required
def records():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = Attendance.query
    
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date >= start)
    
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date <= end)
    
    attendances = query.order_by(Attendance.date.desc(), Attendance.clock_in_time.desc()).all()
    
    return render_template('records.html', attendances=attendances, start_date=start_date, end_date=end_date)

@app.route('/records/export')
@login_required
def export_records():
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    query = Attendance.query
    
    if start_date:
        start = datetime.strptime(start_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date >= start)
    
    if end_date:
        end = datetime.strptime(end_date, '%Y-%m-%d').date()
        query = query.filter(Attendance.date <= end)
    
    attendances = query.order_by(Attendance.date.desc(), Attendance.clock_in_time.desc()).all()
    
    output = io.StringIO()
    output.write('Date,Employee ID,Full Name,Department,Clock In Time,Status\n')
    
    for att in attendances:
        status = 'Late' if att.is_late else 'On Time'
        output.write(f'{att.date},{att.employee.employee_id},{att.employee.full_name},{att.employee.department},{att.clock_in_time.strftime("%H:%M:%S")},{status}\n')
    
    output.seek(0)
    filename = f'attendance_records_{datetime.now().strftime("%Y%m%d")}.csv'
    
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)
