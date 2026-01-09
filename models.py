from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
import pickle

db = SQLAlchemy()

class Admin(UserMixin, db.Model):
    __tablename__ = 'admins'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Employee(db.Model):
    __tablename__ = 'employees'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)

    # Binary embedding stored as pickled vector
    face_embedding = db.Column(db.LargeBinary, nullable=True)

    # Path to saved employee face image
    face_image_path = db.Column(db.String(255), nullable=True)

    # Registration timestamp
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Email for notifications (optional)
    email = db.Column(db.String(120), nullable=True)

    # Phone number (optional)
    phone = db.Column(db.String(20), nullable=True)

    # Active status
    is_active = db.Column(db.Boolean, default=True)

    # -------------------------------
    #   EMBEDDING HANDLING METHODS
    # -------------------------------

    def set_embedding(self, embedding_array):
        """Save numpy embedding array as pickle binary."""
        self.face_embedding = pickle.dumps(embedding_array)

    def get_embedding(self):
        """Return embedding as numpy array or None."""
        if not self.face_embedding:
            return None
        return pickle.loads(self.face_embedding)


class Attendance(db.Model):
    __tablename__ = 'attendances'

    id = db.Column(db.Integer, primary_key=True)
    employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)

    date = db.Column(db.Date, nullable=False)
    clock_in_time = db.Column(db.DateTime, nullable=False)
    clock_out_time = db.Column(db.DateTime, nullable=True)  # New: clock out time

    # late flag
    is_late = db.Column(db.Boolean, default=False)

    # Notes (optional)
    notes = db.Column(db.String(255), nullable=True)

    employee = db.relationship('Employee', backref='attendances')

    @property
    def work_duration(self):
        """Calculate work duration if both clock in and out times exist"""
        if self.clock_in_time and self.clock_out_time:
            delta = self.clock_out_time - self.clock_in_time
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            return f"{hours}h {minutes}m"
        return None


class Settings(db.Model):
    __tablename__ = 'settings'

    id = db.Column(db.Integer, primary_key=True)

    # Work hours configuration
    work_start_hour = db.Column(db.Integer, default=9)
    work_start_minute = db.Column(db.Integer, default=0)
    work_end_hour = db.Column(db.Integer, default=17)
    work_end_minute = db.Column(db.Integer, default=0)

    # Recognition settings
    recognition_threshold = db.Column(db.Float, default=0.55)
    anti_spoofing_enabled = db.Column(db.Boolean, default=True)

    # Auto-capture interval in seconds
    auto_capture_interval = db.Column(db.Integer, default=3)

    # Company info
    company_name = db.Column(db.String(100), default='SmartFace Attendance Pro')

    # Timezone
    timezone = db.Column(db.String(50), default='Asia/Manila')

    # Last updated
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @property
    def work_start_time(self):
        """Return formatted work start time"""
        return f"{self.work_start_hour:02d}:{self.work_start_minute:02d}"

    @property
    def work_end_time(self):
        """Return formatted work end time"""
        return f"{self.work_end_hour:02d}:{self.work_end_minute:02d}"
