from flask_login import UserMixin

class User(UserMixin):
    def __init__(self, user_doc):
        self.id = str(user_doc["_id"])  # Required by Flask-Login
        self.email = user_doc.get("email", "")
        self.full_name = user_doc.get("full_name", "")
        self.phone = user_doc.get("phone", "")
        self.goal = user_doc.get("goal", "")
