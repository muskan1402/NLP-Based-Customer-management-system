# create_admin.py
from backend.app import database, models, crud
from backend.app.database import SessionLocal, engine

# make sure tables exist
models.Base.metadata.create_all(bind=engine)

# open session
db = SessionLocal()

# Define admin data
class Temp:
    username = "admin"
    email = "admin@example.com"
    password = "adminpass"

# create admin (is_admin=True)
crud.create_user(db, Temp, is_admin=True)
print("✅ Admin user created successfully!")
print("   Username: admin")
print("   Password: adminpass")
db.close()
