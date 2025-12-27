from db import engine, Base
from models import User

def create_tables():
    Base.metadata.create_all(bind=engine)
    print("✅ Users table created successfully")

if __name__ == "__main__":
    create_tables()
