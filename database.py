from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base

DATABASE_URL = "sqlite:///./stocks.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables on startup"""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency to get database session for each request"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
