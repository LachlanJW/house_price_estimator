import os
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String
import uvicorn

# Get SQL parameters and create SQLAlchemy engine
load_dotenv()
SQL_PW = os.getenv("SQL_PW")
DB_NAME = 'houses'
TABLE_NAME = 'houses'

sql_string = f"mysql+mysqlconnector://root:{SQL_PW}@localhost:3306/{DB_NAME}"
engine = create_engine(sql_string)  # Set echo=True to print to console

# Create classes for fastAPI-SQL talking
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


class House(Base):
    __tablename__ = 'houses'

    id = Column(Integer, primary_key=True)
    price = Column(Integer)
    date = Column(String)
    street = Column(String)
    suburb = Column(String)
    state = Column(String)
    postcode = Column(Integer)
    lat = Column(Integer)
    lng = Column(Integer)
    beds = Column(Integer)
    baths = Column(Integer)
    parking = Column(Integer)
    propertyType = Column(String)
    landSize = Column(Integer)


# Create FastAPI class
app = FastAPI()


@app.get("/houses/")
def get_houses():
    db = SessionLocal()
    houses = db.query(House).all()
    db.close()
    return houses


if __name__ == "__main__":
    # Create the table
    Base.metadata.create_all(engine)
    uvicorn.run(app, host="0.0.0.0", port=8000)
