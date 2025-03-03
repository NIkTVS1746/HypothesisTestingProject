from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from datetime import datetime, date
from sqlalchemy import create_engine, Column, Integer, String, DateTime, extract, and_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
from typing import Optional, List
from functions import  analyze_market_events 




app = FastAPI()

# Настройки подключения к SQL Server
DATABASE_URL = ""
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class EconomicEvents(Base):
    __tablename__ = "EventReleases"
    __table_args__ = {'extend_existing': True}
    id = Column(Integer, primary_key=True, index=True)
    Date = Column(DateTime)             
    Title = Column(String)             
    Actual_Value = Column(String)       
    Forecast_Value = Column(String)    
    Previous_Value = Column(String)     

class EventDateRange(BaseModel):
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    @validator('end_date')
    def validate_end_date(cls, v, values):  
        if v and values.get('start_date') and v < values['start_date']:
            raise ValueError('end_date must be after start_date')
        return v

class EventFilterCriteria(BaseModel):
    event_name: str = None       
    date_range: Optional[EventDateRange] = None
    limit_rows: int = None        
    threshold_values: list[float] = None  

class Statistics(BaseModel):
    total_hypotheses: int
    correct_hypotheses: int
    success_percentage: float
    win_value: float
    lose_value: float
    result_value: str

class EventData(BaseModel):
    id: int
    date: datetime
    title: str
    actual_value: str
    forecast: str
    previous: str

    class Config:
        from_attributes = True


class EventReleasesResponse(BaseModel):
    events: list[EventData]
    statistics: Statistics

@app.post("/event-releases", response_model=EventReleasesResponse)
def get_economic_events(filter_criteria: EventFilterCriteria):  
    db = SessionLocal()
    try:
        query = db.query(EconomicEvents)

        if filter_criteria.date_range:
            if filter_criteria.date_range.start_date:
                start_of_day = filter_criteria.date_range.start_date.replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                query = query.filter(EconomicEvents.Date >= start_of_day)
            if filter_criteria.date_range.end_date:
                end_of_day = filter_criteria.date_range.end_date.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                )
                query = query.filter(EconomicEvents.Date <= end_of_day)
        
        query = query.filter(
            and_(
                EconomicEvents.Forecast_Value != "",
                EconomicEvents.Forecast_Value != None
            )
        )
        
        if filter_criteria.event_name:
            query = query.filter(EconomicEvents.Title == filter_criteria.event_name)

        query = query.order_by(EconomicEvents.Date)

        total_events = query.count() 
        if total_events == 0:
            raise HTTPException(status_code=404, detail="No events found")

        limit = min(filter_criteria.limit_rows or total_events, total_events)
        events = query.limit(limit).all()
        
        events_data = [
            {
                "id": event.id,
                "date": event.Date,
                "title": event.Title,
                "actual_value": event.Actual_Value,
                "forecast": event.Forecast_Value,
                "previous": event.Previous_Value
            } for event in events
        ]
        
        analysis_results, statistics = analyze_market_events(events_data, filter_criteria.threshold_values)  
        
        results_df = pd.DataFrame(analysis_results) 
        results_df.to_csv('economic_analysis_results.csv', index=False)  
        
        response = EventReleasesResponse(
            events=[EventData(**event_dict) for event_dict in events_data],
            statistics=Statistics(
                total_hypotheses=statistics["total_hypotheses"],
                correct_hypotheses=statistics["successful_trades"],
                success_percentage=statistics["success_rate"],
                win_value=statistics["total_profit"],
                lose_value=statistics["total_loss"],
                result_value=statistics["net_result"]
            )
        )
        
        return response
    except Exception as e:
        print(f"Error: {e}")  
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

