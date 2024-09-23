
from datetime import datetime, timedelta
import pandas as pd


def transfer_time(timestamp):
    if isinstance(timestamp,str):
        try:
            time = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S').date()
        except:
            try: 
                time = datetime.strptime(timestamp, '%Y-%m-%d').date()
            except: 
                try: 
                    time = datetime.strptime(timestamp, '%Y-%m').date()
                except:
                    raise Exception(f"timestamp {timestamp} not available")
    elif isinstance(timestamp,int):
        time = datetime.fromtimestamp(timestamp).date()
    elif isinstance(timestamp,pd.Timestamp):
        time = timestamp.date()
    elif isinstance(timestamp,datetime):
        time = timestamp.date()
    else:
        time = timestamp
    return time