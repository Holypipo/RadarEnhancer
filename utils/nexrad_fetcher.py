import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from datetime import datetime

def download_nexrad_file(station, date_obj, time_obj, output_dir="data"):
    """
    Downloads a NEXRAD Level 2 file from AWS S3.
    """
    # Force anonymous access and specify region
    s3 = boto3.client(
        's3', 
        region_name='us-east-1',
        config=Config(signature_version=UNSIGNED)
    )
    bucket = 'noaa-nexrad-level2'
    
    # Prefix format: YYYY/MM/DD/KXXX/
    # We list all files for that day and station to find the best time match
    prefix = f"{date_obj.year}/{date_obj.month:02d}/{date_obj.day:02d}/{station}/"
    
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            print(f"  No data on {date_obj}. Trying previous day...")
            # Fallback to previous day
            from datetime import timedelta
            prev_date = date_obj - timedelta(days=1)
            prefix = f"{prev_date.year}/{prev_date.month:02d}/{prev_date.day:02d}/{station}/"
            response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            
            if 'Contents' not in response:
                return None, f"No data found for station {station} on {date_obj} or {prev_date}."
        
        # Find the file closest to the requested time
        files = [obj['Key'] for obj in response['Contents']]
        
        def get_time(key):
            # File name example: 2013/05/20/KTLX/KTLX20130520_195527_V06
            try:
                base = os.path.basename(key)
                parts = base.split('_')
                date_part = parts[0][-8:] # 20130520
                time_part = parts[1]      # 195527
                return datetime.strptime(date_part + time_part, "%Y%m%d%H%M%S")
            except:
                return datetime.min

        files.sort(key=lambda x: abs(get_time(x) - datetime.combine(date_obj, time_obj)))
        
        best_match = files[0]
        filename = os.path.basename(best_match)
        local_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(local_path):
            print(f"Downloading {best_match}...")
            s3.download_file(bucket, best_match, local_path)
            
        return local_path, None
    except Exception as e:
        return None, f"S3 Error: {str(e)}"
