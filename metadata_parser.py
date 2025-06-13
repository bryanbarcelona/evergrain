import csv
import datetime
import random
import re
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def validate_headers(headers):
    """Validate CSV headers against expected columns."""
    expected = [
        'ScanID', 'PhotoID', 'YY', 'MM', 'DD', 'HH', 'MN', 'SS',
        'Timestamp', 'Event', 'Scene', 'Location', 'Tags', 'Cluster'
    ]
    if headers != expected:
        raise ValueError(f"Invalid headers. Expected: {expected}, Got: {headers}")

def is_leap_year(year):
    """Check if year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def validate_timestamp_fields(row, row_num):
    """Validate timestamp fields and convert to integers where possible."""
    fields = {'YY': None, 'MM': None, 'DD': None, 'HH': None, 'MN': None, 'SS': None}
    for field in fields:
        value = row[field].strip() if row[field] else None
        if value:
            try:
                fields[field] = int(value)
            except ValueError:
                raise ValueError(f"Row {row_num}: Invalid {field}='{value}', must be numeric")
    # Validate ranges
    if fields['MM'] is not None and not (1 <= fields['MM'] <= 12):
        raise ValueError(f"Row {row_num}: Invalid MM={fields['MM']}, must be 1–12")
    if fields['DD'] is not None:
        max_days = 31 if fields['MM'] in [1, 3, 5, 7, 8, 10, 12] else 30
        if fields['MM'] == 2:
            year = fields['YY'] + 1900 if fields['YY'] is not None else 2000
            max_days = 29 if is_leap_year(year) else 28
        if not (1 <= fields['DD'] <= max_days):
            raise ValueError(f"Row {row_num}: Invalid DD={fields['DD']}, must be 1–{max_days}")
    if fields['HH'] is not None and not (0 <= fields['HH'] <= 23):
        raise ValueError(f"Row {row_num}: Invalid HH={fields['HH']}, must be 0–23")
    if fields['MN'] is not None and not (0 <= fields['MN'] <= 59):
        raise ValueError(f"Row {row_num}: Invalid MN={fields['MN']}, must be 0–59")
    if fields['SS'] is not None and not (0 <= fields['SS'] <= 59):
        raise ValueError(f"Row {row_num}: Invalid SS={fields['SS']}, must be 0–59")
    return fields

def resolve_cluster(cluster, row_num):
    """Resolve Cluster name, handling typos."""
    if not cluster:
        return None
    cluster = cluster.strip().lower()
    tight_match = re.match(r't.*c.*\d+', cluster)
    loose_match = re.match(r'l.*c.*\d+', cluster)
    if tight_match:
        return f"tightCluster{re.search(r'\d+', cluster).group()}"
    elif loose_match:
        return f"looseCluster{re.search(r'\d+', cluster).group()}"
    else:
        raise ValueError(f"Row {row_num}: Invalid Cluster='{cluster}', must be tightClusterN or looseClusterN")

def get_event_range(rows, event, row_num):
    """Determine timestamp range for an Event."""
    if not event:
        return None
    event_rows = [r for r in rows if r['Event'] == event]
    timestamps = []
    for r in event_rows:
        ts = r['timestamp_fields']
        if any(v is not None for v in ts.values()):
            timestamps.append(ts)
    if not timestamps:
        return None
    # Determine range based on available fields
    min_ts = {'YY': 0, 'MM': 1, 'DD': 1, 'HH': 0, 'MN': 0, 'SS': 0}
    max_ts = {'YY': 99, 'MM': 12, 'DD': 31, 'HH': 23, 'MN': 59, 'SS': 59}
    is_timespan = len(event_rows) > 1 and all(ts.get('DD') is None for ts in timestamps if ts.get('MM') is not None)
    for ts in timestamps:
        if ts['YY'] is not None:
            min_ts['YY'] = max(min_ts['YY'], ts['YY'])
            max_ts['YY'] = min(max_ts['YY'], ts['YY'])
        if ts['MM'] is not None:
            min_ts['MM'] = max(min_ts['MM'], ts['MM'])
            max_ts['MM'] = min(max_ts['MM'], ts['MM'])
        if not is_timespan and ts['DD'] is not None:
            min_ts['DD'] = max(min_ts['DD'], ts['DD'])
            max_ts['DD'] = min(max_ts['DD'], ts['DD'])
        if not is_timespan and ts['HH'] is not None:
            min_ts['HH'] = max(min_ts['HH'], ts['HH'])
            max_ts['HH'] = min(max_ts['HH'], ts['HH'])
    return min_ts, max_ts, is_timespan

def get_scene_range(rows, event, scene, row_num):
    """Determine timestamp range for a Scene within an Event."""
    if not scene:
        return None
    scene_rows = [r for r in rows if r['Event'] == event and r['Scene'] == scene]
    timestamps = [r['timestamp_fields'] for r in scene_rows if any(v is not None for v in r['timestamp_fields'].values())]
    if not timestamps:
        return None
    min_ts = {'YY': 0, 'MM': 1, 'DD': 1, 'HH': 0, 'MN': 0, 'SS': 0}
    max_ts = {'YY': 99, 'MM': 12, 'DD': 31, 'HH': 23, 'MN': 59, 'SS': 59}
    for ts in timestamps:
        if ts['YY'] is not None:
            min_ts['YY'] = max(min_ts['YY'], ts['YY'])
            max_ts['YY'] = min(max_ts['YY'], ts['YY'])
        if ts['MM'] is not None:
            min_ts['MM'] = max(min_ts['MM'], ts['MM'])
            max_ts['MM'] = min(max_ts['MM'], ts['MM'])
        if ts['DD'] is not None:
            min_ts['DD'] = max(min_ts['DD'], ts['DD'])
            max_ts['DD'] = min(max_ts['DD'], ts['DD'])
        if ts['HH'] is not None:
            min_ts['HH'] = max(min_ts['HH'], ts['HH'])
            max_ts['HH'] = min(max_ts['HH'], ts['HH'])
    return min_ts, max_ts

def get_cluster_range(rows, event, scene, cluster, row_num):
    """Determine timestamp range for a Cluster."""
    if not cluster:
        return None
    cluster_rows = [r for r in rows if r['Event'] == event and r['Scene'] == scene and r['Cluster'] == cluster]
    timestamps = [r['timestamp_fields'] for r in cluster_rows if any(v is not None for v in r['timestamp_fields'].values())]
    if not timestamps:
        return None
    min_ts = {'YY': 0, 'MM': 1, 'DD': 1, 'HH': 0, 'MN': 0, 'SS': 0}
    max_ts = {'YY': 99, 'MM': 12, 'DD': 31, 'HH': 23, 'MN': 59, 'SS': 59}
    for ts in timestamps:
        if ts['YY'] is not None:
            min_ts['YY'] = max(min_ts['YY'], ts['YY'])
            max_ts['YY'] = min(max_ts['YY'], ts['YY'])
        if ts['MM'] is not None:
            min_ts['MM'] = max(min_ts['MM'], ts['MM'])
            max_ts['MM'] = min(max_ts['MM'], ts['MM'])
        if ts['DD'] is not None:
            min_ts['DD'] = max(min_ts['DD'], ts['DD'])
            max_ts['DD'] = min(max_ts['DD'], ts['DD'])
        if ts['HH'] is not None:
            min_ts['HH'] = max(min_ts['HH'], ts['HH'])
            max_ts['HH'] = min(max_ts['HH'], ts['HH'])
        if ts['MN'] is not None:
            min_ts['MN'] = max(min_ts['MN'], ts['MN'])
            max_ts['MN'] = min(max_ts['MN'], ts['MN'])
    return min_ts, max_ts

def fill_timestamp(row, rows, row_num):
    """Fill blank timestamp fields based on context."""
    ts = row['timestamp_fields']
    if all(v is None for v in ts.values()) and not (row['Event'] or row['Scene'] or row['Cluster']):
        raise ValueError(f"Row {row_num}: No timestamp fields provided and no context")
    
    # Initialize defaults
    filled = ts.copy()
    filled['YY'] = filled['YY'] or 0
    filled['MM'] = filled['MM'] or 1
    filled['DD'] = filled['DD'] or 1
    filled['HH'] = filled['HH'] or 0
    filled['MN'] = filled['MN'] or 0
    filled['SS'] = filled['SS'] or 0

    # Check Cluster
    if row['Cluster']:
        cluster_range = get_cluster_range(rows, row['Event'], row['Scene'], row['Cluster'], row_num)
        if cluster_range:
            min_ts, max_ts = cluster_range
            if filled['SS'] == 0 and any(r['timestamp_fields']['SS'] is not None for r in rows if r['Cluster'] == row['Cluster']):
                filled['SS'] = random.randint(5, 30)
            if filled['MN'] == 0 and 'loose' in row['Cluster'].lower():
                filled['MN'] = random.randint(1, 15)
            for field in ['YY', 'MM', 'DD', 'HH', 'MN']:
                if filled[field] == (1 if field in ['MM', 'DD'] else 0):
                    filled[field] = min_ts[field]
    
    # Check Scene
    if row['Scene'] and not all(filled[f] for f in ['YY', 'MM', 'DD', 'HH', 'MN']):
        scene_range = get_scene_range(rows, row['Event'], row['Scene'], row_num)
        if scene_range:
            min_ts, max_ts = scene_range
            if filled['MN'] == 0:
                filled['MN'] = random.randint(1, 15)
            if filled['HH'] == 0:
                filled['HH'] = min_ts['HH']
            if filled['DD'] == 1:
                filled['DD'] = min_ts['DD']
            if filled['MM'] == 1:
                filled['MM'] = min_ts['MM']
            if filled['YY'] == 0:
                filled['YY'] = min_ts['YY']
    
    # Check Event
    if row['Event'] and not all(filled[f] for f in ['YY', 'MM', 'DD']):
        event_range = get_event_range(rows, row['Event'], row_num)
        if event_range:
            min_ts, max_ts, is_timespan = event_range
            if is_timespan:
                if filled['DD'] == 1:
                    max_days = 31 if filled['MM'] in [1, 3, 5, 7, 8, 10, 12] else 30
                    if filled['MM'] == 2:
                        max_days = 29 if is_leap_year(filled['YY'] + 1900) else 28
                    filled['DD'] = random.randint(1, max_days)
                if filled['HH'] == 0:
                    filled['HH'] = random.randint(0, 23)
                if filled['MN'] == 0:
                    filled['MN'] = random.randint(0, 59)
                if filled['SS'] == 0:
                    filled['SS'] = random.randint(0, 59)
            else:
                if filled['HH'] == 0:
                    filled['HH'] = random.randint(min_ts['HH'], min(min_ts['HH'] + 3, 23))
                if filled['DD'] == 1:
                    filled['DD'] = min_ts['DD']
                if filled['MM'] == 1:
                    filled['MM'] = min_ts['MM']
                if filled['YY'] == 0:
                    filled['YY'] = min_ts['YY']
    
    # Validate filled timestamp
    try:
        year = filled['YY'] + 1900
        datetime.datetime(year, filled['MM'], filled['DD'], filled['HH'], filled['MN'], filled['SS'])
    except ValueError as e:
        raise ValueError(f"Row {row_num}: Invalid filled timestamp: {filled}, error: {e}")
    
    return filled

def generate_filename(row):
    """Generate filename based on filled timestamp and Event."""
    ts = row['filled_timestamp']
    event = row['Event'].strip() if row['Event'] else ''
    # Replace non-alphanumeric characters with underscores
    event = re.sub(r'[^a-zA-Z0-9]', '_', event)
    # Avoid trailing space if event is empty
    event_part = f" {event}" if event else ''
    return f"{ts['YY']:02d}{ts['MM']:02d}{ts['DD']:02d} {ts['HH']:02d}{ts['MN']:02d}{ts['SS']:02d}{event_part}.jpg"

def process_metadata_csv(csv_path):
    """Process metadata.csv, validate inputs, and fill timestamps."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Detect delimiter
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        sample = csvfile.read()  # Read first 1024 bytes for sniffing
        csvfile.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';'])
            delimiter = dialect.delimiter
            logging.info(f"Detected CSV delimiter: '{delimiter}'")
        except csv.Error:
            logging.warning("Could not detect delimiter, defaulting to comma")
            delimiter = ','

    rows = []
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)
        validate_headers(list(reader.fieldnames))
        
        for row_num, row in enumerate(reader, start=2):
            # Validate row length
            print(row)
            if len(row) != len(reader.fieldnames):
                raise ValueError(f"Row {row_num}: Malformed row, expected {len(reader.fieldnames)} columns, got {len(row)}")
            
            # Validate and parse timestamps
            timestamp_fields = validate_timestamp_fields(row, row_num)
            
            # Resolve Cluster
            cluster = resolve_cluster(row['Cluster'], row_num)
            
            # Store row data
            rows.append({
                'row_num': row_num,
                'timestamp_fields': timestamp_fields,
                'Event': row['Event'].strip() if row['Event'] else '',
                'Scene': row['Scene'].strip() if row['Scene'] else '',
                'Location': row['Location'].strip() if row['Location'] else '',
                'Tags': row['Tags'].strip() if row['Tags'] else '',
                'Cluster': cluster,
                'raw_row': row
            })
    
    # Fill timestamps
    processed_rows = []
    for row in rows:
        filled_timestamp = fill_timestamp(row, rows, row['row_num'])
        row['filled_timestamp'] = filled_timestamp
        row['filename'] = generate_filename(row)
        
        # Validate Tags
        tags = row['Tags'].split('|') if row['Tags'] else []
        tags = [t.strip() for t in tags if t.strip()]
        if row['Tags'].startswith('|') or row['Tags'].endswith('|'):
            logging.warning(f"Row {row['row_num']}: Tags have leading/trailing pipe: {row['Tags']}")
        
        processed_rows.append({
            'row_num': row['row_num'],
            'filled_timestamp': {
                'YY': filled_timestamp['YY'],
                'MM': filled_timestamp['MM'],
                'DD': filled_timestamp['DD'],
                'HH': filled_timestamp['HH'],
                'MN': filled_timestamp['MN'],
                'SS': filled_timestamp['SS']
            },
            'datetime': datetime.datetime(
                filled_timestamp['YY'] + 1900,
                filled_timestamp['MM'],
                filled_timestamp['DD'],
                filled_timestamp['HH'],
                filled_timestamp['MN'],
                filled_timestamp['SS']
            ),
            'filename': row['filename'],
            'Event': row['Event'],
            'Scene': row['Scene'],
            'Location': row['Location'],
            'Tags': tags,
            'Cluster': row['Cluster']
        })
    
    return processed_rows

if __name__ == "__main__":
    # Test with provided test CSV
    test_csv_path = r"C:\Users\bryan\Desktop\test_metadata - Copy.csv"  # Update with actual path
    try:
        results = process_metadata_csv(test_csv_path)
        logging.info(f"Processed {len(results)} rows successfully:")
        for result in results:
            logging.info(f"Row {result['row_num']}: {result['filename']}, "
                        f"Timestamp: {result['datetime'].strftime('%Y:%m:%d %H:%M:%S')}, "
                        f"Event: {result['Event']}, Scene: {result['Scene']}, "
                        f"Cluster: {result['Cluster']}, Tags: {result['Tags']}")
    except Exception as e:
        logging.error(f"Error processing CSV: {e}")